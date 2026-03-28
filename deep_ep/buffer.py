import os
import sys
import time
import torch
import torch.distributed as dist
from typing import Callable, List, Tuple, Optional, Union
from collections import defaultdict

# noinspection PyUnresolvedReferences
import deep_ep_cpp
# noinspection PyUnresolvedReferences
from deep_ep_cpp import Config, EventHandle
from .utils import EventOverlap, check_nvlink_connections


# ── Low-latency profiling helper ────────────────────────────────────
# Enabled by DEEPEP_PROFILE=1 environment variable.
# Inserts torch.cuda.synchronize() between each phase (adds overhead!)
# and prints averaged breakdown every _PROFILE_PRINT_EVERY calls.
_PROFILE_ENABLED = os.environ.get('DEEPEP_PROFILE', '0') == '1'
_PROFILE_PRINT_EVERY = int(os.environ.get('DEEPEP_PROFILE_INTERVAL', '20'))
_PROFILE_WARMUP = int(os.environ.get('DEEPEP_PROFILE_WARMUP', '5'))
_LL_PIPELINE_ENABLED = os.environ.get('DEEPEP_LL_PIPELINE', '0') == '1'
_COOP_ENABLED = os.environ.get('DEEPEP_COOP', '0') == '1'


class _LLProfiler:
    """Accumulates wall-clock timings (with cuda sync) for LL phases."""

    def __init__(self, name: str, rank: int):
        self.name = name
        self.rank = rank
        self.timestamps = {}
        self.accum = defaultdict(float)  # phase_name -> total_us
        self.count = 0
        self.warmup_remaining = _PROFILE_WARMUP

    def mark(self, label: str):
        """Record a wall-clock timestamp AFTER synchronizing the GPU."""
        torch.cuda.synchronize()
        self.timestamps[label] = time.perf_counter_ns()

    def finish_iter(self, extra_info: str = ''):
        """Compute per-phase deltas and accumulate."""
        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
            self.timestamps.clear()
            return

        labels = list(self.timestamps.keys())
        times = list(self.timestamps.values())
        for i in range(1, len(labels)):
            phase = f'{labels[i-1]}->{labels[i]}'
            delta_us = (times[i] - times[i-1]) / 1000.0
            self.accum[phase] += delta_us
        if len(labels) >= 2:
            total_us = (times[-1] - times[0]) / 1000.0
            self.accum['TOTAL'] += total_us
        self.count += 1
        self.timestamps.clear()

        if self.count >= _PROFILE_PRINT_EVERY:
            self._print(extra_info)
            self.accum.clear()
            self.count = 0

    def _print(self, extra_info: str = ''):
        n = self.count
        if n == 0:
            return
        parts = []
        for phase, total_us in self.accum.items():
            parts.append(f'{phase}={total_us/n:.1f}us')
        msg = f'[LL_PROFILE rank={self.rank}] {self.name} (avg of {n}): {" | ".join(parts)}'
        if extra_info:
            msg += f' | {extra_info}'
        print(msg, file=sys.stderr, flush=True)


class Buffer:
    """
    The core expert-parallel (EP) communication buffers for Mixture of Experts (MoE) model, which supports:
        - high-throughput intranode all-to-all (dispatch and combine, using NVLink)
        - high-throughput internode all-to-all (dispatch and combine, using RDMA and NVLink)
        - low-latency all-to-all (dispatch and combine, using RDMA)

    Attributes:
        num_sms: the SMs used in high-throughput kernels.
        rank: the local rank number.
        group_size: the number of ranks in the group.
        group: the communication group.
        num_nvl_bytes: the buffer size for intranode NVLink communication.
        num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
        runtime: the C++ runtime.
    """

    num_sms: int = 20

    def __init__(self,
                 group: Optional[dist.ProcessGroup],
                 num_nvl_bytes: int = 0,
                 num_rdma_bytes: int = 0,
                 low_latency_mode: bool = False,
                 num_qps_per_rank: int = 24,
                 allow_nvlink_for_low_latency_mode: bool = True,
                 allow_mnnvl: bool = False,
                 use_fabric: bool = False,
                 explicitly_destroy: bool = False,
                 enable_shrink: bool = False,
                 comm: Optional["mpi4py.MPI.Comm"] = None) -> None:  # noqa: F821
        """
        Initialize the communication buffer.

        Arguments:
            group: the communication group.
            num_nvl_bytes: the buffer size for intranode NVLink communication.
            num_rdma_bytes: the buffer size for internode (also for intranode with low-latency mode) RDMA communication.
            low_latency_mode: whether to enable low-latency mode.
            num_qps_per_rank: the number of QPs for RDMA, the low-latency mode requires that this number equals
                to the number of local experts.
            allow_nvlink_for_low_latency_mode: whether allow NVLink traffic for low-latency mode, you should notice
                this is somehow incompatible with the hook-based overlapping.
                Warning: PCIe connections may lead to errors due to memory ordering issues,
                please make sure all connections are via NVLink.
            allow_mnnvl: whether to allow MNNVL
            use_fabric: whether to use fabric API for memory buffers.
            enable_shrink: whether to enable shrink mode. The enable mode allocates a mask buffer to support masking ranks dynamically.
            explicitly_destroy: If this flag is set to True, you need to explicitly call `destroy()` to release resources;
                otherwise, the resources will be released by the destructor.
                Note: Releasing resources in the destructor may cause Python's exception handling process to hang.
            comm: the `mpi4py.MPI.Comm` communicator to use in case the group parameter is absent.
        """
        check_nvlink_connections(group)

        # Initialize the CPP runtime
        if group is not None:
            self.rank = group.rank()
            self.group = group
            self.group_size = group.size()

            # If the group uses a CPU-only backend (e.g. gloo), GPU tensor
            # collectives (all_to_all_single, all_to_all, all_reduce) will fail.
            # Detect this and lazily create an NCCL sub-group for GPU ops.
            self._nccl_group = None  # created lazily in _get_gpu_group()

            def all_gather_object(obj):
                object_list = [None] * self.group_size
                dist.all_gather_object(object_list, obj, group)
                return object_list
        elif comm is not None:
            self.rank = comm.Get_rank()
            self.group = comm
            self.group_size = comm.Get_size()
            self._nccl_group = None  # not applicable for mpi4py

            def all_gather_object(obj):
                return comm.allgather(obj)
        else:
            raise ValueError("Either 'group' or 'comm' must be provided.")
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = low_latency_mode
        self.explicitly_destroy = explicitly_destroy
        self.enable_shrink = enable_shrink
        self.runtime = deep_ep_cpp.Buffer(self.rank, self.group_size, num_nvl_bytes, num_rdma_bytes, low_latency_mode, explicitly_destroy,
                                          enable_shrink, use_fabric)

        # Synchronize device IDs
        local_device_id = self.runtime.get_local_device_id()
        device_ids = all_gather_object(local_device_id)

        # Synchronize IPC handles
        local_ipc_handle = self.runtime.get_local_ipc_handle()
        ipc_handles = all_gather_object(local_ipc_handle)

        # Synchronize NVSHMEM unique IDs / EFA address exchange
        root_unique_id = None
        if self.runtime.get_num_rdma_ranks() > 1 or low_latency_mode:
            # In EFA mode, we don't use NVSHMEM. The RDMA buffer is allocated
            # via cudaMalloc and the EFA transport is initialized separately.
            # We pass a dummy unique_id to satisfy the sync() API contract.
            # The actual EFA address exchange will happen after sync().
            root_unique_id = bytearray(128)  # Dummy 128-byte ID

        # Make CPP runtime available
        self.runtime.sync(device_ids, ipc_handles, root_unique_id)
        assert self.runtime.is_available()

        # Initialize EFA transport for internode RDMA
        if self.runtime.get_num_rdma_ranks() > 1 and num_rdma_bytes > 0:
            self._init_efa()

    def _init_efa(self):
        """Initialize EFA transport: create endpoint, register RDMA buffer, exchange addresses."""
        def allgather_fn(local_data_bytes):
            """Python allgather wrapper for C++ EFA transport."""
            local_bytes = bytes(local_data_bytes)
            all_data = [None] * self.group_size
            if isinstance(self.group, dist.ProcessGroup):
                dist.all_gather_object(all_data, local_bytes, self.group)
            else:
                all_data = self.group.allgather(local_bytes)
            return all_data

        self.runtime.init_efa(allgather_fn)
        self._efa_initialized = True

        # Cache immutable internode metadata to avoid recomputation every transfer
        num_local_ranks = 8
        my_node = self.rank // num_local_ranks
        inter_ranks_set = set()
        inter_ranks_list = []
        intra_ranks_list = []
        for r in range(self.group_size):
            if r // num_local_ranks != my_node:
                inter_ranks_set.add(r)
                inter_ranks_list.append(r)
            else:
                intra_ranks_list.append(r)
        self._my_node = my_node
        self._num_local_ranks = num_local_ranks
        self._inter_ranks_set = inter_ranks_set
        self._inter_ranks_list = inter_ranks_list
        self._intra_ranks_list = intra_ranks_list
        self._has_inter = len(inter_ranks_list) > 0
        self._half_rdma = self.num_rdma_bytes // 2

    def _get_gpu_group(self) -> dist.ProcessGroup:
        """Return a process group suitable for GPU tensor collectives.

        If ``self.group`` already supports NCCL operations (i.e. it is an NCCL
        or NCCL-capable backend), return it directly.  Otherwise, lazily create
        a matching NCCL sub-group covering the same global ranks and cache it
        in ``self._nccl_group``.

        The gloo group (``self.group``) is still used for CPU object collectives
        like ``all_gather_object`` and ``barrier``.
        """
        # Fast path: mpi4py comm or already NCCL
        if not isinstance(self.group, dist.ProcessGroup):
            return self.group  # mpi4py — caller handles separately

        if self._nccl_group is not None:
            return self._nccl_group

        # Check if the existing group already supports GPU collectives
        backend_name = dist.get_backend(self.group)
        if backend_name == "nccl":
            self._nccl_group = self.group
            return self._nccl_group

        # Need to create an NCCL group.  Determine the global ranks in
        # self.group so we can create a matching NCCL ProcessGroup.
        # Use dist.get_process_group_ranks() (available since PyTorch 1.13).
        global_ranks = dist.get_process_group_ranks(self.group)
        if len(global_ranks) == 0:
            # Fallback: assume group covers world
            global_ranks = list(range(self.group_size))

        self._nccl_group = dist.new_group(ranks=global_ranks, backend="nccl")
        if self.rank == 0:
            print(f"[DeepEP-EFA] Created NCCL sub-group for GPU collectives "
                  f"(original backend: {backend_name}, ranks: {global_ranks})",
                  file=sys.stderr, flush=True)
        return self._nccl_group

    def destroy(self):
        """
        Destroy the cpp runtime and release resources.

        """

        assert self.explicitly_destroy, '`explicitly_destroy` flag must be set'

        self.runtime.destroy()
        self.runtime = None

    @staticmethod
    def is_sm90_compiled():
        return deep_ep_cpp.is_sm90_compiled()

    @staticmethod
    def set_num_sms(new_num_sms: int) -> None:
        """
        Set the number of SMs to use in high-throughput kernels.

        Arguments:
            new_num_sms: the new number to be set.
        """

        assert new_num_sms % 2 == 0, 'The SM count must be even'
        Buffer.num_sms = new_num_sms

    @staticmethod
    def capture() -> EventOverlap:
        """
        Capture a CUDA event on the current stream, i.e. `torch.cuda.current_stream()`.

        Returns:
            event: the captured event.
        """
        return EventOverlap(EventHandle())

    @staticmethod
    def get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank: int, hidden: int, num_ranks: int, num_experts: int) -> int:
        """
        Get a minimum size requirement for the RDMA buffer. The size calculation will be done with BF16.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_ranks: the number of EP group ranks.
            num_experts: the number of all experts.

        Returns:
            size: the RDMA buffer size recommended.
        """
        return deep_ep_cpp.get_low_latency_rdma_size_hint(num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts)

    def get_comm_stream(self) -> torch.Stream:
        """
        Get the communication stream.

        Returns:
            stream: the communication stream.
        """
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(stream_id=ts.stream_id, device_index=ts.device_index, device_type=ts.device_type)

    def get_local_buffer_tensor(self,
                                dtype: torch.dtype,
                                size: Optional[torch.Size] = None,
                                offset: int = 0,
                                use_rdma_buffer: bool = False) -> torch.Tensor:
        """
        Get the raw buffer (slice supported) as a PyTorch tensor.

        Argument:
            dtype: the data type (PyTorch `dtype`) for the tensor.
            size: the slice size (by elements) to get from the buffer.
            offset: the offset of the beginning element.
            use_rdma_buffer: whether to return the RDMA buffer.
        """
        tensor = self.runtime.get_local_buffer_tensor(dtype, offset, use_rdma_buffer)
        if size is None:
            return tensor

        assert tensor.numel() >= size.numel()
        return tensor[:size.numel()].view(size)

    @staticmethod
    def _unpack_bias(bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        bias_0, bias_1 = None, None
        if isinstance(bias, torch.Tensor):
            bias_0 = bias
        elif isinstance(bias, tuple):
            assert len(bias) == 2
            bias_0, bias_1 = bias
        return bias_0, bias_1

    @staticmethod
    def get_dispatch_config(num_ranks: int) -> Config:
        """
        Get a recommended dispatch config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 24, 256, 6, 128),
            4: Config(Buffer.num_sms, 6, 256, 6, 128),
            8: Config(Buffer.num_sms, 6, 256, 6, 128),
            16: Config(Buffer.num_sms, 36, 288, 20, 128),
            24: Config(Buffer.num_sms, 32, 288, 8, 128),
            32: Config(Buffer.num_sms, 32, 288, 8, 128),
            48: Config(Buffer.num_sms, 32, 288, 8, 128),
            64: Config(Buffer.num_sms, 32, 288, 8, 128),
            96: Config(Buffer.num_sms, 20, 480, 12, 128),
            128: Config(Buffer.num_sms, 20, 560, 12, 128),
            144: Config(Buffer.num_sms, 32, 720, 12, 128),
            160: Config(Buffer.num_sms, 28, 720, 12, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    @staticmethod
    def get_combine_config(num_ranks: int) -> Config:
        """
        Get a recommended combine config.

        Argument:
            num_ranks: the number of ranks.

        Returns:
            config: the recommended config.
        """

        # TODO: automatically tune
        config_map = {
            2: Config(Buffer.num_sms, 10, 256, 6, 128),
            4: Config(Buffer.num_sms, 9, 256, 6, 128),
            8: Config(Buffer.num_sms, 4, 256, 6, 128),
            16: Config(Buffer.num_sms, 4, 288, 12, 128),
            24: Config(Buffer.num_sms, 1, 288, 8, 128),
            32: Config(Buffer.num_sms, 1, 288, 8, 128),
            48: Config(Buffer.num_sms, 1, 288, 8, 128),
            64: Config(Buffer.num_sms, 1, 288, 8, 128),
            96: Config(Buffer.num_sms, 1, 480, 8, 128),
            128: Config(Buffer.num_sms, 1, 560, 8, 128),
            144: Config(Buffer.num_sms, 2, 720, 8, 128),
            160: Config(Buffer.num_sms, 2, 720, 8, 128),
        }
        assert num_ranks in config_map, f'Unsupported number of EP ranks: {num_ranks}'
        return config_map[num_ranks]

    # noinspection PyTypeChecker
    def get_dispatch_layout(self, topk_idx: torch.Tensor, num_experts: int,
                            previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                            allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, EventOverlap]:
        """
        Calculate the layout required for later communication.

        Arguments:
            topk_idx: `[num_tokens, num_topk]`, dtype must be `deep_ep.topk_idx_t` (typically `torch.int64`), the expert
                indices selected by each token, `-1` means no selections.
            num_experts: the number of experts.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event = \
            self.runtime.get_dispatch_layout(topk_idx, num_experts, getattr(previous_event, 'event', None),
                                             async_finish, allocate_on_comm_stream)
        return num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, EventOverlap(event)

    # noinspection PyTypeChecker
    def dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 handle: Optional[Tuple] = None,
                 num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                 is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                 topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None,
                 expert_alignment: int = 1, num_worst_tokens: int = 0,
                 config: Optional[Config] = None,
                 previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                 allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
                  Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Dispatch tokens to different ranks, both intranode and internode settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `torch.Tensor` or tuple of `torch.Tensor`, for the first type, the shape must be `[num_tokens, hidden]`,
                and type must be `torch.bfloat16`; for the second type, the first element of the tuple must be shaped as
                `[num_tokens, hidden]` with type `torch.float8_e4m3fn`, the second must be `[num_tokens, hidden // 128]`
                 (requiring divisible) with type `torch.float`.
            handle: an optional communication handle, if set, the CPU will reuse the layout information to save some time.
            num_tokens_per_rank: `[num_ranks]` with `torch.int`, the number of tokens to be sent to each rank.
            num_tokens_per_rdma_rank: `[num_rdma_ranks]` with `torch.int`, the number of tokens to be sent to each RDMA
                rank (with the same GPU index), return `None` for intranode settings.
            is_token_in_rank: `[num_tokens, num_ranks]` with `torch.bool`, whether a token be sent to a rank.
            num_tokens_per_expert: `[num_experts]` with `torch.int`, the number of tokens to be sent to each expert.
            topk_idx: `[num_tokens, num_topk]` with `deep_ep.topk_idx_t` (typically `torch.int64`), the expert indices
                selected by each token, `-1` means no selections.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the expert weights of each token to dispatch.
            expert_alignment: align the number of tokens received by each local expert to this variable.
            num_worst_tokens: the worst number of tokens to receive, if specified, there will be no CPU sync, and it
                will be CUDA-graph compatible. Please also notice that this flag is for intranode only.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: received tokens, the same type and tuple as the input `x`, but the number of tokens equals to the
                received token count.
            recv_topk_idx: received expert indices.
            recv_topk_weights: received expert weights.
            num_recv_tokens_per_expert_list: Python list shaped `[num_local_experts]`, the received token count by
                each local expert, aligned to the input `expert_alignment`. If `num_worst_tokens` is specified, the list
                will be empty.
            handle: the returned communication handle.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_dispatch_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_dispatch(x, handle, num_tokens_per_rank, num_tokens_per_rdma_rank, is_token_in_rank,
                                           num_tokens_per_expert, topk_idx, topk_weights, expert_alignment, num_worst_tokens, config,
                                           previous_event, async_finish, allocate_on_comm_stream)

        # Launch the kernel with cached or non-cached mode
        x, x_scales = x if isinstance(x, tuple) else (x, None)
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head = handle
            num_recv_tokens = recv_src_idx.size(0)
            recv_x, recv_x_scales, _, _, _, _, _, _, _, _, event = self.runtime.intranode_dispatch(
                x, x_scales, None, None, None, is_token_in_rank, None, num_recv_tokens, rank_prefix_matrix, channel_prefix_matrix,
                expert_alignment, num_worst_tokens, config, getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            return (recv_x, recv_x_scales) if x_scales is not None else recv_x, None, None, None, None, EventOverlap(event)
        else:
            assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
            recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event = \
                self.runtime.intranode_dispatch(x, x_scales, topk_idx, topk_weights,
                                                num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert, 0, None, None,
                                                expert_alignment, num_worst_tokens, config,
                                                getattr(previous_event, 'event', None), async_finish, allocate_on_comm_stream)
            handle = (rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, is_token_in_rank, send_head)
            return (
                recv_x, recv_x_scales
            ) if x_scales is not None else recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(
                event)

    # noinspection PyTypeChecker
    def combine(self, x: torch.Tensor, handle: Tuple,
                topk_weights: Optional[torch.Tensor] = None,
                bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                config: Optional[Config] = None,
                previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Combine (reduce) tokens (addition **without** weights) from different ranks, both intranode and internode
            settings are supported.
        Intranode kernels require all the ranks should be visible via NVLink.
        Internode kernels require the ranks in a node should be visible via NVLink, while the ranks with the same GPU
            index should be visible via RDMA.

        Arguments:
            x: `[num_tokens, hidden]` with `torch.bfloat16`, the tokens to send for reducing to its original ranks.
            handle: a must-set communication handle, you can obtain this from the dispatch function.
            topk_weights: `[num_tokens, num_topk]` with `torch.float`, the tokens' top-k weights for reducing to its original ranks.
            bias: 0, 1 or 2 `[num_tokens, hidden]` with `torch.bfloat16` final bias to the output.
            config: the performance tuning config.
            previous_event: the event to wait before actually executing the kernel.
            async_finish: the current stream will not wait for the communication kernels to be finished if set.
            allocate_on_comm_stream: control whether all the allocated tensors' ownership to be on the communication stream.

        Returns:
            recv_x: the reduced token from its dispatched ranks.
            recv_topk_weights: the reduced top-k weights from its dispatch ranks.
            event: the event after executing the kernel (valid only if `async_finish` is set).
        """
        # Default config
        config = self.get_combine_config(self.group_size) if config is None else config

        # Internode
        if self.runtime.get_num_rdma_ranks() > 1:
            return self.internode_combine(x, handle, topk_weights, bias, config, previous_event, async_finish, allocate_on_comm_stream)

        # NOTES: the second `_` is for the sending side, so we should use the third one
        rank_prefix_matrix, _, channel_prefix_matrix, src_idx, is_recv_token_in_rank, send_head = handle
        bias_0, bias_1 = Buffer._unpack_bias(bias)

        # Launch the kernel
        recv_x, recv_topk_weights, event = self.runtime.intranode_combine(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix,
                                                                          channel_prefix_matrix, send_head, config,
                                                                          getattr(previous_event, 'event',
                                                                                  None), async_finish, allocate_on_comm_stream)
        return recv_x, recv_topk_weights, EventOverlap(event)

    # noinspection PyTypeChecker
    def internode_dispatch(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                           handle: Optional[Tuple] = None,
                           num_tokens_per_rank: Optional[torch.Tensor] = None, num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
                           is_token_in_rank: Optional[torch.Tensor] = None, num_tokens_per_expert: Optional[torch.Tensor] = None,
                           topk_idx: Optional[torch.Tensor] = None, topk_weights: Optional[torch.Tensor] = None, expert_alignment: int = 1,
                           num_worst_tokens: int = 0, config: Optional[Config] = None,
                           previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                           allocate_on_comm_stream: bool = False) -> \
            Tuple[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Optional[torch.Tensor],
            Optional[torch.Tensor], List[int], Tuple, EventOverlap]:
        """
        Optimized EFA internode dispatch — vectorized Python loops.
        """
        assert config is not None
        rank = self.rank
        num_ranks = self.group_size
        num_local_ranks = 8
        num_rdma_ranks = num_ranks // num_local_ranks
        num_experts_total = None

        x_orig = x
        x_data, x_scales = x if isinstance(x, tuple) else (x, None)
        is_fp8 = x_scales is not None

        num_tokens = x_data.size(0)
        hidden = x_data.size(1)

        # ====================================================================
        # Cached dispatch mode
        # ====================================================================
        if handle is not None:
            assert topk_idx is None and topk_weights is None
            is_token_in_rank = handle[0]
            efa_info = self._efa_dispatch_info
            recv_gbl_rank_prefix_sum = efa_info['recv_gbl_rank_prefix_sum']
            recv_rdma_rank_prefix_sum = efa_info['recv_rdma_rank_prefix_sum']
            send_counts = efa_info['send_counts']
            recv_counts = efa_info['recv_counts']
            send_token_indices = efa_info['send_token_indices']
            recv_src_meta = efa_info['recv_src_meta']
            num_recv_tokens = efa_info['num_recv_tokens']
            flat_sorted_token_ids = efa_info['flat_sorted_token_ids']

            recv_x_data, recv_x_scales = self._efa_do_dispatch_data(
                x_data, x_scales, None, None, is_token_in_rank,
                send_counts, recv_counts, send_token_indices,
                num_recv_tokens, num_ranks, hidden,
                flat_sorted_token_ids=flat_sorted_token_ids)

            if is_fp8:
                recv_x_result = (recv_x_data, recv_x_scales)
            else:
                recv_x_result = recv_x_data
            return recv_x_result, None, None, None, None, EventOverlap(EventHandle())

        # ====================================================================
        # Fresh dispatch
        # ====================================================================
        assert num_tokens_per_rank is not None and is_token_in_rank is not None and num_tokens_per_expert is not None
        num_experts_total = num_tokens_per_expert.size(0)
        num_local_experts = num_experts_total // num_ranks

        # ------------------------------------------------------------------
        # Iter 26: Pre-allocate dispatch metadata buffers
        # ------------------------------------------------------------------
        dispatch_meta_key = (num_ranks, num_experts_total, num_tokens, num_tokens_per_rank.dtype)
        if not hasattr(self, '_dispatch_meta_cache') or self._dispatch_meta_cache_key != dispatch_meta_key:
            self._dispatch_meta_cache = {
                'recv_counts_tensor': torch.empty(num_ranks, dtype=num_tokens_per_rank.dtype, device='cuda'),
                'gbl_num_tokens_per_expert': torch.empty(num_experts_total, dtype=num_tokens_per_expert.dtype, device='cuda'),
                'recv_cumsum_for_meta': torch.zeros(num_ranks + 1, dtype=torch.int64, device='cuda'),
            }
            self._dispatch_meta_cache_key = dispatch_meta_key
        dmc = self._dispatch_meta_cache

        # ------------------------------------------------------------------
        # Step 1: Exchange token counts — single .tolist() for both
        # ------------------------------------------------------------------
        send_counts_tensor = num_tokens_per_rank.clone()
        recv_counts_tensor = dmc['recv_counts_tensor']
        dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self._get_gpu_group())

        all_counts = torch.cat([send_counts_tensor, recv_counts_tensor]).tolist()
        send_counts = all_counts[:num_ranks]
        recv_counts = all_counts[num_ranks:]

        # ------------------------------------------------------------------
        # Step 2: Prefix sums — vectorized (no Python loop)
        # ------------------------------------------------------------------
        recv_gbl_rank_prefix_sum = torch.cumsum(recv_counts_tensor, dim=0).to(torch.int32)
        num_recv_tokens = int(recv_gbl_rank_prefix_sum[-1].item())

        # RDMA rank prefix sum: reshape and sum per group instead of Python loop
        recv_rdma_counts = recv_counts_tensor.view(num_rdma_ranks, num_local_ranks).sum(dim=1).to(torch.int32)
        recv_rdma_rank_prefix_sum = torch.cumsum(recv_rdma_counts, dim=0).to(torch.int32)

        # ------------------------------------------------------------------
        # Step 3: Expert count exchange — single .tolist() instead of per-expert .item()
        # Iter 26: Reuse pre-allocated buffer
        # ------------------------------------------------------------------
        gbl_num_tokens_per_expert = dmc['gbl_num_tokens_per_expert']
        gbl_num_tokens_per_expert.copy_(num_tokens_per_expert)
        dist.all_reduce(gbl_num_tokens_per_expert, group=self._get_gpu_group())
        local_expert_start = rank * num_local_experts
        local_expert_end = local_expert_start + num_local_experts

        local_expert_counts = gbl_num_tokens_per_expert[local_expert_start:local_expert_end]
        if expert_alignment > 1:
            local_expert_counts = ((local_expert_counts + expert_alignment - 1) // expert_alignment) * expert_alignment
        num_recv_tokens_per_expert_list = local_expert_counts.tolist()

        # ------------------------------------------------------------------
        # Step 4: Build per-destination send token lists — FUSED CUDA KERNEL
        # Single kernel replaces: nonzero + argsort + bincount + cumsum
        # ------------------------------------------------------------------
        flat_sorted_token_ids, send_counts_i32, send_cumsum, total_send_val = \
            deep_ep_cpp.moe_routing_sort(is_token_in_rank)

        # Build send_token_indices as list of views into the flat sorted array
        send_cumsum_cpu = send_cumsum.tolist()
        send_token_indices = []
        for dst in range(num_ranks):
            start = send_cumsum_cpu[dst]
            end = send_cumsum_cpu[dst + 1]
            send_token_indices.append(flat_sorted_token_ids[start:end])

        # ------------------------------------------------------------------
        # Step 5: Dispatch data
        # ------------------------------------------------------------------
        recv_x_data, recv_x_scales = self._efa_do_dispatch_data(
            x_data, x_scales, topk_idx, topk_weights, is_token_in_rank,
            send_counts, recv_counts, send_token_indices,
            num_recv_tokens, num_ranks, hidden,
            flat_sorted_token_ids=flat_sorted_token_ids)

        # ------------------------------------------------------------------
        # Step 6: All-to-all topk_idx and topk_weights — FUSED CUDA KERNEL
        # Iter 26: Pre-allocated topk packed buffers
        # ------------------------------------------------------------------
        recv_topk_idx = None
        recv_topk_weights = None
        if topk_idx is not None:
            num_topk = topk_idx.size(1)

            # Fused topk remap kernel — replaces repeat_interleave + broadcasting + where
            remapped_topk, remapped_weights = deep_ep_cpp.topk_remap(
                topk_idx, topk_weights, flat_sorted_token_ids, send_cumsum,
                total_send_val, num_ranks, num_local_experts)

            total_recv = num_recv_tokens

            # Pack topk_idx (int64, 8B) + topk_weights (float32, 4B) into single
            # uint8 buffer for ONE NCCL all_to_all_single instead of TWO.
            # Per-token packed size: num_topk * (8 + 4) = num_topk * 12 bytes
            topk_idx_bytes = num_topk * 8   # int64
            topk_w_bytes = num_topk * 4     # float32
            packed_topk_bpt = topk_idx_bytes + topk_w_bytes
            total_send_topk = sum(send_counts)

            # Iter 26: Pre-allocate topk packed buffers
            topk_cache_key = (num_topk, num_tokens, num_ranks)
            if not hasattr(self, '_topk_cache') or self._topk_cache_key != topk_cache_key:
                max_topk_tokens = num_tokens * num_topk
                self._topk_cache = {
                    'send_packed': torch.empty(max_topk_tokens * packed_topk_bpt, dtype=torch.uint8, device='cuda'),
                    'recv_packed': torch.empty(max(1, max_topk_tokens) * packed_topk_bpt, dtype=torch.uint8, device='cuda'),
                    'max_topk_tokens': max_topk_tokens,
                }
                self._topk_cache_key = topk_cache_key

            tc = self._topk_cache
            if total_send_topk <= tc['max_topk_tokens']:
                send_topk_packed = tc['send_packed'][:total_send_topk * packed_topk_bpt]
            else:
                send_topk_packed = torch.empty(total_send_topk * packed_topk_bpt, dtype=torch.uint8, device='cuda')

            if total_send_topk > 0:
                send_topk_view = send_topk_packed.view(total_send_topk, packed_topk_bpt)
                send_topk_view[:, :topk_idx_bytes] = remapped_topk.contiguous().view(torch.uint8).view(total_send_topk, topk_idx_bytes)
                send_topk_view[:, topk_idx_bytes:] = remapped_weights.contiguous().view(torch.uint8).view(total_send_topk, topk_w_bytes)

            if total_recv <= tc['max_topk_tokens']:
                recv_topk_packed = tc['recv_packed'][:max(1, total_recv) * packed_topk_bpt]
            else:
                recv_topk_packed = torch.empty(max(1, total_recv) * packed_topk_bpt, dtype=torch.uint8, device='cuda')

            send_splits = [s * packed_topk_bpt for s in send_counts]
            recv_splits = [r * packed_topk_bpt for r in recv_counts]
            dist.all_to_all_single(
                recv_topk_packed[:total_recv * packed_topk_bpt] if total_recv > 0 else recv_topk_packed[:0],
                send_topk_packed,
                output_split_sizes=recv_splits, input_split_sizes=send_splits,
                group=self._get_gpu_group())

            # Unpack
            if total_recv > 0:
                recv_topk_view = recv_topk_packed[:total_recv * packed_topk_bpt].view(total_recv, packed_topk_bpt)
                recv_topk_idx = recv_topk_view[:, :topk_idx_bytes].contiguous().view(torch.int64).view(total_recv, num_topk)
                recv_topk_weights = recv_topk_view[:, topk_idx_bytes:].contiguous().view(torch.float32).view(total_recv, num_topk)
            else:
                recv_topk_idx = torch.empty((0, num_topk), dtype=topk_idx.dtype, device='cuda')
                recv_topk_weights = torch.empty((0, num_topk), dtype=torch.float32, device='cuda')

        # ------------------------------------------------------------------
        # Step 7: Build recv_src_meta — FUSED CUDA KERNEL
        # Iter 26: Reuse pre-allocated recv_cumsum_for_meta buffer
        # ------------------------------------------------------------------
        recv_cumsum_for_meta = dmc['recv_cumsum_for_meta']
        recv_cumsum_for_meta[0] = 0
        recv_cumsum_for_meta[1:] = torch.cumsum(recv_counts_tensor, dim=0).to(torch.int64)
        recv_src_meta = deep_ep_cpp.build_recv_src_meta(
            recv_counts_tensor, recv_cumsum_for_meta,
            num_recv_tokens, num_ranks, num_local_ranks)

        # ------------------------------------------------------------------
        # Step 8: Handle num_worst_tokens
        # ------------------------------------------------------------------
        if num_worst_tokens > 0:
            if num_recv_tokens < num_worst_tokens:
                pad_n = num_worst_tokens - num_recv_tokens
                recv_x_data = torch.cat([recv_x_data, torch.zeros((pad_n, hidden), dtype=recv_x_data.dtype, device='cuda')], dim=0)
                if recv_x_scales is not None:
                    num_scale_cols = recv_x_scales.size(1)
                    # Iter 26: recv_x_scales is already row-major contiguous, so cat result is too
                    recv_x_scales = torch.cat([recv_x_scales.contiguous(),
                                               torch.zeros((pad_n, num_scale_cols), dtype=recv_x_scales.dtype, device='cuda')], dim=0)
                if recv_topk_idx is not None:
                    num_topk = recv_topk_idx.size(1)
                    recv_topk_idx = torch.cat([recv_topk_idx, torch.full((pad_n, num_topk), -1, dtype=recv_topk_idx.dtype, device='cuda')], dim=0)
                    recv_topk_weights = torch.cat([recv_topk_weights, torch.zeros((pad_n, num_topk), dtype=torch.float32, device='cuda')], dim=0)
                recv_src_meta = torch.cat([recv_src_meta, torch.zeros((pad_n, 8), dtype=torch.uint8, device='cuda')], dim=0)
            num_recv_tokens_per_expert_list = []

        # ------------------------------------------------------------------
        # Step 9: Build handle
        # Iter 22: Reuse cached dummy tensors for handle matrices (never used in EFA path)
        # ------------------------------------------------------------------
        num_channels = config.num_sms // 2
        if not hasattr(self, '_handle_cache') or self._handle_cache_key != (num_rdma_ranks, num_ranks, num_channels, num_tokens):
            self._handle_cache = {
                'rdma_ch': torch.zeros((num_rdma_ranks, num_channels), dtype=torch.int32, device='cuda'),
                'gbl_ch': torch.zeros((num_ranks, num_channels), dtype=torch.int32, device='cuda'),
                'recv_rdma_ch': torch.zeros((num_rdma_ranks, num_channels), dtype=torch.int32, device='cuda'),
                'recv_gbl_ch': torch.zeros((num_ranks, num_channels), dtype=torch.int32, device='cuda'),
                'send_rdma_head': torch.zeros((num_tokens, num_rdma_ranks), dtype=torch.int32, device='cuda'),
            }
            self._handle_cache_key = (num_rdma_ranks, num_ranks, num_channels, num_tokens)

        hc = self._handle_cache
        rdma_channel_prefix_matrix = hc['rdma_ch']
        gbl_channel_prefix_matrix = hc['gbl_ch']
        recv_rdma_channel_prefix_matrix = hc['recv_rdma_ch']
        recv_gbl_channel_prefix_matrix = hc['recv_gbl_ch']
        send_rdma_head = hc['send_rdma_head']
        send_nvl_head = torch.zeros((int(recv_rdma_rank_prefix_sum[-1].item()) if num_rdma_ranks > 0 else num_recv_tokens,
                                     num_local_ranks), dtype=torch.int32, device='cuda')

        self._efa_dispatch_info = {
            'recv_gbl_rank_prefix_sum': recv_gbl_rank_prefix_sum,
            'recv_rdma_rank_prefix_sum': recv_rdma_rank_prefix_sum,
            'send_counts': send_counts,
            'recv_counts': recv_counts,
            'send_token_indices': send_token_indices,
            'recv_src_meta': recv_src_meta,
            'num_recv_tokens': num_recv_tokens,
            'flat_sorted_token_ids': flat_sorted_token_ids,
        }

        handle = (is_token_in_rank,
                  rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
                  recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
                  recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
                  recv_src_meta, send_rdma_head, send_nvl_head)

        if is_fp8:
            recv_x_result = (recv_x_data, recv_x_scales)
        else:
            recv_x_result = recv_x_data

        return recv_x_result, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, EventOverlap(EventHandle())

    def _efa_do_dispatch_data(self, x_data, x_scales, topk_idx, topk_weights,
                               is_token_in_rank, send_counts, recv_counts,
                               send_token_indices, num_recv_tokens, num_ranks, hidden,
                               flat_sorted_token_ids=None):
        """Helper: pack, all-to-all, and unpack the hidden data (and FP8 scales).
        
        Uses single vectorized gather and batched _efa_transfer_multi for FP8.
        Iter 26: Pre-allocated recv buffers to avoid per-call torch.empty().
        """
        is_fp8 = x_scales is not None
        use_efa = getattr(self, '_efa_initialized', False)

        # Vectorized pack: single gather using pre-sorted flat token indices
        if flat_sorted_token_ids is not None and flat_sorted_token_ids.numel() > 0:
            send_x_flat = x_data[flat_sorted_token_ids].contiguous()
        elif flat_sorted_token_ids is not None:
            send_x_flat = torch.empty((0, hidden), dtype=x_data.dtype, device='cuda')
        else:
            # Fallback for cached dispatch without flat_sorted_token_ids
            send_x_list = []
            for dst in range(num_ranks):
                idx = send_token_indices[dst]
                if idx.numel() > 0:
                    send_x_list.append(x_data[idx].contiguous())
                else:
                    send_x_list.append(torch.empty((0, hidden), dtype=x_data.dtype, device='cuda'))
            send_x_flat = torch.cat(send_x_list, dim=0).contiguous()

        # Iter 26: Pre-allocated dispatch data recv buffers
        num_scale_cols = x_scales.size(1) if is_fp8 else 0
        dispatch_data_key = (hidden, x_data.dtype, is_fp8, num_scale_cols, x_scales.dtype if is_fp8 else None)
        if not hasattr(self, '_dispatch_data_cache') or self._dispatch_data_cache_key != dispatch_data_key:
            # Allocate max-size buffers. For MoE dispatch, num_recv_tokens per rank is
            # bounded by num_tokens * num_topk. With 4096 tokens, topk=8: max ~32768.
            # Use num_tokens * 8 as upper bound (covers topk=8 with all tokens to this rank).
            num_tokens = x_data.size(0)
            max_recv = num_tokens * 8
            self._dispatch_data_cache = {
                'recv_x_max': torch.empty((max_recv, hidden), dtype=x_data.dtype, device='cuda'),
                'recv_scales_max': torch.empty((max_recv, num_scale_cols), dtype=x_scales.dtype, device='cuda') if is_fp8 else None,
                'max_recv': max_recv,
            }
            self._dispatch_data_cache_key = dispatch_data_key

        ddc = self._dispatch_data_cache
        if num_recv_tokens <= ddc['max_recv']:
            recv_x_flat = ddc['recv_x_max'][:num_recv_tokens]
        else:
            recv_x_flat = torch.empty((num_recv_tokens, hidden), dtype=x_data.dtype, device='cuda')

        if use_efa:
            if is_fp8:
                # Batch data + scales into single _efa_transfer_multi call
                if flat_sorted_token_ids is not None and flat_sorted_token_ids.numel() > 0:
                    send_scales_flat = x_scales[flat_sorted_token_ids].contiguous()
                elif flat_sorted_token_ids is not None:
                    send_scales_flat = torch.empty((0, num_scale_cols), dtype=x_scales.dtype, device='cuda')
                else:
                    send_scales_list = []
                    for dst in range(num_ranks):
                        idx = send_token_indices[dst]
                        if idx.numel() > 0:
                            send_scales_list.append(x_scales[idx].contiguous())
                        else:
                            send_scales_list.append(torch.empty((0, num_scale_cols), dtype=x_scales.dtype, device='cuda'))
                    send_scales_flat = torch.cat(send_scales_list, dim=0).contiguous()

                if num_recv_tokens <= ddc['max_recv'] and ddc['recv_scales_max'] is not None:
                    recv_x_scales_out = ddc['recv_scales_max'][:num_recv_tokens]
                else:
                    recv_x_scales_out = torch.empty((num_recv_tokens, num_scale_cols), dtype=x_scales.dtype, device='cuda')

                self._efa_transfer_multi([
                    (send_x_flat, recv_x_flat, hidden * x_data.element_size()),
                    (send_scales_flat, recv_x_scales_out, num_scale_cols * x_scales.element_size()),
                ], send_counts, recv_counts)

                # Iter 26: Removed .T.contiguous().T — recv tensor is already row-major contiguous
                return recv_x_flat, recv_x_scales_out
            else:
                self._efa_transfer(send_x_flat, recv_x_flat, send_counts, recv_counts,
                                   hidden * x_data.element_size())
        else:
            elem_size_hidden = hidden
            send_splits = [s * elem_size_hidden for s in send_counts]
            recv_splits = [r * elem_size_hidden for r in recv_counts]
            dist.all_to_all_single(
                recv_x_flat.view(-1), send_x_flat.view(-1),
                output_split_sizes=recv_splits, input_split_sizes=send_splits,
                group=self._get_gpu_group())

        recv_x_scales_out = None
        if is_fp8:
            # Vectorized scales gather
            if flat_sorted_token_ids is not None and flat_sorted_token_ids.numel() > 0:
                send_scales_flat = x_scales[flat_sorted_token_ids].contiguous()
            elif flat_sorted_token_ids is not None:
                send_scales_flat = torch.empty((0, num_scale_cols), dtype=x_scales.dtype, device='cuda')
            else:
                send_scales_list = []
                for dst in range(num_ranks):
                    idx = send_token_indices[dst]
                    if idx.numel() > 0:
                        send_scales_list.append(x_scales[idx].contiguous())
                    else:
                        send_scales_list.append(torch.empty((0, num_scale_cols), dtype=x_scales.dtype, device='cuda'))
                send_scales_flat = torch.cat(send_scales_list, dim=0).contiguous()

            if num_recv_tokens <= ddc['max_recv'] and ddc['recv_scales_max'] is not None:
                recv_x_scales_out = ddc['recv_scales_max'][:num_recv_tokens]
            else:
                recv_x_scales_out = torch.empty((num_recv_tokens, num_scale_cols), dtype=x_scales.dtype, device='cuda')

            # Non-EFA path: separate all_to_all for scales
            send_s_splits = [s * num_scale_cols for s in send_counts]
            recv_s_splits = [r * num_scale_cols for r in recv_counts]
            dist.all_to_all_single(
                recv_x_scales_out.view(-1), send_scales_flat.view(-1),
                output_split_sizes=recv_s_splits, input_split_sizes=send_s_splits,
                group=self._get_gpu_group())

            # Iter 26: Removed .T.contiguous().T — recv tensor is already row-major contiguous

        return recv_x_flat, recv_x_scales_out

    def _print_timing(self, _t, inter_send_bytes, inter_recv_bytes):
        """Print timing breakdown for _efa_transfer. Only call from rank 0."""
        def _ms(a, b):
            if a in _t and b in _t:
                return (_t[b] - _t[a]) * 1000.0
            return 0.0

        total = _ms('start', 'end')
        python_setup = _ms('start', 'after_python_setup')
        intra_pack = _ms('after_python_setup', 'after_intra_pack')
        nccl_a2a = _ms('after_intra_pack', 'after_nccl_a2a')
        intra_unpack = _ms('after_nccl_a2a', 'after_intra_unpack')
        recv_zero = _ms('after_intra_unpack', 'after_recv_zero')
        inter_pack = _ms('after_recv_zero', 'after_inter_pack')
        efa_offset_build = _ms('after_inter_pack', 'after_efa_offset_build')
        offset_exchange = _ms('after_efa_offset_build', 'after_offset_exchange')
        efa_rdma = _ms('after_offset_exchange', 'after_efa_rdma')
        barrier = _ms('after_efa_rdma', 'after_barrier')
        inter_unpack = _ms('after_barrier', 'end')

        line = (f'[EFA_TIMING] total={total:.2f}ms | '
              f'py_setup={python_setup:.2f} intra_pack={intra_pack:.2f} nccl_a2a={nccl_a2a:.2f} '
              f'intra_unpack={intra_unpack:.2f} recv_zero={recv_zero:.2f} inter_pack={inter_pack:.2f} '
              f'efa_off_build={efa_offset_build:.2f} off_exchange={offset_exchange:.2f} '
              f'efa_rdma={efa_rdma:.2f} barrier={barrier:.2f} inter_unpack={inter_unpack:.2f} | '
              f'inter_send={inter_send_bytes}B inter_recv={inter_recv_bytes}B\n')
        # Write to file to avoid stdout suppression by bench_kineto
        with open('/tmp/efa_timing.log', 'a') as f:
            f.write(line)

    def _efa_transfer(self, send_flat, recv_flat, send_counts, recv_counts, bytes_per_token):
        """Transfer data using EFA RDMA for inter-node and NCCL for intra-node.
        
        Optimized (Iter 22):
        - CUDA event-based sync replaces torch.cuda.synchronize()
        - Overlap offset exchange (default stream) with pack phase (comm_stream)
        - C++ efa_all_to_all uses cudaStreamSynchronize instead of cudaDeviceSynchronize
        """
        # Record event on default stream to ensure send_flat data is ready.
        # comm_stream will wait on this event before reading send_flat.
        # This replaces torch.cuda.synchronize() — avoids stalling ALL streams.
        default_stream = torch.cuda.current_stream()
        data_ready_event = torch.cuda.Event()
        data_ready_event.record(default_stream)

        num_ranks = len(send_counts)
        num_local_ranks = self._num_local_ranks
        my_node = self._my_node
        inter_ranks_set = self._inter_ranks_set
        inter_ranks_list = self._inter_ranks_list
        has_inter = self._has_inter
        half_rdma = self._half_rdma

        elem_per_token = bytes_per_token // send_flat.element_size()

        # Pure-Python prefix sum — avoids GPU tensor creation + cumsum + .tolist() sync
        send_elem_counts = [send_counts[r] * elem_per_token for r in range(num_ranks)]
        recv_elem_counts = [recv_counts[r] * elem_per_token for r in range(num_ranks)]
        send_offsets = [0] * (num_ranks + 1)
        recv_offsets = [0] * (num_ranks + 1)
        for r in range(num_ranks):
            send_offsets[r + 1] = send_offsets[r] + send_elem_counts[r]
            recv_offsets[r + 1] = recv_offsets[r] + recv_elem_counts[r]

        send_flat_1d = send_flat.view(-1)
        recv_flat_1d = recv_flat.view(-1)

        # Compute inter-node sizes
        inter_send_bytes = 0
        inter_recv_bytes = 0
        if has_inter:
            inter_send_bytes = sum(send_counts[r] * bytes_per_token for r in inter_ranks_list)
            inter_recv_bytes = sum(recv_counts[r] * bytes_per_token for r in inter_ranks_list)

        # Build NCCL list-based a2a: views into send_flat/recv_flat for intra-node ranks
        has_intra = False
        _empty = torch.empty(0, dtype=send_flat.dtype, device='cuda')
        nccl_send_list = []
        nccl_recv_list = []
        for r in range(num_ranks):
            if r not in inter_ranks_set:
                s_off = send_offsets[r]
                s_n = send_elem_counts[r]
                r_off = recv_offsets[r]
                r_n = recv_elem_counts[r]
                nccl_send_list.append(send_flat_1d[s_off:s_off + s_n] if s_n > 0 else _empty)
                nccl_recv_list.append(recv_flat_1d[r_off:r_off + r_n] if r_n > 0 else _empty)
                if s_n > 0 or r_n > 0:
                    has_intra = True
            else:
                nccl_send_list.append(_empty)
                nccl_recv_list.append(_empty)

        comm_stream = self.get_comm_stream()

        # ==================================================================
        # Phase 1: Build EFA offset arrays + start offset exchange EARLY
        #           (overlaps with pack on comm_stream)
        # ==================================================================
        rdma_send_buf = None
        rdma_recv_buf = None
        efa_send_sizes = None
        efa_send_offsets = None
        efa_recv_sizes = None
        efa_recv_offsets = None
        remote_recv_offsets = None

        if has_inter and (inter_send_bytes > 0 or inter_recv_bytes > 0):
            assert inter_send_bytes <= half_rdma and inter_recv_bytes <= half_rdma

            # Build EFA offset arrays FIRST (pure CPU, no GPU dependency)
            efa_send_sizes = []
            efa_send_offsets = []
            efa_recv_sizes = []
            efa_recv_offsets = []
            send_pack_off = 0
            recv_pack_off = 0
            for r in range(num_ranks):
                if r in inter_ranks_set:
                    s_bytes = send_counts[r] * bytes_per_token
                    r_bytes = recv_counts[r] * bytes_per_token
                    efa_send_sizes.append(s_bytes)
                    efa_send_offsets.append(send_pack_off)
                    efa_recv_sizes.append(r_bytes)
                    efa_recv_offsets.append(recv_pack_off)
                    send_pack_off += s_bytes
                    recv_pack_off += r_bytes
                else:
                    efa_send_sizes.append(0)
                    efa_send_offsets.append(0)
                    efa_recv_sizes.append(0)
                    efa_recv_offsets.append(0)

            # Start offset exchange on DEFAULT stream — overlaps with pack on comm_stream
            my_offsets_tensor = torch.tensor(efa_recv_offsets, dtype=torch.int64, device='cuda')
            remote_recv_offsets_tensor = torch.empty_like(my_offsets_tensor)
            dist.all_to_all_single(
                remote_recv_offsets_tensor, my_offsets_tensor,
                group=self._get_gpu_group())

            # Get RDMA buffers and start packing on comm_stream
            rdma_send_buf = self.runtime.get_local_buffer_tensor(torch.uint8, 0, True)
            rdma_recv_buf = self.runtime.get_local_buffer_tensor(torch.uint8, half_rdma, True)

            with torch.cuda.stream(comm_stream):
                # Wait for send_flat data to be ready on default stream
                comm_stream.wait_event(data_ready_event)

                # NOTE: Do NOT zero rdma_recv_buf here. The .zero_() loads data into
                # GPU L2 cache, and subsequent DMA-BUF RDMA writes bypass L2, causing
                # stale cache reads during unpack. We unpack exact amounts from the
                # RDMA buffer, so zeroing is unnecessary.

                if inter_send_bytes > 0:
                    rdma_send_view = rdma_send_buf[:inter_send_bytes].view(send_flat.dtype)
                    inter_blocks = self._find_contiguous_blocks(
                        num_ranks, num_local_ranks, my_node,
                        send_elem_counts, send_offsets, is_inter=True)
                    pack_off = 0
                    for src_off, n_elem in inter_blocks:
                        if n_elem > 0:
                            rdma_send_view[pack_off:pack_off + n_elem] = send_flat_1d[src_off:src_off + n_elem]
                            pack_off += n_elem

            # Materialize remote_recv_offsets (GPU→CPU, overlapped with comm_stream pack)
            remote_recv_offsets = remote_recv_offsets_tensor.tolist()

        # ==================================================================
        # Phase 2: NCCL list-based a2a (intra-node, overlaps with pack completion)
        # ==================================================================
        if has_intra:
            dist.all_to_all(nccl_recv_list, nccl_send_list, group=self._get_gpu_group())

        # ==================================================================
        # Phase 3: EFA RDMA writes
        # ==================================================================
        if has_inter and (inter_send_bytes > 0 or inter_recv_bytes > 0):
            comm_stream.synchronize()

            self.runtime.efa_all_to_all(
                rdma_send_buf[:max(1, inter_send_bytes)],
                efa_send_sizes,
                efa_send_offsets,
                rdma_recv_buf[:max(1, inter_recv_bytes)],
                efa_recv_sizes,
                remote_recv_offsets)

            # NOTE: No post-RDMA barrier needed. efa_all_to_all waits for all
            # RX completions and calls cudaDeviceSynchronize, so all received
            # data is visible when it returns. Each rank only unpacks its own data.

        # ==================================================================
        # Phase 4: Unpack inter-node recv (contiguous block copies)
        # ==================================================================
        if has_inter and (inter_send_bytes > 0 or inter_recv_bytes > 0):
            rdma_recv_view = rdma_recv_buf[:max(1, inter_recv_bytes)].view(recv_flat.dtype)

            recv_blocks = self._find_contiguous_blocks(
                num_ranks, num_local_ranks, my_node,
                recv_elem_counts, recv_offsets, is_inter=True)
            unpack_off = 0
            for dst_off, n_elem in recv_blocks:
                if n_elem > 0:
                    recv_flat_1d[dst_off:dst_off + n_elem] = rdma_recv_view[unpack_off:unpack_off + n_elem]
                    unpack_off += n_elem

    def _efa_transfer_multi(self, transfers, send_counts, recv_counts):
        """Transfer multiple (send_flat, recv_flat, bytes_per_token) tuples with shared
        send_counts/recv_counts in a SINGLE call — one offset exchange, one barrier.
        
        Optimized (Iter 22): 
        - CUDA event-based sync replaces torch.cuda.synchronize()
        - Merged intra-node NCCL calls: pack all payloads into single uint8 buffer
        - Overlap offset exchange (default stream) with pack phase (comm_stream)
        
        ONLY called when len(transfers) >= 2. Single-payload transfers use _efa_transfer.
        """
        assert len(transfers) >= 2, \
            f"_efa_transfer_multi requires >= 2 payloads, got {len(transfers)}. Use _efa_transfer for single payload."

        # Record event on default stream to ensure all send tensors are ready.
        default_stream = torch.cuda.current_stream()
        data_ready_event = torch.cuda.Event()
        data_ready_event.record(default_stream)

        num_ranks = len(send_counts)
        num_local_ranks = self._num_local_ranks
        my_node = self._my_node
        inter_ranks_set = self._inter_ranks_set
        inter_ranks_list = self._inter_ranks_list
        has_inter = self._has_inter
        half_rdma = self._half_rdma

        # Compute per-rank packed byte sizes across all payloads
        packed_send_bytes_per_rank = [0] * num_ranks
        packed_recv_bytes_per_rank = [0] * num_ranks
        for send_flat, recv_flat, bpt in transfers:
            for r in range(num_ranks):
                packed_send_bytes_per_rank[r] += send_counts[r] * bpt
                packed_recv_bytes_per_rank[r] += recv_counts[r] * bpt

        # Build per-payload offset info for NCCL and EFA (pure-Python prefix sums)
        payload_infos = []
        for idx, (send_flat, recv_flat, bpt) in enumerate(transfers):
            elem_per_token = bpt // send_flat.element_size()
            send_elem_counts = [send_counts[r] * elem_per_token for r in range(num_ranks)]
            recv_elem_counts = [recv_counts[r] * elem_per_token for r in range(num_ranks)]
            send_offsets = [0] * (num_ranks + 1)
            recv_offsets = [0] * (num_ranks + 1)
            for r in range(num_ranks):
                send_offsets[r + 1] = send_offsets[r] + send_elem_counts[r]
                recv_offsets[r + 1] = recv_offsets[r] + recv_elem_counts[r]
            payload_infos.append({
                'send_flat': send_flat,
                'recv_flat': recv_flat,
                'bpt': bpt,
                'elem_per_token': elem_per_token,
                'send_elem_counts': send_elem_counts,
                'recv_elem_counts': recv_elem_counts,
                'send_offsets': send_offsets,
                'recv_offsets': recv_offsets,
            })

        # Inter-node total sizes
        inter_send_bytes = 0
        inter_recv_bytes = 0
        if has_inter:
            inter_send_bytes = sum(packed_send_bytes_per_rank[r] for r in inter_ranks_list)
            inter_recv_bytes = sum(packed_recv_bytes_per_rank[r] for r in inter_ranks_list)

        comm_stream = self.get_comm_stream()

        # ==================================================================
        # Phase 1: Build EFA offsets + start offset exchange EARLY
        #           + pack all payloads on comm_stream (overlapped)
        # ==================================================================
        rdma_send_buf = None
        rdma_recv_buf = None
        efa_send_sizes = None
        efa_send_offsets = None
        efa_recv_sizes = None
        efa_recv_offsets = None
        remote_recv_offsets = None

        if has_inter and (inter_send_bytes > 0 or inter_recv_bytes > 0):
            assert inter_send_bytes <= half_rdma, \
                f"Inter-node send data ({inter_send_bytes}) exceeds RDMA send buffer ({half_rdma})"
            assert inter_recv_bytes <= half_rdma, \
                f"Inter-node recv data ({inter_recv_bytes}) exceeds RDMA recv buffer ({half_rdma})"

            # Build packed EFA offset arrays FIRST (pure CPU, no GPU dependency)
            efa_send_sizes = []
            efa_send_offsets = []
            efa_recv_sizes = []
            efa_recv_offsets = []
            send_pack_off = 0
            recv_pack_off = 0
            for r in range(num_ranks):
                if r in inter_ranks_set:
                    s_bytes = packed_send_bytes_per_rank[r]
                    r_bytes = packed_recv_bytes_per_rank[r]
                    efa_send_sizes.append(s_bytes)
                    efa_send_offsets.append(send_pack_off)
                    efa_recv_sizes.append(r_bytes)
                    efa_recv_offsets.append(recv_pack_off)
                    send_pack_off += s_bytes
                    recv_pack_off += r_bytes
                else:
                    efa_send_sizes.append(0)
                    efa_send_offsets.append(0)
                    efa_recv_sizes.append(0)
                    efa_recv_offsets.append(0)

            # Start offset exchange on DEFAULT stream — overlaps with pack on comm_stream
            my_offsets_tensor = torch.tensor(efa_recv_offsets, dtype=torch.int64, device='cuda')
            remote_recv_offsets_tensor = torch.empty_like(my_offsets_tensor)
            dist.all_to_all_single(
                remote_recv_offsets_tensor, my_offsets_tensor,
                group=self._get_gpu_group())

            # Get RDMA buffers and start packing on comm_stream
            rdma_send_buf = self.runtime.get_local_buffer_tensor(torch.uint8, 0, True)
            rdma_recv_buf = self.runtime.get_local_buffer_tensor(torch.uint8, half_rdma, True)

            with torch.cuda.stream(comm_stream):
                # Wait for send tensors to be ready on default stream
                comm_stream.wait_event(data_ready_event)

                # NOTE: Do NOT zero rdma_recv_buf here. The .zero_() loads data into
                # GPU L2 cache, and subsequent DMA-BUF RDMA writes bypass L2, causing
                # stale cache reads during unpack. We unpack exact amounts from the
                # RDMA buffer, so zeroing is unnecessary.

                # Pack: for each inter-node rank, concatenate all payloads' data
                # RDMA layout: [rank0_payload0, rank0_payload1, rank1_payload0, ...]
                if inter_send_bytes > 0:
                    pack_off = 0  # byte offset into rdma_send_buf
                    for r in inter_ranks_list:
                        for pi in payload_infos:
                            n_bytes = send_counts[r] * pi['bpt']
                            if n_bytes > 0:
                                src_flat = pi['send_flat'].view(-1)
                                src_off_elem = pi['send_offsets'][r]
                                n_elem = pi['send_elem_counts'][r]
                                rdma_send_buf[pack_off:pack_off + n_bytes].view(pi['send_flat'].dtype)[:n_elem] = \
                                    src_flat[src_off_elem:src_off_elem + n_elem]
                            pack_off += n_bytes

            # Materialize remote_recv_offsets (GPU→CPU, overlapped with comm_stream pack)
            remote_recv_offsets = remote_recv_offsets_tensor.tolist()

        # ==================================================================
        # Phase 2: NCCL list-based all-to-all for each payload (intra-node)
        # Iter 22: Reverted to per-payload NCCL (merged pack was slower for FP8
        # due to pack/unpack overhead outweighing NCCL launch savings)
        # ==================================================================
        for pi in payload_infos:
            send_flat_1d = pi['send_flat'].view(-1)
            recv_flat_1d = pi['recv_flat'].view(-1)
            _empty = torch.empty(0, dtype=pi['send_flat'].dtype, device='cuda')
            has_intra = False
            nccl_send_list = []
            nccl_recv_list = []
            for r in range(num_ranks):
                if r not in inter_ranks_set:
                    s_off = pi['send_offsets'][r]
                    s_n = pi['send_elem_counts'][r]
                    r_off = pi['recv_offsets'][r]
                    r_n = pi['recv_elem_counts'][r]
                    nccl_send_list.append(send_flat_1d[s_off:s_off + s_n] if s_n > 0 else _empty)
                    nccl_recv_list.append(recv_flat_1d[r_off:r_off + r_n] if r_n > 0 else _empty)
                    if s_n > 0 or r_n > 0:
                        has_intra = True
                else:
                    nccl_send_list.append(_empty)
                    nccl_recv_list.append(_empty)
            if has_intra:
                dist.all_to_all(nccl_recv_list, nccl_send_list, group=self._get_gpu_group())

        # ==================================================================
        # Phase 3: Single EFA RDMA transfer of packed buffer
        # ==================================================================
        if has_inter and (inter_send_bytes > 0 or inter_recv_bytes > 0):
            comm_stream.synchronize()

            self.runtime.efa_all_to_all(
                rdma_send_buf[:max(1, inter_send_bytes)],
                efa_send_sizes,
                efa_send_offsets,
                rdma_recv_buf[:max(1, inter_recv_bytes)],
                efa_recv_sizes,
                remote_recv_offsets)

            # NOTE: No post-RDMA barrier needed. efa_all_to_all waits for all
            # RX completions and calls cudaDeviceSynchronize, so all received
            # data is visible when it returns.

        # ==================================================================
        # Phase 4: Unpack all payloads from RDMA recv buf
        # ==================================================================
        if has_inter and (inter_send_bytes > 0 or inter_recv_bytes > 0):
            unpack_off = 0  # byte offset into rdma_recv_buf
            for r in inter_ranks_list:
                for pi in payload_infos:
                    n_bytes = recv_counts[r] * pi['bpt']
                    if n_bytes > 0:
                        recv_flat_1d = pi['recv_flat'].view(-1)
                        dst_off_elem = pi['recv_offsets'][r]
                        n_elem = pi['recv_elem_counts'][r]
                        recv_flat_1d[dst_off_elem:dst_off_elem + n_elem] = \
                            rdma_recv_buf[unpack_off:unpack_off + n_bytes].view(pi['recv_flat'].dtype)[:n_elem]
                    unpack_off += n_bytes

    @staticmethod
    def _find_contiguous_blocks(num_ranks, num_local_ranks, my_node,
                                elem_counts_list, offsets_list, is_inter):
        """Find contiguous blocks of inter-node (or intra-node) data in a flat buffer.
        
        Returns list of (offset, num_elems) tuples for contiguous blocks.
        For 2-node case, inter-node data is typically a single contiguous block.
        """
        blocks = []
        block_start_off = None
        block_elems = 0
        for r in range(num_ranks):
            is_inter_rank = (r // num_local_ranks != my_node)
            if is_inter_rank == is_inter:
                n_elem = int(elem_counts_list[r])
                off = int(offsets_list[r])
                if block_start_off is None:
                    block_start_off = off
                    block_elems = n_elem
                elif off == block_start_off + block_elems:
                    block_elems += n_elem
                else:
                    if block_elems > 0:
                        blocks.append((block_start_off, block_elems))
                    block_start_off = off
                    block_elems = n_elem
            else:
                if block_start_off is not None and block_elems > 0:
                    blocks.append((block_start_off, block_elems))
                block_start_off = None
                block_elems = 0
        if block_start_off is not None and block_elems > 0:
            blocks.append((block_start_off, block_elems))
        return blocks

    # noinspection PyTypeChecker
    def internode_combine(self, x: torch.Tensor, handle: Union[tuple, list],
                          topk_weights: Optional[torch.Tensor] = None,
                          bias: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                          config: Optional[Config] = None,
                          previous_event: Optional[EventOverlap] = None, async_finish: bool = False,
                          allocate_on_comm_stream: bool = False) -> \
            Tuple[torch.Tensor, Optional[torch.Tensor], EventOverlap]:
        """
        Optimized internode combine — vectorized scatter-add.
        Iter 25: Pre-allocated combine buffers + in-place bias.
        """
        assert config is not None
        rank = self.rank
        num_ranks = self.group_size

        is_combined_token_in_rank = handle[0]
        efa_info = self._efa_dispatch_info
        recv_counts_dispatch = efa_info['recv_counts']
        send_counts_dispatch = efa_info['send_counts']
        send_token_indices = efa_info['send_token_indices']
        flat_sorted_token_ids = efa_info['flat_sorted_token_ids']
        num_combined_tokens = is_combined_token_in_rank.size(0)
        hidden = x.size(1)
        num_topk = topk_weights.size(1) if topk_weights is not None else 0

        bias_0, bias_1 = Buffer._unpack_bias(bias)

        combine_send_counts = recv_counts_dispatch
        combine_recv_counts = send_counts_dispatch

        total_send = sum(combine_send_counts)
        total_recv = sum(combine_recv_counts)

        # Iter 25: Pre-allocated combine buffers
        # Use max capacity based on dispatch info. Re-allocate if key changes (different config).
        cache_key = (num_combined_tokens, hidden, num_topk, x.dtype)
        if not hasattr(self, '_combine_cache') or self._combine_cache_key != cache_key:
            # Max possible total_recv = num_tokens * num_topk_per_token
            # But we don't know num_topk_per_token here (it's the topk from dispatch).
            # Use a reasonable upper bound: num_combined_tokens * 16 (max topk seen in practice)
            max_recv = num_combined_tokens * max(num_topk, 8)
            self._combine_cache = {
                'recv_x_max': torch.empty((max_recv, hidden), dtype=x.dtype, device='cuda'),
                'recv_w_max': torch.empty((max_recv, max(1, num_topk)), dtype=torch.float32, device='cuda') if num_topk > 0 else None,
                'combined_x': torch.zeros((num_combined_tokens, hidden), dtype=x.dtype, device='cuda'),
                'combined_w': torch.zeros((num_combined_tokens, max(1, num_topk)), dtype=torch.float32, device='cuda') if num_topk > 0 else None,
                'max_recv': max_recv,
            }
            self._combine_cache_key = cache_key

        cc = self._combine_cache

        assert x.size(0) == total_send or x.size(0) >= total_send, \
            f"Combine input size {x.size(0)} doesn't match expected {total_send}"
        send_x = x[:total_send].contiguous()

        # Use pre-allocated recv buffer (with fallback if too small)
        if total_recv <= cc['max_recv']:
            recv_x_flat = cc['recv_x_max'][:total_recv]
        else:
            recv_x_flat = torch.empty((total_recv, hidden), dtype=x.dtype, device='cuda')

        use_efa = getattr(self, '_efa_initialized', False)
        if use_efa:
            if topk_weights is not None:
                # Batch x + topk_weights into single _efa_transfer_multi call
                send_w = topk_weights[:total_send].contiguous()
                recv_w_flat = cc['recv_w_max'][:total_recv] if (total_recv <= cc['max_recv'] and cc['recv_w_max'] is not None) else torch.empty((total_recv, num_topk), dtype=torch.float32, device='cuda')
                self._efa_transfer_multi([
                    (send_x, recv_x_flat, hidden * x.element_size()),
                    (send_w, recv_w_flat, num_topk * 4),
                ], combine_send_counts, combine_recv_counts)
            else:
                self._efa_transfer(send_x, recv_x_flat, combine_send_counts, combine_recv_counts,
                                   hidden * x.element_size())
        else:
            send_splits = [s * hidden for s in combine_send_counts]
            recv_splits = [r * hidden for r in combine_recv_counts]
            dist.all_to_all_single(
                recv_x_flat.view(-1), send_x.view(-1),
                output_split_sizes=recv_splits, input_split_sizes=send_splits,
                group=self._get_gpu_group())

        # Also all-to-all topk_weights if present (non-EFA path, or post-batch for EFA)
        combined_topk_weights = None
        if topk_weights is not None:
            if use_efa:
                # Already transferred via _efa_transfer_multi above
                pass
            else:
                send_w = topk_weights[:total_send].contiguous()
                recv_w_flat = cc['recv_w_max'][:total_recv] if (total_recv <= cc['max_recv'] and cc['recv_w_max'] is not None) else torch.empty((total_recv, num_topk), dtype=torch.float32, device='cuda')
                send_w_splits = [s * num_topk for s in combine_send_counts]
                recv_w_splits = [r * num_topk for r in combine_recv_counts]
                dist.all_to_all_single(
                    recv_w_flat.view(-1), send_w.view(-1),
                    output_split_sizes=recv_w_splits, input_split_sizes=send_w_splits,
                    group=self._get_gpu_group())

        # ------------------------------------------------------------------
        # Vectorized scatter-add: single index_add_ instead of per-rank loop
        # recv_x_flat is ordered by source rank. flat_sorted_token_ids maps
        # each entry back to the original token position (same order as send).
        # Iter 25: Reuse pre-allocated output buffers (zero + index_add_)
        # ------------------------------------------------------------------
        combined_x = cc['combined_x']
        combined_x.zero_()

        if total_recv > 0 and flat_sorted_token_ids.numel() > 0:
            combined_x.index_add_(0, flat_sorted_token_ids[:total_recv], recv_x_flat[:total_recv].to(combined_x.dtype))

        if topk_weights is not None:
            combined_topk_weights = cc['combined_w']
            combined_topk_weights.zero_()
            if total_recv > 0 and flat_sorted_token_ids.numel() > 0:
                combined_topk_weights.index_add_(0, flat_sorted_token_ids[:total_recv], recv_w_flat[:total_recv])

        # Apply biases — Iter 25: in-place to avoid tensor allocation
        if bias_0 is not None:
            combined_x.add_(bias_0)
        if bias_1 is not None:
            combined_x.add_(bias_1)

        return combined_x, combined_topk_weights, EventOverlap(EventHandle())

    def clean_low_latency_buffer(self, num_max_dispatch_tokens_per_rank: int, hidden: int, num_experts: int) -> None:
        """
        As low-latency kernels require part of the buffer to be zero-initialized, so it is vital to clean the buffer
            if the buffer is dirty at some time.
        For example, after running the normal dispatch/combine, you must run this function before executing any
            low-latency kernel.

        Arguments:
            num_max_dispatch_tokens_per_rank: the maximum number of tokens to dispatch, all the ranks must hold the same value.
            hidden: the hidden dimension of each token.
            num_experts: the number of all experts.
        """
        # Python implementation: reset the low-latency combine buffer index
        # The original C++ zeroes the signaling/recv buffers; in our Python impl
        # we don't use persistent RDMA buffers, so this is mostly a no-op.
        self._ll_buffer_idx = 0
        # Reset the combine buffer if it exists
        if hasattr(self, '_ll_combine_buffer'):
            for buf in self._ll_combine_buffer:
                if buf is not None:
                    buf.zero_()
        # Also reset the dispatch stash
        self._ll_dispatch_stash = None

    # =====================================================================
    # Low-latency dispatch/combine: pure Python + NCCL implementation
    # =====================================================================

    def _ll_per_token_cast_to_fp8(self, x: torch.Tensor, round_scale: bool = False, use_ue8m0: bool = False):
        """Per-token FP8 cast: x [M, N] bf16 -> (x_fp8 [M, N] fp8, scales [M, N//128] float)."""
        assert x.dim() == 2
        m, n = x.shape
        group_size = 128
        assert n % group_size == 0, f"hidden {n} must be divisible by {group_size}"
        x_grouped = x.view(m, -1, group_size)  # [M, G, 128]
        x_amax = x_grouped.abs().float().amax(dim=2).clamp(min=1e-4)  # [M, G]
        scales = x_amax / 448.0  # [M, G]
        if round_scale:
            # Round to nearest power of 2
            scales = torch.exp2(torch.ceil(torch.log2(scales)))
        inv_scales = 448.0 / x_amax
        if round_scale:
            inv_scales = 1.0 / scales
        x_fp8 = (x_grouped.float() * inv_scales.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, n)
        if use_ue8m0:
            # Pack scales as UE8M0: extract exponent byte from float32
            # Each float32 scale -> 1 byte exponent, pack 4 into an int32
            # scales shape: [M, G], output shape: [M, G // 4] int32
            scales_bytes = scales.view(dtype=torch.int32)  # reinterpret as int32
            exponents = ((scales_bytes >> 23) & 0xFF).to(torch.uint8)  # [M, G]
            # Pack 4 exponents into one int32
            num_groups = exponents.shape[1]
            assert num_groups % 4 == 0
            exponents_i32 = exponents.to(torch.int32).view(m, -1, 4)  # [M, G//4, 4]
            packed = (exponents_i32[:, :, 0] |
                      (exponents_i32[:, :, 1] << 8) |
                      (exponents_i32[:, :, 2] << 16) |
                      (exponents_i32[:, :, 3] << 24))  # [M, G//4]
            return x_fp8, packed.to(torch.int32)
        return x_fp8, scales

    def _ll_get_cache(self, num_tokens, hidden, num_ranks, num_local_experts,
                      num_max_dispatch_tokens_per_rank, num_topk, use_fp8, use_ue8m0):
        """Lazily initialize and return pre-allocated LL tensor cache.
        
        Caches fixed-shape tensors and max-size variable buffers to eliminate
        per-call torch.empty()/torch.zeros()/torch.arange() allocation overhead.
        
        Iter 21: Extended with pre-allocated buffers for:
        - C++ dispatch route+pack scratch (send_counts, send_cumsum, total_send_out, send_packed_max, etc.)
        - C++ recv unpack scratch (recv_cumsum, pair_counts, expert_cumsum, etc.)
        - packed_recv_x, packed_recv_scales (big output tensors)
        - recv_packed buffer (big intermediate)
        - inverse_sort, global_expert_ids (combine helpers)
        - combined_x, combined_x_f32 (combine output + accumulator)

        Iter 27: When EFA is initialized, allocate send_packed and recv_packed from
        the RDMA buffer so EFA RDMA writes can use them as source/destination directly.
        """
        cache_key = (num_tokens, hidden, num_ranks, num_local_experts,
                     num_max_dispatch_tokens_per_rank, num_topk, use_fp8, use_ue8m0)
        if hasattr(self, '_ll_cache') and self._ll_cache_key == cache_key:
            return self._ll_cache

        N = num_max_dispatch_tokens_per_rank * num_ranks
        if use_fp8:
            if use_ue8m0:
                num_scale_cols = hidden // 128 // 4  # UE8M0 packs 4 exponents per int32
                scale_elem_size = 4  # int32
            else:
                num_scale_cols = hidden // 128
                scale_elem_size = 4  # float32
        else:
            num_scale_cols = 0
            scale_elem_size = 0
        max_valid = num_tokens * num_topk  # max possible valid entries

        # Determine packed_bytes_per_token for max-size buffer allocation
        if use_fp8:
            data_bpt = hidden  # fp8 = 1 byte per element
            scale_bpt = num_scale_cols * scale_elem_size
        else:
            data_bpt = hidden * 2  # bf16 = 2 bytes per element
            scale_bpt = 0
        meta_bpt = 8
        packed_bpt = data_bpt + scale_bpt + meta_bpt
        total_pairs = num_local_experts * num_ranks
        # max_valid is the max per-rank SEND count (bounded by num_tokens * num_topk)
        # max_recv is the max per-rank RECEIVE count: since all ranks send up to max_valid tokens,
        # and a given rank can receive from all ranks, worst case is num_ranks * max_valid.
        # But in practice it's much less. Use num_ranks * max_valid for safety.
        max_recv = num_ranks * max_valid

        # === Iter 27: RDMA buffer layout for EFA LL ===
        # When EFA is initialized, allocate send_packed, recv_packed, and combine
        # send/recv buffers from the RDMA buffer for zero-copy RDMA transfers.
        # Layout (within each half):
        #   [0, count_region): count exchange (num_ranks * 4, 8KB aligned)
        #   [data_offset, ...): packed dispatch/combine data
        use_efa = getattr(self, '_efa_initialized', False)
        if use_efa:
            half_rdma = self._half_rdma
            # Count region: each rank's send_counts array (num_ranks * 4 bytes)
            count_region = num_ranks * 4
            data_offset = ((count_region + 8191) // 8192) * 8192  # 8KB aligned

            # Max send packed size: max_valid * packed_bpt bytes
            # Max combine send size: max_recv * hidden * 2 bytes (bf16)
            # We use the larger of the two since dispatch and combine don't overlap
            dispatch_send_bytes = max(1, max_valid) * packed_bpt
            combine_send_bytes = max(1, max_recv) * hidden * 2  # bf16
            max_send_bytes = max(dispatch_send_bytes, combine_send_bytes)

            # For recv: fixed-slot layout — num_ranks * slot_size per rank
            dispatch_slot_size = num_max_dispatch_tokens_per_rank * packed_bpt
            combine_slot_size = num_max_dispatch_tokens_per_rank * hidden * 2
            dispatch_recv_bytes = num_ranks * dispatch_slot_size
            combine_recv_bytes = num_ranks * combine_slot_size
            max_recv_bytes = max(dispatch_recv_bytes, combine_recv_bytes)

            # Verify RDMA buffer has enough space
            send_avail = half_rdma - data_offset
            recv_avail = half_rdma - data_offset
            assert max_send_bytes <= send_avail, \
                f"LL send needs {max_send_bytes}B but RDMA send has {send_avail}B"
            assert max_recv_bytes <= recv_avail, \
                f"LL recv needs {max_recv_bytes}B but RDMA recv has {recv_avail}B"

            # Allocate from RDMA buffer
            rdma_send_packed = self.runtime.get_local_buffer_tensor(
                torch.uint8, data_offset, True)[:dispatch_send_bytes]
            rdma_recv_packed = self.runtime.get_local_buffer_tensor(
                torch.uint8, half_rdma + data_offset, True)[:dispatch_recv_bytes]
            # Combine send/recv also from RDMA buffer
            rdma_combine_send = self.runtime.get_local_buffer_tensor(
                torch.uint8, data_offset, True)[:combine_send_bytes]
            rdma_combine_recv = self.runtime.get_local_buffer_tensor(
                torch.uint8, half_rdma + data_offset, True)[:combine_recv_bytes]

        cache = {
            # Constant tensors (avoid per-call CUDA kernel launches)
            'arange_num_tokens': torch.arange(num_tokens, device='cuda'),
            'arange_num_ranks': torch.arange(num_ranks, device='cuda', dtype=torch.int64),
            'arange_num_ranks_i32': torch.arange(num_ranks, device='cuda', dtype=torch.int32),
            # Pre-allocated for count exchange
            'recv_counts_tensor': torch.empty(num_ranks, dtype=torch.int32, device='cuda'),
            # For combine: pre-allocate max-size recv buffer
            # Iter 27: From RDMA buffer when EFA is available, else regular GPU memory
            'combine_recv_max': (rdma_combine_recv.view(torch.bfloat16).reshape(-1, hidden)[:max(1, max_valid)]
                                 if use_efa else
                                 torch.empty((max(1, max_valid), hidden), dtype=torch.bfloat16, device='cuda')),
            # For dispatch: pre-allocate recv_packed buffer (Iter 22)
            # Use 2 * max_valid as reasonable upper bound; fall back to dynamic allocation if exceeded
            'recv_packed_max_tokens': max_valid * 2,
            'recv_packed_max': (rdma_recv_packed
                                if use_efa else
                                torch.empty(max(1, max_valid * 2) * packed_bpt, dtype=torch.uint8, device='cuda')),
            # Cache metadata
            'N': N,
            'num_scale_cols': num_scale_cols,
            'packed_bpt': packed_bpt,
            'data_bpt': data_bpt,
            'scale_bpt': scale_bpt,

            # === Iter 21: Pre-allocated dispatch route+pack scratch ===
            'dispatch_send_counts': torch.zeros(num_ranks, dtype=torch.int32, device='cuda'),
            'dispatch_send_cumsum': torch.zeros(num_ranks + 1, dtype=torch.int64, device='cuda'),
            'dispatch_total_send_out': torch.zeros(1, dtype=torch.int32, device='cuda'),
            # Iter 27: From RDMA buffer when EFA is available
            'dispatch_send_packed_max': (rdma_send_packed
                                         if use_efa else
                                         torch.empty(max(1, max_valid) * packed_bpt, dtype=torch.uint8, device='cuda')),
            'dispatch_sorted_token_ids_max': torch.empty(max(1, max_valid), dtype=torch.int64, device='cuda'),
            'dispatch_sorted_local_eids_max': torch.empty(max(1, max_valid), dtype=torch.int32, device='cuda'),

            # === Iter 21: Pre-allocated recv unpack scratch ===
            'recv_cumsum': torch.zeros(num_ranks + 1, dtype=torch.int32, device='cuda'),
            'pair_counts': torch.zeros(total_pairs, dtype=torch.int32, device='cuda'),
            'packed_recv_count': torch.zeros(num_local_experts, dtype=torch.int32, device='cuda'),
            'packed_recv_layout_range': torch.zeros(num_local_experts, num_ranks, dtype=torch.int64, device='cuda'),
            'expert_cumsum': torch.zeros(num_local_experts + 1, dtype=torch.int32, device='cuda'),
            'pair_cumsum': torch.zeros(total_pairs, dtype=torch.int32, device='cuda'),
            'packed_recv_src_info': torch.zeros(num_local_experts, N, dtype=torch.bfloat16, device='cuda'),
            'recv_expert_ids_max': torch.empty(max(1, max_recv), dtype=torch.int32, device='cuda'),
            'recv_expert_pos_max': torch.empty(max(1, max_recv), dtype=torch.int64, device='cuda'),
            'sort_order_recv_max': torch.empty(max(1, max_recv), dtype=torch.int64, device='cuda'),
            'pair_write_counters': torch.zeros(total_pairs, dtype=torch.int32, device='cuda'),

            # === Iter 21: Pre-allocated packed_recv_x and packed_recv_scales ===
            'packed_recv_x_fp8': torch.zeros((num_local_experts, N, hidden), dtype=torch.float8_e4m3fn, device='cuda') if use_fp8 else None,
            'packed_recv_x_bf16': torch.zeros((num_local_experts, N, hidden), dtype=torch.bfloat16, device='cuda') if not use_fp8 else None,
            'packed_recv_scales_f32': torch.zeros((num_local_experts, N, num_scale_cols), dtype=torch.float32, device='cuda') if (use_fp8 and not use_ue8m0) else None,
            'packed_recv_scales_i32': torch.zeros((num_local_experts, N, num_scale_cols), dtype=torch.int32, device='cuda') if (use_fp8 and use_ue8m0) else None,

            # === Iter 21: Pre-allocated combine helpers ===
            'inverse_sort_max': torch.empty(max(1, max_recv), dtype=torch.long, device='cuda'),
            'global_expert_ids_max': torch.empty(max(1, max_valid), dtype=torch.int64, device='cuda'),

            # === Iter 21: Pre-allocated combine output + f32 accumulator ===
            'combined_x': torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device='cuda'),
            'combined_x_f32': torch.zeros(num_tokens, hidden, dtype=torch.float32, device='cuda'),

            # === Iter 27: EFA RDMA buffer metadata ===
            'use_efa': use_efa,
            'efa_data_offset': data_offset if use_efa else 0,
            # For combine: RDMA send buffer for combine data (reuses dispatch send region)
            'combine_send_rdma': (rdma_combine_send if use_efa else None),

            # === Iter 31: Pinned CPU tensors for overlapped NCCL count exchange ===
            # These avoid .tolist() which would sync the default CUDA stream.
            'recv_counts_pin': torch.empty(num_ranks, dtype=torch.int32, device='cpu', pin_memory=True),
            'send_counts_pin': torch.empty(num_ranks, dtype=torch.int32, device='cpu', pin_memory=True),

            # === Iter 32: Pre-allocated combine send_cumsum for slot-based addressing ===
            'combine_send_cumsum_i32': torch.zeros(num_ranks + 1, dtype=torch.int32, device='cuda'),

            # === Iter 33: Cached CUDA events to avoid per-call Python object creation ===
            'count_event': torch.cuda.Event(),
            'pack_done_event': torch.cuda.Event(),

            # === Iter 33: Pre-computed fixed recv offsets for EFA (constant per cache key) ===
            'dispatch_recv_offsets': [self.rank * dispatch_slot_size] * num_ranks if use_efa else None,
            'combine_recv_offsets': [self.rank * combine_slot_size] * num_ranks if use_efa else None,
            'dispatch_slot_size': dispatch_slot_size if use_efa else 0,
            'combine_slot_size': combine_slot_size if use_efa else 0,
        }
        # Iter 33: Eagerly create nccl_count_stream during cache init
        if use_efa and not hasattr(self, '_nccl_count_stream'):
            self._nccl_count_stream = self.get_comm_stream()
        self._ll_cache = cache
        self._ll_cache_key = cache_key
        return cache

    # noinspection PyTypeChecker
    def low_latency_dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                             num_max_dispatch_tokens_per_rank: int, num_experts: int,
                             cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
                             dispatch_wait_recv_cost_stats: Optional[torch.Tensor] = None,
                             use_fp8: bool = True, round_scale: bool = False, use_ue8m0: bool = False,
                             async_finish: bool = False, return_recv_hook: bool = False,
                             topk_weights: Optional[torch.Tensor] = None) -> \
            Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple, EventOverlap, Callable]:
        """
        Optimized low-latency dispatch using fused C++ CUDA kernels + EFA/NCCL.
        Key optimizations:
        - All large tensors pre-allocated in cache (zero per-call allocations)
        - C++ v2 APIs accept pre-allocated buffers, return send_counts as CPU list
          (eliminates .tolist() CPU sync for send_counts)
        - C++ ll_compute_combine_helpers: fused inverse_sort + global_expert_ids
          (replaces ~9 Python kernel launches with 2 CUDA kernels)
        - Iter 27: EFA RDMA for multi-node (2 EFA calls: count + data),
          NCCL fallback for single-node (2 NCCL all_to_all calls)
        - Single .tolist() CPU sync for recv_counts only
        """
        rank = self.rank
        num_ranks = self.group_size
        num_local_experts = num_experts // num_ranks
        num_tokens = x.size(0)
        hidden = x.size(1)
        N = num_max_dispatch_tokens_per_rank * num_ranks
        num_topk = topk_idx.size(1)

        # Profiling
        prof = None
        if _PROFILE_ENABLED:
            if not hasattr(self, '_ll_dispatch_profiler'):
                self._ll_dispatch_profiler = _LLProfiler('DISPATCH', rank)
            prof = self._ll_dispatch_profiler
            prof.mark('start')

        # Get pre-allocated cache
        ll_cache = self._ll_get_cache(num_tokens, hidden, num_ranks, num_local_experts,
                                       num_max_dispatch_tokens_per_rank, num_topk,
                                       use_fp8, use_ue8m0)

        if prof: prof.mark('cache')

        # ------------------------------------------------------------------
        # Step 1: FP8 casting (if requested) — Iter 23: fused C++ CUDA kernel
        # ------------------------------------------------------------------
        if use_fp8:
            x_fp8, x_scales = deep_ep_cpp.per_token_cast_to_fp8(x, round_scale, use_ue8m0)
            num_scale_cols = x_scales.size(1)
        else:
            x_fp8 = x
            x_scales = None
            num_scale_cols = 0

        if prof: prof.mark('fp8_cast')

        # ------------------------------------------------------------------
        # Step 2+4: Routing + packing
        # For EFA mode: SPLIT into count_only + pack_only to overlap pack with NCCL count exchange
        # For other modes: FUSED route_and_pack_v3 (no overlap opportunity)
        # ------------------------------------------------------------------
        x_data_bytes = x_fp8.view(torch.uint8).reshape(num_tokens, -1).contiguous()
        if use_fp8:
            x_scales_bytes = x_scales.view(torch.uint8).reshape(num_tokens, -1).contiguous()
        else:
            x_scales_bytes = None

        data_bytes_per_token = ll_cache['data_bpt']
        scale_bytes_per_token = ll_cache['scale_bpt']
        meta_bytes_per_token = 8  # 2 x int32
        packed_bytes_per_token = ll_cache['packed_bpt']

        # ------------------------------------------------------------------
        # Step 3: Count exchange + Data exchange
        # Use EFA RDMA for multi-node, NCCL fallback for single-node
        # ------------------------------------------------------------------
        use_efa = getattr(self, '_efa_initialized', False)
        use_pipeline = use_efa and _LL_PIPELINE_ENABLED
        use_coop = use_efa and _COOP_ENABLED

        _zero_done = False  # Iter 35: tracks whether packed_recv_x/scales already zeroed

        recv_counts_tensor = ll_cache['recv_counts_tensor']
        send_counts_tensor = ll_cache['dispatch_send_counts']

        # For non-EFA or pipeline paths: use fused route_and_pack_v3
        # For EFA (non-pipeline) path: use split count_only + pack_only for NCCL overlap
        # For coop path: use cooperative kernel (dispatch_send replaces route_and_pack)
        if use_coop:
            # === Cooperative Kernel Dispatch Path (Iter 50) ===
            # 
            # Flow: Worker thread handles route exchange, metadata, RDMA, barriers
            #       in parallel with cooperative GPU kernels.
            #
            # Python just launches coop_send + coop_recv kernels and waits.
            # All internode coordination (RDMA count exchange, metadata upload,
            # data RDMA, barriers) runs on the LLPipelineWorker CPU thread.

            half_rdma = self._half_rdma
            hidden = x.size(1)
            hidden_dim_scale = num_scale_cols if use_fp8 else 0
            x_elemsize = x_fp8.element_size()  # 1 for FP8, 2 for BF16

            # Compute coop token dimensions (must match C++ kernel's token_stride)
            def _round_up_16(v):
                return (v + 15) // 16 * 16
            coop_token_dim = _round_up_16(hidden * x_elemsize)
            # Use original scale element size (float32=4), NOT x_scales_bytes (uint8=1)
            x_scale_elemsize = x_scales.element_size() if use_fp8 else 0  # 4 for float32
            coop_token_scale_dim = _round_up_16(hidden_dim_scale * x_scale_elemsize)
            coop_token_stride = coop_token_dim + coop_token_scale_dim + 16  # +16 for metadata
            coop_slot_size = num_max_dispatch_tokens_per_rank * coop_token_stride

            # Combine token dim (no scale, no metadata — just BF16 expert output)
            combine_token_dim = _round_up_16(hidden * 2)  # bf16

            # topk_weights is required for cooperative kernels (packed into metadata)
            if topk_weights is None:
                topk_weights = torch.ones((num_tokens, num_topk), dtype=torch.float32, device=x.device)

            # Initialize coop scratch (once)
            max_recv = num_tokens * num_topk * num_ranks
            if not getattr(self, '_coop_initialized', False):
                self.runtime.coop_init(num_tokens, num_topk, num_experts, max_recv)
                self._coop_initialized = True
                if rank == 0:
                    print(f"[Rank {rank}] Cooperative kernel scratch initialized (max_recv={max_recv})", file=sys.stderr)

            # Initialize coop worker (once) — configures LLPipelineWorker for coop mode
            if not getattr(self, '_coop_worker_initialized', False):
                # Ensure LL pipeline is initialized first
                if not getattr(self, '_ll_pipeline_initialized', False):
                    self.runtime.init_ll_pipeline(coop_token_stride, num_max_dispatch_tokens_per_rank)
                    self._ll_pipeline_initialized = True
                self.runtime.coop_worker_init(
                    num_experts, num_topk, num_max_dispatch_tokens_per_rank,
                    coop_token_stride, combine_token_dim)
                self._coop_worker_initialized = True
                if rank == 0:
                    print(f"[Rank {rank}] Coop worker initialized", file=sys.stderr)

            # Get a CUDA stream for cooperative kernel launches
            if not hasattr(self, '_coop_stream'):
                self._coop_stream = torch.cuda.Stream()
            coop_stream = self._coop_stream

            # Reset GDR flags for this iteration
            self.runtime.coop_reset_flags()

            _dbg = getattr(self, '_coop_dbg_count', 0)
            if _dbg < 3 and rank == 0:
                print(f"[COOP-DBG r{rank}] iter={_dbg} entering coop dispatch (worker mode), use_fp8={use_fp8}, "
                      f"hidden={hidden}, hidden_dim_scale={hidden_dim_scale}, x_fp8.shape={x_fp8.shape}, "
                      f"stride={coop_token_stride}", file=sys.stderr, flush=True)

            # Step 1: Signal tx_ready (send buffer available)
            self.runtime.coop_signal_tx_ready()

            # Step 2: Start worker thread for coop dispatch (returns immediately)
            # The worker will: wait route_done → RDMA scatter routing → wait peers →
            # compute metadata → upload to GPU → signal num_recv_tokens_flag →
            # wait pack_done → RDMA data transfer → signal dispatch_recv_flag →
            # wait dispatch_recv_done → barrier → set tx_ready
            self.runtime.start_coop_dispatch()
            if prof: prof.mark('worker_start')

            # Step 3: Prepare recv buffers BEFORE launching cooperative kernels.
            # CRITICAL: Cooperative kernels need ALL SMs on the GPU. If any kernel
            # (e.g., memset from zero_()) is running on ANY stream, the cooperative
            # launch will be blocked indefinitely. Synchronize the default stream
            # to ensure all GPU work is complete before launching cooperative kernels.
            # Note: event-based sync (wait_event) is INSUFFICIENT because cooperative
            # kernels require ALL SMs to be physically free, not just stream ordering.
            max_recv_for_buf = max(max_recv, num_local_experts * num_max_dispatch_tokens_per_rank)
            coop_recv_key = '_coop_recv_buf'
            if not hasattr(self, coop_recv_key) or self._coop_recv_buf.size(0) < max_recv_for_buf:
                self._coop_recv_buf = torch.zeros(
                    (max_recv_for_buf, hidden),
                    dtype=torch.bfloat16, device='cuda')
            packed_recv_x_flat = self._coop_recv_buf[:max_recv_for_buf]
            packed_recv_x_flat.zero_()
            packed_recv_scales = None
            out_num_tokens = torch.zeros(num_local_experts, dtype=torch.int32, device='cuda')
            topk_idx_i32 = topk_idx.to(torch.int32) if topk_idx.dtype != torch.int32 else topk_idx

            # Synchronize default stream: ensures zero_() and torch.zeros() GPU kernels
            # have completed and released their SMs before cooperative kernel launch.
            torch.cuda.current_stream().synchronize()

            # Step 4: Launch coop_dispatch_send on coop stream (cooperative kernel, all SMs)
            with torch.cuda.stream(coop_stream):
                self.runtime.coop_dispatch_send(
                    x_fp8, x_scales if use_fp8 else None,
                    topk_idx_i32, topk_weights,
                    num_experts, num_tokens, hidden, hidden_dim_scale, 1,  # dp_size=1
                    coop_stream.cuda_stream)

            if _dbg < 3 and rank == 0:
                print(f"[COOP-DBG r{rank}] coop_dispatch_send launched, launching recv...", file=sys.stderr, flush=True)
            if prof: prof.mark('coop_send')

            with torch.cuda.stream(coop_stream):
                src_elemsize = x_fp8.element_size()
                src_scale_elemsize = x_scales.element_size() if use_fp8 else 0
                self.runtime.coop_dispatch_recv(
                    packed_recv_x_flat,
                    None,  # no scales output
                    out_num_tokens,
                    num_experts, hidden, src_elemsize, src_scale_elemsize, hidden_dim_scale,
                    coop_stream.cuda_stream)

            # Step 5: Wait for coop stream (both send and recv kernels complete)
            coop_stream.synchronize()
            if prof: prof.mark('coop_recv')

            # Step 6: Wait for worker thread to finish (barrier, tx_ready)
            self.runtime.wait_coop_dispatch_done()
            if prof: prof.mark('worker_done')

            # Step 7: Get per-rank counts from worker (computed during route exchange)
            send_counts, recv_counts, total_send, total_recv = self.runtime.coop_worker_get_counts()

            if _dbg < 3 and rank == 0:
                print(f"[COOP-DBG r{rank}] dispatch COMPLETE (worker mode), send={total_send}, recv={total_recv}",
                      file=sys.stderr, flush=True)

            # Reshape output
            max_per_expert = num_max_dispatch_tokens_per_rank
            packed_recv_x = packed_recv_x_flat[:num_local_experts * max_per_expert].view(
                num_local_experts, max_per_expert, hidden)
            packed_recv_count = out_num_tokens

            # Stash for combine
            self._ll_dispatch_stash = {
                'send_counts': send_counts,
                'recv_counts': recv_counts,
                'use_coop': True,
                'coop_token_stride': coop_token_stride,
                'combine_token_dim': combine_token_dim,
                'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank,
            }

            # Create dummy handle and event for test compatibility
            dummy_src_info = torch.zeros(num_local_experts, num_max_dispatch_tokens_per_rank,
                                         dtype=torch.int32, device='cuda')
            dummy_layout_range = torch.zeros(num_local_experts, num_max_dispatch_tokens_per_rank,
                                              dtype=torch.int64, device='cuda')
            handle = (dummy_src_info, dummy_layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts)
            event = EventOverlap(EventHandle())
            hook = (lambda: None) if return_recv_hook else None

            self._coop_dbg_count = _dbg + 1

            if prof:
                prof.mark('end')
                prof.finish_iter(f'send={total_send} recv={total_recv}')

            return packed_recv_x, packed_recv_count, handle, event, hook

        elif not use_efa or use_pipeline:
            # FUSED route + pack (used by NCCL-only/intranode paths)
            send_packed_max, send_counts_tensor, sorted_token_ids_max, sorted_local_eids_max = \
                deep_ep_cpp.ll_dispatch_route_and_pack_v3(
                    topk_idx, x_data_bytes, x_scales_bytes,
                    ll_cache['dispatch_send_counts'],
                    ll_cache['dispatch_send_cumsum'],
                    ll_cache['dispatch_total_send_out'],
                    ll_cache['dispatch_send_packed_max'],
                    ll_cache['dispatch_sorted_token_ids_max'],
                    ll_cache['dispatch_sorted_local_eids_max'],
                    num_ranks, num_local_experts,
                    data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token)
            if prof: prof.mark('route_pack')

        if use_pipeline:
            # === GDRCopy Pipeline path: overlap NCCL + EFA RDMA ===
            half_rdma = self._half_rdma
            data_offset = ll_cache['efa_data_offset']

            # Initialize or update pipeline worker
            packed_bpt = ll_cache['packed_bpt']
            if not getattr(self, '_ll_pipeline_initialized', False):
                self.runtime.init_ll_pipeline(packed_bpt, num_max_dispatch_tokens_per_rank)
                self._ll_pipeline_initialized = True
                self._ll_pipeline_gdr_dispatch = self.runtime.ll_pipeline_get_dispatch_gdr_ptrs()
                self._ll_pipeline_gdr_combine = self.runtime.ll_pipeline_get_combine_gdr_ptrs()
                self._ll_pipeline_packed_bpt = packed_bpt
            elif packed_bpt != self._ll_pipeline_packed_bpt:
                # packed_bytes_per_token changed (e.g., BF16 vs FP8 mode) — update config
                self.runtime.init_ll_pipeline(packed_bpt, num_max_dispatch_tokens_per_rank)
                self._ll_pipeline_packed_bpt = packed_bpt

            # Phase 1: Count only (separate from packing for overlap)
            # This is already done by ll_dispatch_route_and_pack_v3 above, which
            # computed send_counts_tensor and packed data in one fused kernel.
            # We can't split it now since it already ran. So we proceed with:
            #   - NCCL count exchange (uses send_counts from the fused kernel)
            #   - Signal CPU worker with send_counts + pack_done after NCCL
            #   - CPU worker issues RDMA writes

            # Phase 2: NCCL count exchange (implicit stream sync with pack kernel)
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self._get_gpu_group())

            if prof: prof.mark('nccl_count')

            # .tolist() syncs default stream → gets CPU-side counts
            # All GPU work (pack kernel + NCCL + gdr_signal_counts) is complete after this.
            recv_counts = recv_counts_tensor.tolist()
            send_counts = send_counts_tensor.tolist()
            total_send = sum(send_counts)
            total_recv = sum(recv_counts)

            if prof: prof.mark('tolist')

            # Slice send results to actual size
            send_packed = send_packed_max[:max(1, total_send) * packed_bytes_per_token]
            sorted_token_ids = sorted_token_ids_max[:total_send]
            sorted_local_eids = sorted_local_eids_max[:total_send]

            # Pipeline dispatch: CPU worker issues RDMA writes
            slot_size = num_max_dispatch_tokens_per_rank * packed_bytes_per_token
            total_recv_region = num_ranks * slot_size

            if total_send > 0 or total_recv > 0:
                if prof: prof.mark('pre_efa')

                # Start pipeline dispatch: writes send_counts and recv_counts to GDR memory,
                # then signals the CPU worker to issue RDMA writes
                self.runtime.ll_pipeline_start_dispatch(send_counts, recv_counts)

                # Wait for pipeline to complete (RDMA writes done + cudaDeviceSynchronize)
                self.runtime.ll_pipeline_wait_dispatch()

                if prof: prof.mark('efa_a2a')

                # Iter 32: Pass RDMA recv buffer directly to unpack kernels (no torch.cat)
                rdma_data_recv = self.runtime.get_local_buffer_tensor(
                    torch.uint8, half_rdma + data_offset, True)[:total_recv_region]
                recv_packed = rdma_data_recv  # kernels handle slot-based addressing
            else:
                recv_packed = ll_cache['recv_packed_max'][:0]
                slot_size = 0  # no slot-based addressing for empty case
                if prof: prof.mark('pre_efa'); prof.mark('efa_a2a')

        elif use_efa:
            # === EFA path (Iter 44): In-band count exchange via imm data ===
            # No NCCL count exchange needed — token counts sent in RDMA imm data.
            # Phase 1: count_only kernel (fast, ~30us) on default stream
            # Phase 2: pack_only kernel launched on default stream (runs async)
            # Phase 3: D2H copy of send_counts on comm_stream (overlaps with pack)
            # Phase 4: EFA RDMA data exchange with in-band token counts
            half_rdma = self._half_rdma
            data_offset = ll_cache['efa_data_offset']

            # Phase 1: Count only — writes send_counts + cumsum on default stream
            deep_ep_cpp.ll_dispatch_count_only(
                topk_idx,
                ll_cache['dispatch_send_counts'],
                ll_cache['dispatch_send_cumsum'],
                ll_cache['dispatch_total_send_out'],
                num_ranks, num_local_experts)

            # Record event after count kernel (default stream)
            count_event = ll_cache['count_event']
            count_event.record()

            if prof: prof.mark('count_only')

            # Phase 2: Pack only — runs async on default stream
            deep_ep_cpp.ll_dispatch_pack_only(
                topk_idx, x_data_bytes, x_scales_bytes,
                ll_cache['dispatch_send_cumsum'],
                ll_cache['dispatch_send_packed_max'],
                ll_cache['dispatch_sorted_token_ids_max'],
                ll_cache['dispatch_sorted_local_eids_max'],
                num_ranks, num_local_experts,
                data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token)

            if prof: prof.mark('pack_launch')

            # Phase 3: D2H copy of send_counts on comm_stream (overlaps with pack)
            nccl_stream = self._nccl_count_stream
            with torch.cuda.stream(nccl_stream):
                # Wait for count kernel to complete before reading send_counts
                nccl_stream.wait_event(count_event)
                # Async D2H copy of send_counts to pinned CPU tensor
                ll_cache['send_counts_pin'].copy_(send_counts_tensor, non_blocking=True)

            # Ensure pack kernel (default stream) completes before EFA reads send data.
            pack_done = ll_cache['pack_done_event']
            pack_done.record()  # records on default stream (where pack_only ran)
            nccl_stream.wait_event(pack_done)  # comm_stream waits for pack

            if prof: prof.mark('nccl_count')

            # Phase 4: Fused EFA RDMA data exchange
            slot_size = num_max_dispatch_tokens_per_rank * packed_bytes_per_token
            total_recv_region = num_ranks * slot_size
            rdma_data_recv = self.runtime.get_local_buffer_tensor(
                torch.uint8, half_rdma + data_offset, True)[:total_recv_region]

            # Iter 35: Launch .zero_() on default stream BEFORE the C++ call.
            # Runs concurrently with EFA RDMA transfer (different memory).
            if use_fp8:
                packed_recv_x = ll_cache['packed_recv_x_fp8']
                packed_recv_x.zero_()
                if use_ue8m0:
                    packed_recv_scales = ll_cache['packed_recv_scales_i32']
                else:
                    packed_recv_scales = ll_cache['packed_recv_scales_f32']
                packed_recv_scales.zero_()
            else:
                packed_recv_x = ll_cache['packed_recv_x_bf16']
                packed_recv_x.zero_()
                packed_recv_scales = None

            if prof: prof.mark('pre_efa')

            # Use NCCL all-to-all for count exchange (reliable, acts as barrier).
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self._get_gpu_group())

            # Pre-dispatch barrier: flush ALL local GPU streams (default stream
            # where unpack/reduce kernels run, NCCL stream), then cross-rank
            # barrier so all ranks' GPU work is done before EFA RDMA writes.
            torch.cuda.synchronize()
            dist.barrier(group=self.group)

            # C++ reads recv_counts from recv_counts_tensor, computes offsets,
            # and calls efa_all_to_all without imm counts.
            send_counts, recv_counts, total_send, total_recv = \
                self.runtime.ll_efa_dispatch_data_v3(
                    ll_cache['send_counts_pin'],
                    ll_cache['dispatch_send_packed_max'],
                    rdma_data_recv,
                    recv_counts_tensor,
                    packed_bytes_per_token, slot_size)

            if prof: prof.mark('efa_a2a')

            # Slice send results to actual size (needed for unpack + combine stash)
            sorted_token_ids_max = ll_cache['dispatch_sorted_token_ids_max']
            sorted_local_eids_max = ll_cache['dispatch_sorted_local_eids_max']
            sorted_token_ids = sorted_token_ids_max[:total_send]
            sorted_local_eids = sorted_local_eids_max[:total_send]

            # Iter 32: Pass RDMA recv buffer directly to unpack kernels (no torch.cat)
            recv_packed = rdma_data_recv  # kernels handle slot-based addressing
            # .zero_() already done above (overlapped with EFA)
            _zero_done = True
        else:
            # === NCCL fallback path (single-node EP8) ===
            slot_size = 0  # contiguous flat layout, no slot-based addressing
            dist.all_to_all_single(recv_counts_tensor, send_counts_tensor, group=self._get_gpu_group())

            if prof: prof.mark('nccl_count')

            # Read counts (NCCL implicitly syncs default CUDA stream)
            recv_counts = recv_counts_tensor.tolist()
            send_counts = send_counts_tensor.tolist()
            total_send = sum(send_counts)
            total_recv = sum(recv_counts)

            if prof: prof.mark('tolist')

            # Slice send results to actual size
            send_packed = send_packed_max[:max(1, total_send) * packed_bytes_per_token]
            sorted_token_ids = sorted_token_ids_max[:total_send]
            sorted_local_eids = sorted_local_eids_max[:total_send]

            # Recv buffer
            if total_recv <= ll_cache['recv_packed_max_tokens']:
                recv_packed = ll_cache['recv_packed_max'][:max(1, total_recv) * packed_bytes_per_token]
            else:
                recv_packed = torch.empty(max(1, total_recv) * packed_bytes_per_token, dtype=torch.uint8, device='cuda')

            # 2nd all-to-all: packed data
            send_byte_splits = [s * packed_bytes_per_token for s in send_counts]
            recv_byte_splits = [r * packed_bytes_per_token for r in recv_counts]

            if prof: prof.mark('pre_efa')

            dist.all_to_all_single(
                recv_packed[:total_recv * packed_bytes_per_token] if total_recv > 0 else recv_packed[:0],
                send_packed,
                output_split_sizes=recv_byte_splits, input_split_sizes=send_byte_splits,
                group=self._get_gpu_group())

            if prof: prof.mark('efa_a2a'); prof.mark('extract_recv')

        # ------------------------------------------------------------------
        # Step 5: Unpack and arrange into [num_local_experts, N, hidden]
        # ------------------------------------------------------------------
        if total_recv == 0:
            if use_fp8:
                packed_recv_x = ll_cache['packed_recv_x_fp8']
                packed_recv_x.zero_()
                if use_ue8m0:
                    packed_recv_scales = ll_cache['packed_recv_scales_i32']
                else:
                    packed_recv_scales = ll_cache['packed_recv_scales_f32']
                packed_recv_scales.zero_()
            else:
                packed_recv_x = ll_cache['packed_recv_x_bf16']
                packed_recv_x.zero_()
                packed_recv_scales = None
            packed_recv_count = ll_cache['packed_recv_count']
            packed_recv_count.zero_()
            packed_recv_src_info = ll_cache['packed_recv_src_info']
            packed_recv_src_info.zero_()
            packed_recv_layout_range = ll_cache['packed_recv_layout_range']
            packed_recv_layout_range.zero_()
            # Pre-compute combine helpers even for total_recv == 0 (total_send may be > 0)
            if total_send > 0:
                _, global_expert_ids = deep_ep_cpp.ll_compute_combine_helpers(
                    torch.empty(0, dtype=torch.long, device='cuda'),  # sort_order (unused)
                    ll_cache['dispatch_send_cumsum'],
                    sorted_local_eids,
                    ll_cache['inverse_sort_max'],
                    ll_cache['global_expert_ids_max'],
                    0, total_send, num_ranks, num_local_experts)
            else:
                global_expert_ids = torch.empty(0, dtype=torch.int64, device='cuda')
            self._ll_dispatch_stash = {
                'sorted_send_token_ids': sorted_token_ids,
                'sorted_send_local_eids': sorted_local_eids,
                'send_counts': send_counts,
                'recv_counts': recv_counts,
                'send_counts_tensor': send_counts_tensor,
                'recv_counts_tensor': recv_counts_tensor,
                'recv_expert_ids': torch.empty(0, dtype=torch.long, device='cuda'),  # Iter 33: int64 for combine
                'recv_flat_sort_order': torch.empty(0, dtype=torch.long, device='cuda'),
                'recv_expert_pos': torch.empty(0, dtype=torch.long, device='cuda'),
                'inverse_sort': torch.empty(0, dtype=torch.long, device='cuda'),
                'global_expert_ids': global_expert_ids,
            }
            handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts)
            event = EventOverlap(EventHandle())
            hook = (lambda: None) if return_recv_hook else None
            if prof:
                prof.mark('end')
                prof.finish_iter()
            if use_fp8:
                return (packed_recv_x, packed_recv_scales), packed_recv_count, handle, event, hook
            else:
                return packed_recv_x, packed_recv_count, handle, event, hook

        # ------------------------------------------------------------------
        # Step 5: FUSED C++ unpack + scatter with pre-allocated buffers
        # Uses ll_recv_unpack_v2: all scratch buffers from cache
        # ------------------------------------------------------------------
        # Get pre-allocated output tensors and zero them
        # Iter 35: For EFA path, .zero_() was already launched before efa_all_to_all
        # (overlapped with EFA transfer). For other paths, zero here.
        if use_fp8:
            packed_recv_x = ll_cache['packed_recv_x_fp8']
            if not _zero_done:
                packed_recv_x.zero_()
            if use_ue8m0:
                packed_recv_scales = ll_cache['packed_recv_scales_i32']
            else:
                packed_recv_scales = ll_cache['packed_recv_scales_f32']
            if not _zero_done:
                packed_recv_scales.zero_()
        else:
            packed_recv_x = ll_cache['packed_recv_x_bf16']
            if not _zero_done:
                packed_recv_x.zero_()
            packed_recv_scales = None

        if prof: prof.mark('zero_recv')

        hidden_bytes = hidden * packed_recv_x.element_size()
        scale_cols_bytes = num_scale_cols * packed_recv_scales.element_size() if packed_recv_scales is not None else 0

        packed_recv_count, packed_recv_src_info, packed_recv_layout_range, \
            recv_expert_ids, recv_expert_pos, sort_order_recv = \
            deep_ep_cpp.ll_recv_unpack_v2(
                recv_packed if slot_size > 0 else (recv_packed[:total_recv * packed_bytes_per_token] if total_recv > 0 else recv_packed[:0]),
                recv_counts_tensor,
                packed_recv_x,
                packed_recv_scales,
                # Pre-allocated scratch:
                ll_cache['recv_cumsum'],
                ll_cache['pair_counts'],
                ll_cache['packed_recv_count'],
                ll_cache['packed_recv_layout_range'],
                ll_cache['expert_cumsum'],
                ll_cache['pair_cumsum'],
                ll_cache['packed_recv_src_info'],
                ll_cache['recv_expert_ids_max'],
                ll_cache['recv_expert_pos_max'],
                ll_cache['sort_order_recv_max'],
                ll_cache['pair_write_counters'],
                total_recv, num_ranks, num_local_experts,
                data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token,
                N, hidden_bytes, scale_cols_bytes, slot_size)

        if prof: prof.mark('unpack')

        # Update cumulative stats
        if cumulative_local_expert_recv_stats is not None:
            cumulative_local_expert_recv_stats.add_(packed_recv_count.to(cumulative_local_expert_recv_stats.dtype))

        # Pre-compute combine helpers using fused C++ kernel
        # Replaces: inverse_sort scatter (~3 launches) + repeat_interleave + global_expert_ids (~6 launches)
        # With: 2 simple CUDA kernels
        inverse_sort, global_expert_ids = deep_ep_cpp.ll_compute_combine_helpers(
            sort_order_recv,
            ll_cache['dispatch_send_cumsum'],
            sorted_local_eids,
            ll_cache['inverse_sort_max'],
            ll_cache['global_expert_ids_max'],
            total_recv, total_send, num_ranks, num_local_experts)

        if prof: prof.mark('combine_helpers')

        # Stash for combine
        self._ll_dispatch_stash = {
            'sorted_send_token_ids': sorted_token_ids,
            'sorted_send_local_eids': sorted_local_eids,
            'send_counts': send_counts,
            'recv_counts': recv_counts,
            'send_counts_tensor': send_counts_tensor,
            'recv_counts_tensor': recv_counts_tensor,
            'recv_expert_ids': recv_expert_ids.to(torch.long),  # Iter 33: cast once here, avoid per-combine cast
            'recv_flat_sort_order': sort_order_recv,
            'recv_expert_pos': recv_expert_pos,
            'inverse_sort': inverse_sort,
            'global_expert_ids': global_expert_ids,
            'num_max_dispatch_tokens_per_rank': num_max_dispatch_tokens_per_rank,
        }

        handle = (packed_recv_src_info, packed_recv_layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts)
        event = EventOverlap(EventHandle())
        hook = (lambda: None) if return_recv_hook else None
        if prof:
            prof.mark('end')
            prof.finish_iter()
        if use_fp8:
            return (packed_recv_x, packed_recv_scales), packed_recv_count, handle, event, hook
        else:
            return packed_recv_x, packed_recv_count, handle, event, hook

    # noinspection PyTypeChecker
    def low_latency_combine(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor,
                            handle: tuple, use_logfmt: bool = False, zero_copy: bool = False, async_finish: bool = False,
                            return_recv_hook: bool = False, out: Optional[torch.Tensor] = None,
                            combine_wait_recv_cost_stats: Optional[torch.Tensor] = None) -> \
            Tuple[torch.Tensor, EventOverlap, Callable]:
        """
        Optimized low-latency combine using fused CUDA kernel + EFA/NCCL all-to-all.
        Key optimizations:
        - Pre-allocated combined_x and combined_x_f32 from cache (Iter 21)
        - ll_combine_weighted_reduce_v2 with pre-allocated f32 accumulator
        - Pre-allocated combine_recv buffer from cache (Iter 19)
        - Iter 27: EFA RDMA for multi-node, NCCL fallback for single-node
        """
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
        num_ranks = self.group_size
        num_local_experts = num_experts // num_ranks
        num_tokens = topk_idx.size(0)
        num_topk = topk_idx.size(1)

        # Profiling
        prof = None
        if _PROFILE_ENABLED:
            if not hasattr(self, '_ll_combine_profiler'):
                self._ll_combine_profiler = _LLProfiler('COMBINE', self.rank)
            prof = self._ll_combine_profiler
            prof.mark('start')

        stash = self._ll_dispatch_stash

        # Check if this is a cooperative kernel combine
        use_coop_combine = stash.get('use_coop', False)

        if use_coop_combine:
            # === Cooperative Kernel Combine Path (Iter 49) ===
            _dbg = getattr(self, '_coop_dbg_count', 0)
            rank = self.rank
            if _dbg <= 3 and rank == 0:
                print(f"[COOP-DBG r{rank}] entering coop combine", file=sys.stderr, flush=True)
            # 
            # === Cooperative Kernel Combine Path (Iter 50 — Worker Thread) ===
            #
            # Flow: Worker thread handles combine RDMA + barriers.
            # combine_send → worker waits combine_send_done → issue combine RDMA
            # → signal combine_recv_flag → combine_recv → wait combine_recv_done → barrier

            send_counts = stash['send_counts']
            recv_counts = stash['recv_counts']
            num_max_dpr = stash['num_max_dispatch_tokens_per_rank']
            total_recv = sum(recv_counts)
            total_send = sum(send_counts)
            hidden = x.size(-1)

            # Get coop stream
            coop_stream = self._coop_stream

            # tx_ready is already set by the dispatch worker's barrier step

            # Step 1: Start worker thread for coop combine (returns immediately)
            self.runtime.start_coop_combine()
            if prof: prof.mark('worker_start')

            # Step 2: Prepare output buffer. Use torch.empty (no initialization kernel)
            # to avoid SM contention with cooperative kernels.
            combined_x = out if out is not None else torch.empty(
                (num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            topk_idx_i32 = topk_idx.to(torch.int32) if topk_idx.dtype != torch.int32 else topk_idx

            # Step 3: Launch coop_combine_send on coop stream
            if _dbg <= 3 and rank == 0:
                print(f"[COOP-DBG r{rank}] launching combine_send (worker mode), x.shape={x.shape}", file=sys.stderr, flush=True)
            with torch.cuda.stream(coop_stream):
                self.runtime.coop_combine_send(
                    x.view(-1, hidden),
                    hidden, 1,  # dp_size=1
                    coop_stream.cuda_stream)
            if prof: prof.mark('coop_combine_send')
            with torch.cuda.stream(coop_stream):
                self.runtime.coop_combine_recv(
                    topk_idx_i32, topk_weights,
                    combined_x,
                    num_experts, num_tokens, hidden,
                    False,  # accumulate=False
                    coop_stream.cuda_stream)

            # Step 4: Wait for coop stream (both kernels complete)
            coop_stream.synchronize()
            if prof: prof.mark('coop_combine_recv')

            # Step 5: Wait for worker thread to finish (barrier, tx_ready)
            self.runtime.wait_coop_combine_done()
            if prof: prof.mark('worker_done')

            if _dbg <= 3 and rank == 0:
                print(f"[COOP-DBG r{rank}] combine COMPLETE (worker mode)", file=sys.stderr, flush=True)

            event = EventOverlap(EventHandle())
            hook = (lambda: None) if return_recv_hook else None

            if prof:
                prof.mark('end')
                prof.finish_iter(f'send={total_send} recv={total_recv}')

            return combined_x, event, hook

        send_counts = stash['send_counts']
        recv_counts = stash['recv_counts']
        recv_expert_ids = stash['recv_expert_ids']
        recv_expert_pos = stash['recv_expert_pos']

        total_recv = sum(recv_counts)
        total_send = sum(send_counts)

        # ------------------------------------------------------------------
        # Step 1: Gather expert outputs and unsort to dispatch-receive order
        # Use pre-computed inverse_sort from dispatch
        # Iter 33: For EFA paths, write directly into RDMA send buffer
        # ------------------------------------------------------------------
        ll_cache = getattr(self, '_ll_cache', None)
        use_efa = ll_cache is not None and ll_cache.get('use_efa', False)
        use_pipeline = use_efa and _LL_PIPELINE_ENABLED and getattr(self, '_ll_pipeline_initialized', False)

        # Note: The pre-combine barrier was removed in favor of a post-combine barrier.
        # The post-combine barrier at the end of low_latency_combine ensures all ranks
        # finish combine RDMA before any rank starts the next dispatch. This also
        # protects the recv buffer from back-to-back combine overwrites (since no rank
        # can start the next dispatch → combine cycle until all ranks finish the
        # current combine).
        barrier_work = None

        # Set up RDMA send buffer early so gather+unsort can write directly into it
        rdma_combine_send = None
        if (use_efa or use_pipeline) and total_recv > 0:
            data_offset = ll_cache['efa_data_offset']
            bytes_per_elem = 2  # bf16
            combine_send_bytes = total_recv * hidden * bytes_per_elem
            rdma_combine_send = self.runtime.get_local_buffer_tensor(
                torch.uint8, data_offset, True)
            # View RDMA send buffer as bf16 for direct write
            rdma_send_bf16 = rdma_combine_send[:combine_send_bytes].view(torch.bfloat16).reshape(total_recv, hidden)

        if total_recv > 0:
            sorted_expert_outputs = x[recv_expert_ids, recv_expert_pos]  # Iter 33: already int64 from stash
            inverse_sort = stash['inverse_sort']
            if rdma_combine_send is not None:
                # Iter 33: Write directly into RDMA send buffer (eliminates copy_to_rdma)
                torch.index_select(sorted_expert_outputs, 0, inverse_sort, out=rdma_send_bf16)
                combine_send_flat = rdma_send_bf16
            else:
                combine_send_flat = sorted_expert_outputs[inverse_sort].contiguous()
        else:
            combine_send_flat = torch.empty((0, hidden), dtype=torch.bfloat16, device='cuda')

        if prof: prof.mark('gather_unsort')

        # ------------------------------------------------------------------
        # Step 2: All-to-all — EFA for multi-node, NCCL for single-node
        # In combine, the roles are reversed vs dispatch:
        #   combine_send_sizes[i] = dispatch_recv_counts[i] * hidden * 2 (bf16)
        #   combine_recv_sizes[i] = dispatch_send_counts[i] * hidden * 2 (bf16)
        # ------------------------------------------------------------------

        if use_pipeline:
            # === GDRCopy Pipeline path for combine ===
            half_rdma = self._half_rdma
            data_offset = ll_cache['efa_data_offset']
            bytes_per_elem = 2  # bf16
            num_max = stash.get('num_max_dispatch_tokens_per_rank', num_max_dispatch_tokens_per_rank)

            combine_send_bytes = total_recv * hidden * bytes_per_elem
            combine_recv_bytes = total_send * hidden * bytes_per_elem

            # Fixed-slot layout for combine recv
            combine_slot_size = num_max * hidden * bytes_per_elem
            total_combine_recv_region = num_ranks * combine_slot_size

            rdma_combine_recv = self.runtime.get_local_buffer_tensor(
                torch.uint8, half_rdma + data_offset, True)[:total_combine_recv_region]

            # Iter 33: combine_send_flat already in RDMA send buffer (written by gather+unsort above)
            if prof: prof.mark('copy_to_rdma')

            # Iter 45: Wait for async barrier (started before gather+unsort).
            # This ensures all ranks' previous reduce kernels have completed,
            # protecting the shared recv buffer from overwrite.
            if barrier_work is not None:
                barrier_work.wait()

            if prof: prof.mark('barrier')

            # Pre-combine barrier: ensure all ranks' reduce kernels from the
            # previous iteration are done before any rank starts new combine
            # RDMA writes that overwrite the combine recv buffer.
            # Note: torch.cuda.synchronize() is required to flush the default
            # stream (where gather+unsort runs), but the reduce from the
            # PREVIOUS iteration was already sync'd by the pre-dispatch barrier.
            # We still need the cross-rank dist.barrier() though, because fast
            # ranks might skip ahead to iteration N+1's combine while slow ranks
            # are still in iteration N's reduce.
            torch.cuda.synchronize()
            dist.barrier(group=self.group)

            if prof: prof.mark('pre_efa')

            # Iter 39: Fused C++ combine data transfer — computes offsets internally
            # For combine: we send recv_counts tokens (what we received), receive send_counts tokens (what we sent)
            # Iter 45: ll_efa_combine_data now adds comm_stream→default_stream event sync
            self.runtime.ll_efa_combine_data(
                recv_counts, send_counts,
                rdma_combine_send[:max(1, combine_send_bytes)],
                rdma_combine_recv,
                hidden, bytes_per_elem, combine_slot_size)

            if prof: prof.mark('efa_a2a')

            # Iter 32: Pass RDMA recv buffer directly to reduce kernel (no torch.cat)
            # The kernel handles slot-based addressing via combine_slot_size + send_cumsum
            combine_recv_flat = rdma_combine_recv.view(torch.bfloat16) if total_send > 0 else torch.empty((0, hidden), dtype=torch.bfloat16, device='cuda')

        elif use_efa:
            # === EFA path for combine ===
            half_rdma = self._half_rdma
            data_offset = ll_cache['efa_data_offset']
            bytes_per_elem = 2  # bf16
            num_max = stash.get('num_max_dispatch_tokens_per_rank', num_max_dispatch_tokens_per_rank)

            combine_send_bytes = total_recv * hidden * bytes_per_elem
            combine_recv_bytes = total_send * hidden * bytes_per_elem

            # Fixed-slot layout for combine recv
            combine_slot_size = num_max * hidden * bytes_per_elem
            total_combine_recv_region = num_ranks * combine_slot_size

            rdma_combine_recv = self.runtime.get_local_buffer_tensor(
                torch.uint8, half_rdma + data_offset, True)[:total_combine_recv_region]

            # Iter 33: combine_send_flat already in RDMA send buffer (written by gather+unsort above)
            if prof: prof.mark('copy_to_rdma')

            # Iter 45: Wait for async barrier (started before gather+unsort).
            if barrier_work is not None:
                barrier_work.wait()

            if prof: prof.mark('barrier')

            # Pre-combine barrier (see use_pipeline path for rationale).
            torch.cuda.synchronize()
            dist.barrier(group=self.group)

            if prof: prof.mark('pre_efa')

            # Iter 39: Fused C++ combine data transfer — computes offsets internally
            # Iter 45: ll_efa_combine_data now adds comm_stream→default_stream event sync
            self.runtime.ll_efa_combine_data(
                recv_counts, send_counts,
                rdma_combine_send[:max(1, combine_send_bytes)],
                rdma_combine_recv,
                hidden, bytes_per_elem, combine_slot_size)

            if prof: prof.mark('efa_a2a')

            # Iter 32: Pass RDMA recv buffer directly to reduce kernel (no torch.cat)
            combine_recv_flat = rdma_combine_recv.view(torch.bfloat16) if total_send > 0 else torch.empty((0, hidden), dtype=torch.bfloat16, device='cuda')
        else:
            # === NCCL fallback path ===
            combine_slot_size = 0  # contiguous flat layout
            if ll_cache is not None and total_send <= ll_cache['combine_recv_max'].size(0):
                combine_recv_flat = ll_cache['combine_recv_max'][:max(1, total_send)]
            else:
                combine_recv_flat = torch.empty((max(1, total_send), hidden), dtype=torch.bfloat16, device='cuda')

            send_splits = [r * hidden for r in recv_counts]
            recv_splits = [s * hidden for s in send_counts]

            if prof: prof.mark('copy_to_rdma'); prof.mark('barrier'); prof.mark('pre_efa')

            dist.all_to_all_single(
                combine_recv_flat[:total_send].view(-1) if total_send > 0 else combine_recv_flat.view(-1)[:0],
                combine_send_flat.view(-1),
                output_split_sizes=recv_splits, input_split_sizes=send_splits,
                group=self._get_gpu_group())

            if prof: prof.mark('efa_a2a'); prof.mark('extract_recv')

        # ------------------------------------------------------------------
        # Step 3: Fused weighted reduction with pre-allocated f32 accumulator
        # ------------------------------------------------------------------
        # Iter 34: Only zero combined_x when total_send == 0 (no reduce kernel to overwrite it).
        # When total_send > 0, ll_combine_weighted_reduce_v2 zeros combined_x_f32 and then
        # f32_to_bf16_kernel overwrites ALL num_tokens*hidden elements of combined_x.
        if out is not None:
            combined_x = out
            if total_send == 0:
                combined_x.zero_()
        elif ll_cache is not None:
            combined_x = ll_cache['combined_x']
            if total_send == 0:
                combined_x.zero_()
        else:
            combined_x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')

        if total_send > 0:
            sorted_send_token_ids = stash['sorted_send_token_ids']
            global_expert_ids = stash['global_expert_ids']

            # Iter 33: Build send_cumsum via GPU cumsum (no Python loop)
            combine_send_cumsum = None
            if combine_slot_size > 0 and ll_cache is not None:
                cumsum_buf = ll_cache['combine_send_cumsum_i32']
                # cumsum_buf[0] is 0 from init; cumsum fills [1:]
                torch.cumsum(stash['send_counts_tensor'], dim=0, out=cumsum_buf[1:])
                combine_send_cumsum = cumsum_buf

            if ll_cache is not None:
                deep_ep_cpp.ll_combine_weighted_reduce_v2(
                    combine_recv_flat if combine_slot_size > 0 else combine_recv_flat[:total_send],
                    sorted_send_token_ids[:total_send],
                    global_expert_ids[:total_send],
                    topk_idx,
                    topk_weights,
                    combined_x,
                    ll_cache['combined_x_f32'],
                    combine_send_cumsum,
                    total_send, num_tokens, hidden, num_topk,
                    num_ranks, combine_slot_size)
            else:
                deep_ep_cpp.ll_combine_weighted_reduce(
                    combine_recv_flat[:total_send],
                    sorted_send_token_ids[:total_send],
                    global_expert_ids[:total_send],
                    topk_idx,
                    topk_weights,
                    combined_x,
                    total_send, num_tokens, hidden, num_topk)

        if prof:
            prof.mark('end')
            prof.finish_iter(f'send={total_send} recv={total_recv}')

        # Post-combine barrier removed in Iter 47: redundant with pre-dispatch
        # barrier (torch.cuda.synchronize + NCCL collective) at start of next dispatch.

        event = EventOverlap(EventHandle())
        hook = (lambda: None) if return_recv_hook else None
        return combined_x, event, hook

    def low_latency_update_mask_buffer(self, rank_to_mask: int, mask: bool = False):
        """
        Mask (unmask) a rank during communication (dispatch, combine, and clean).
        Python implementation using a local mask tensor.
        """
        if not hasattr(self, '_ll_mask_status'):
            self._ll_mask_status = torch.zeros(self.group_size, dtype=torch.int, device='cuda')
        self._ll_mask_status[rank_to_mask] = 1 if mask else 0

    def low_latency_query_mask_buffer(self, mask_status: torch.Tensor):
        """
        Query the mask status of all ranks.
        Python implementation.
        """
        if not hasattr(self, '_ll_mask_status'):
            self._ll_mask_status = torch.zeros(self.group_size, dtype=torch.int, device='cuda')
        mask_status.copy_(self._ll_mask_status)

    def low_latency_clean_mask_buffer(self):
        """
        Clean the mask buffer.
        Python implementation.
        """
        if hasattr(self, '_ll_mask_status'):
            self._ll_mask_status.zero_()

    def get_next_low_latency_combine_buffer(self, handle: object):
        """
        Get the raw registered RDMA buffer tensor for next low-latency combine, so that the next combine kernel can skip the copying.

        Arguments:
            handle: the communication handle given by the `dispatch` function.

        Returns:
            buffer: the raw RDMA low-latency buffer as a BF16 PyTorch tensor with shape
                `[num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden]`, you should fill this buffer
                by yourself.
        """
        src_info, layout_range, num_max_dispatch_tokens_per_rank, hidden, num_experts = handle
        num_local_experts = num_experts // self.group_size
        N = num_max_dispatch_tokens_per_rank * self.group_size
        # Allocate or return the combine buffer
        if not hasattr(self, '_ll_combine_buffer'):
            self._ll_combine_buffer = [None, None]
        idx = getattr(self, '_ll_buffer_idx', 0)
        if self._ll_combine_buffer[idx] is None or self._ll_combine_buffer[idx].shape != (num_local_experts, N, hidden):
            self._ll_combine_buffer[idx] = torch.zeros((num_local_experts, N, hidden), dtype=torch.bfloat16, device='cuda')
        return self._ll_combine_buffer[idx]
