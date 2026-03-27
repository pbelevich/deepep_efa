#pragma once

// Forcibly disable NDEBUG
#ifdef NDEBUG
#undef NDEBUG
#endif

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>

#include <tuple>
#include <utility>
#include <vector>

#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#ifdef ENABLE_EFA
#include "transport/efa_worker.h"
#include "transport/ll_pipeline.h"
#endif

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME deep_ep_cpp
#endif

namespace shared_memory {

union MemHandleInner {
    cudaIpcMemHandle_t cuda_ipc_mem_handle;
    CUmemFabricHandle cu_mem_fabric_handle;
};

struct MemHandle {
    MemHandleInner inner;
    size_t size;
};

constexpr size_t HANDLE_SIZE = sizeof(MemHandle);

class SharedMemoryAllocator {
public:
    SharedMemoryAllocator(bool use_fabric);
    void malloc(void** ptr, size_t size);
    void free(void* ptr);
    void get_mem_handle(MemHandle* mem_handle, void* ptr);
    void open_mem_handle(void** ptr, MemHandle* mem_handle);
    void close_mem_handle(void* ptr);

private:
    bool use_fabric;
};
}  // namespace shared_memory

namespace deep_ep {

struct Buffer {
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "The number of maximum NVLink peers must be 8");

private:
    // Low-latency mode buffer
    int low_latency_buffer_idx = 0;
    bool low_latency_mode = false;

    // NVLink Buffer
    int64_t num_nvl_bytes;
    void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    void** buffer_ptrs_gpu = nullptr;

    // NVSHMEM Buffer
    int64_t num_rdma_bytes;
    void* rdma_buffer_ptr = nullptr;

    // Shrink mode buffer
    bool enable_shrink = false;
    int* mask_buffer_ptr = nullptr;
    int* sync_buffer_ptr = nullptr;

    // Device info and communication
    int device_id;
    int num_device_sms;
    int rank, rdma_rank, nvl_rank;
    int num_ranks, num_rdma_ranks, num_nvl_ranks;
    shared_memory::MemHandle ipc_handles[NUM_MAX_NVL_PEERS];

    // Stream for communication
    at::cuda::CUDAStream comm_stream;

    // EFA transport manager (replaces NVSHMEM for internode RDMA)
#ifdef ENABLE_EFA
    std::unique_ptr<efa::EfaWorkerManager> efa_manager_;
    // RDMA rank of this GPU (= rank / 8 for internode, or rank for low-latency)
    int efa_rdma_rank_ = -1;
    // VMM allocation for the RDMA buffer (used instead of cudaMalloc for DMA-BUF support)
    CUmemGenericAllocationHandle rdma_vmm_handle_ = 0;
    CUdeviceptr rdma_vmm_ptr_ = 0;
    size_t rdma_vmm_size_ = 0;

    // GDRCopy-signaled LL pipeline worker
    std::unique_ptr<efa::LLPipelineWorker> ll_pipeline_;
    bool ll_pipeline_initialized_ = false;

    // Cooperative kernel scratch buffers (allocated once, reused)
    struct CoopScratch {
        // Dispatch send: per-token-expert offset and per-expert prefix sums
        uint32_t* token_offset = nullptr;       // [max_tokens * max_topk]
        uint32_t* expert_offsets = nullptr;      // [num_experts]
        uint32_t* num_routed = nullptr;          // [dp_groups * num_experts] — GDR-mapped

        // Grid-wide atomics
        uint32_t* grid_counter = nullptr;        // [1] — reset to 0 each call
        uint32_t* grid_counter_2 = nullptr;      // [1] — second counter for recv kernel

        // NVLink sync (trivial for NODE_SIZE=1)
        uint32_t* sync_counter = nullptr;        // [1] — persistent epoch counter
        uint32_t** sync_ptrs = nullptr;          // null for NODE_SIZE=1
        uint8_t** recv_ptrs = nullptr;           // null for NODE_SIZE=1
        uint8_t** send_ptrs = nullptr;           // null for NODE_SIZE=1

        // Dispatch recv metadata (computed by CPU via GDRCopy MMIO, read by GPU)
        // GdrVec allows the CPU worker to write directly via MMIO (~1us) instead
        // of synchronous cudaMemcpy (~5-15us per call). The GPU reads these via
        // normal global memory loads.
        efa::GdrVec<uint32_t> gdr_source_rank;           // [max_recv_tokens]
        efa::GdrVec<uint32_t> gdr_source_offset;         // [max_recv_tokens]
        efa::GdrVec<uint32_t> gdr_padded_index;          // [max_recv_tokens]
        efa::GdrVec<uint32_t> gdr_tokens_per_expert;     // [experts_per_rank]
        efa::GdrVec<uint32_t> gdr_num_recv_tokens;       // [2]: [total, efa_only]
        efa::GdrVec<uint32_t> gdr_combine_send_offset;   // [max_recv_tokens]

        // GDR-mapped num_routed for zero-copy CPU read (GPU writes via kernel)
        efa::GdrVec<uint32_t> gdr_num_routed;            // [num_experts]

        // Legacy raw pointers (aliases to GdrVec device_ptr() for kernel launches)
        uint32_t* source_rank = nullptr;         // = gdr_source_rank.device_ptr()
        uint32_t* source_offset = nullptr;       // = gdr_source_offset.device_ptr()
        uint32_t* padded_index = nullptr;        // = gdr_padded_index.device_ptr()
        uint32_t* tokens_per_expert = nullptr;   // = gdr_tokens_per_expert.device_ptr()
        uint32_t* num_recv_tokens = nullptr;     // = gdr_num_recv_tokens.device_ptr()
        uint32_t* combine_send_offset = nullptr; // = gdr_combine_send_offset.device_ptr()

        // Combine token counter (GPU-only, not CPU-written)
        uint32_t* combine_token_counter = nullptr; // [1]

        // GDR flags for cooperative kernel ↔ CPU communication
        // (separate from the LLPipelineWorker's flags — the coop kernels
        //  need additional flags beyond pack_done/recv_done)
        efa::GdrFlag dispatch_route_done;        // GPU→CPU: routing counts ready
        efa::GdrFlag dispatch_pack_done;         // GPU→CPU: dispatch send buffer packed
        efa::GdrFlag tx_ready;                   // CPU→GPU: send buffer available
        efa::GdrFlag num_recv_tokens_flag;       // CPU→GPU: recv metadata ready
        efa::GdrFlag dispatch_recv_flag;         // CPU→GPU: all dispatch EFA data arrived
        efa::GdrFlag dispatch_recv_done;         // GPU→CPU: recv kernel complete
        efa::GdrFlag combine_send_done;          // GPU→CPU: combine pack complete
        efa::GdrFlag combine_recv_flag;          // CPU→GPU: all combine EFA data arrived
        efa::GdrFlag combine_recv_done;          // GPU→CPU: combine recv kernel complete

        // GDR counters for auto-signaling GPU when RDMA completions reach target
        efa::GdrCounter dispatch_rdma_counter;   // auto-sets dispatch_recv_flag
        efa::GdrCounter combine_rdma_counter;    // auto-sets combine_recv_flag

        // Allocation tracking
        int max_tokens_allocated = 0;
        int max_topk_allocated = 0;
        int num_experts_allocated = 0;
        int max_recv_tokens_allocated = 0;
        bool initialized = false;
    };
    CoopScratch coop_scratch_;
#endif

    // After IPC/NVSHMEM synchronization, this flag will be true
    bool available = false;

    // Whether explicit `destroy()` is required.
    bool explicitly_destroy;
    // After `destroy()` be called, this flag will be true
    bool destroyed = false;

    // Barrier signals
    int* barrier_signal_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
    int** barrier_signal_ptrs_gpu = nullptr;

    // Workspace
    void* workspace = nullptr;

    // Host-side MoE info
    volatile int* moe_recv_counter = nullptr;
    int* moe_recv_counter_mapped = nullptr;

    // Host-side expert-level MoE info
    volatile int* moe_recv_expert_counter = nullptr;
    int* moe_recv_expert_counter_mapped = nullptr;

    // Host-side RDMA-level MoE info
    volatile int* moe_recv_rdma_counter = nullptr;
    int* moe_recv_rdma_counter_mapped = nullptr;

    shared_memory::SharedMemoryAllocator shared_memory_allocator;

public:
    Buffer(int rank,
           int num_ranks,
           int64_t num_nvl_bytes,
           int64_t num_rdma_bytes,
           bool low_latency_mode,
           bool explicitly_destroy,
           bool enable_shrink,
           bool use_fabric);

    ~Buffer() noexcept(false);

    bool is_available() const;

    bool is_internode_available() const;

    int get_num_rdma_ranks() const;

    int get_rdma_rank() const;

    int get_root_rdma_rank(bool global) const;

    int get_local_device_id() const;

    pybind11::bytearray get_local_ipc_handle() const;

    pybind11::bytearray get_local_nvshmem_unique_id() const;

    torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

    torch::Stream get_comm_stream() const;

    void sync(const std::vector<int>& device_ids,
              const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
              const std::optional<pybind11::bytearray>& root_unique_id_opt);

    void destroy();

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>> get_dispatch_layout(
        const torch::Tensor& topk_idx,
        int num_experts,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>>
    intranode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_idx,
        const torch::Tensor& rank_prefix_matrix,
        const torch::Tensor& channel_prefix_matrix,
        const torch::Tensor& send_head,
        const Config& config,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::vector<int>,
               torch::Tensor,
               torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<torch::Tensor>,
               std::optional<EventHandle>>
    internode_dispatch(const torch::Tensor& x,
                       const std::optional<torch::Tensor>& x_scales,
                       const std::optional<torch::Tensor>& topk_idx,
                       const std::optional<torch::Tensor>& topk_weights,
                       const std::optional<torch::Tensor>& num_tokens_per_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                       const torch::Tensor& is_token_in_rank,
                       const std::optional<torch::Tensor>& num_tokens_per_expert,
                       int cached_num_recv_tokens,
                       int cached_num_rdma_recv_tokens,
                       const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                       const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                       const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                       int expert_alignment,
                       int num_worst_tokens,
                       const Config& config,
                       std::optional<EventHandle>& previous_event,
                       bool async,
                       bool allocate_on_comm_stream);

    std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
        const torch::Tensor& x,
        const std::optional<torch::Tensor>& topk_weights,
        const std::optional<torch::Tensor>& bias_0,
        const std::optional<torch::Tensor>& bias_1,
        const torch::Tensor& src_meta,
        const torch::Tensor& is_combined_token_in_rank,
        const torch::Tensor& rdma_channel_prefix_matrix,
        const torch::Tensor& rdma_rank_prefix_sum,
        const torch::Tensor& gbl_channel_prefix_matrix,
        const torch::Tensor& combined_rdma_head,
        const torch::Tensor& combined_nvl_head,
        const Config& config,
        std::optional<EventHandle>& previous_event,
        bool async,
        bool allocate_on_comm_stream);

    void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

    std::tuple<torch::Tensor,
               std::optional<torch::Tensor>,
               torch::Tensor,
               torch::Tensor,
               torch::Tensor,
               std::optional<EventHandle>,
               std::optional<std::function<void()>>>
    low_latency_dispatch(const torch::Tensor& x,
                         const torch::Tensor& topk_idx,
                         const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                         const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                         int num_max_dispatch_tokens_per_rank,
                         int num_experts,
                         bool use_fp8,
                         bool round_scale,
                         bool use_ue8m0,
                         bool async,
                         bool return_recv_hook);

    std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
        const torch::Tensor& x,
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        const torch::Tensor& src_info,
        const torch::Tensor& layout_range,
        const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
        int num_max_dispatch_tokens_per_rank,
        int num_experts,
        bool use_logfmt,
        bool zero_copy,
        bool async,
        bool return_recv_hook,
        const std::optional<torch::Tensor>& out = std::nullopt);

    torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const;

    void low_latency_update_mask_buffer(int rank_to_mask, bool mask);

    void low_latency_query_mask_buffer(const torch::Tensor& mask_status);

    void low_latency_clean_mask_buffer();

#ifdef ENABLE_EFA
    // Initialize EFA transport: create endpoint, register RDMA buffer, exchange addresses
    void init_efa(const pybind11::function& allgather_fn);

    // EFA all-to-all: variable-length RDMA transfer between all RDMA ranks
    // send_buf/recv_buf: GPU buffers
    // send_sizes/recv_sizes: per-RDMA-rank byte counts (host vectors)
    // send_offsets/recv_offsets: per-RDMA-rank byte offsets into send_buf/recv_buf (host vectors)
    // Uses the NVL rank 0 GPU on each node to do the RDMA transfer.
    // All GPUs within a node share the same RDMA buffer (via RDMA rank mapping).
    void efa_all_to_all(const torch::Tensor& send_buf,
                        const std::vector<int64_t>& send_sizes,
                        const std::vector<int64_t>& send_offsets,
                        const torch::Tensor& recv_buf,
                        const std::vector<int64_t>& recv_sizes,
                        const std::vector<int64_t>& recv_offsets,
                        int override_tag = -1,       // -1 = auto-allocate, else use this tag
                        int override_baseline = -1,  // -1 = auto-capture, else use this baseline
                        // Iter 44: In-band token count exchange
                        const std::vector<int32_t>* send_token_counts = nullptr,  // per-peer token counts to embed in imm data
                        std::vector<int32_t>* recv_token_counts = nullptr,        // filled with per-peer token counts from imm data
                        int packed_bytes_per_token = 0);  // needed for two-phase wait when recv_token_counts is set

    // EFA RDMA count exchange: scatter per-rank send_counts to all peers via RDMA,
    // wait for recv_counts from all peers. Uses the count region at offset 0 in RDMA buffer.
    // Returns (recv_counts_list, total_recv, send_counts_list, total_send).
    // The cuda_event_ptr is a CUDA event recorded after the count kernel completed;
    // this function syncs that event before reading send_counts from GPU.
    std::tuple<std::vector<int32_t>, int, std::vector<int32_t>, int>
    efa_count_exchange(const torch::Tensor& send_counts_gpu,
                       int64_t cuda_event_ptr);

    // LL Pipeline: Initialize the GDRCopy-signaled async RDMA worker
    void init_ll_pipeline(int packed_bytes_per_token,
                          int num_max_dispatch_tokens_per_rank);

    // LL Pipeline: Start async dispatch RDMA (called after GPU pack kernel launched)
    // send_counts/recv_counts: from .tolist() after NCCL count exchange (CPU-side)
    void ll_pipeline_start_dispatch(const std::vector<int32_t>& send_counts,
                                    const std::vector<int32_t>& recv_counts);

    // LL Pipeline: Wait for dispatch RDMA to complete
    void ll_pipeline_wait_dispatch();

    // LL Pipeline: Start async combine RDMA
    // send_counts/recv_counts: known from dispatch phase
    void ll_pipeline_start_combine(const std::vector<int32_t>& send_counts,
                                   const std::vector<int32_t>& recv_counts);

    // LL Pipeline: Wait for combine RDMA to complete
    void ll_pipeline_wait_combine();

    // LL Pipeline: Execute EFA-native barrier (replaces dist.barrier())
    void ll_pipeline_barrier();

    // Get GDR device pointers for GPU kernel signaling
    // Returns (pack_done_ptr, send_counts_ptr) as int64 for dispatch
    std::tuple<int64_t, int64_t> ll_pipeline_get_dispatch_gdr_ptrs();
    // Returns (pack_done_ptr, send_counts_ptr) as int64 for combine
    std::tuple<int64_t, int64_t> ll_pipeline_get_combine_gdr_ptrs();

    // Iter 39: Fused LL dispatch data transfer — computes offsets + calls efa_all_to_all internally
    // Eliminates Python-side offset computation overhead (~35us)
    // send_counts/recv_counts: from pinned CPU .tolist() (already fast)
    // packed_bytes_per_token, slot_size: pre-computed constants
    // send_buf: the packed send buffer (already in RDMA-registered region)
    // recv_buf: the RDMA recv region (half_rdma + data_offset)
    void ll_efa_dispatch_data(const std::vector<int32_t>& send_counts,
                              const std::vector<int32_t>& recv_counts,
                              const torch::Tensor& send_buf,
                              const torch::Tensor& recv_buf,
                              int packed_bytes_per_token,
                              int64_t slot_size);

    // Iter 42: Fused LL dispatch data transfer v2 — reads counts from pinned CPU tensors
    // Eliminates Python-side nccl_stream.synchronize() + .tolist() + sum() overhead
    // The comm_stream sync inside efa_all_to_all waits for NCCL + D2H + pack_done
    // Returns (send_counts, recv_counts, total_send, total_recv) for Python-side unpack/stash
    std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
    ll_efa_dispatch_data_v2(const torch::Tensor& send_counts_pin,
                            const torch::Tensor& recv_counts_pin,
                            const torch::Tensor& send_buf,
                            const torch::Tensor& recv_buf,
                            int packed_bytes_per_token,
                            int64_t slot_size);

    // Iter 44: Fused LL dispatch data transfer v3 — NO NCCL count exchange
    // Token counts are sent in-band via EFA imm data (encode_with_tokens).
    // Flow:
    //   1. cudaStreamSynchronize(comm_stream) — waits for D2H of send_counts + pack_done
    //   2. Read send_counts from pinned CPU memory
    //   3. Call efa_all_to_all with send_token_counts → recv_token_counts extracted from imm
    //   4. Copy recv_counts to recv_counts_gpu tensor
    //   5. Return send_counts, recv_counts (from imm), total_send, total_recv
    // Saves ~240us by eliminating NCCL all_to_all_single for count exchange.
    std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
    ll_efa_dispatch_data_v3(const torch::Tensor& send_counts_pin,
                            const torch::Tensor& send_buf,
                            const torch::Tensor& recv_buf,
                            const torch::Tensor& recv_counts_gpu,  // GPU tensor to fill with recv_counts
                            int packed_bytes_per_token,
                            int64_t slot_size);

    // Iter 42: Fused LL combine data transfer v2 — reads counts from pinned CPU tensors
    std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
    ll_efa_combine_data_v2(const torch::Tensor& send_counts_pin,
                           const torch::Tensor& recv_counts_pin,
                           const torch::Tensor& send_buf,
                           const torch::Tensor& recv_buf,
                           int hidden,
                           int bytes_per_elem,
                           int64_t combine_slot_size);

    // Iter 39: Fused LL combine data transfer
    void ll_efa_combine_data(const std::vector<int32_t>& send_counts,
                              const std::vector<int32_t>& recv_counts,
                              const torch::Tensor& send_buf,
                              const torch::Tensor& recv_buf,
                              int hidden,
                              int bytes_per_elem,
                              int64_t combine_slot_size,
                              int override_tag = -1,
                              int override_baseline = -1);

    // Iter 41: EFA RDMA barrier — replaces dist.barrier() for inter-rank sync
    // Each rank sends zero-byte RDMA writes with imm data to all peers,
    // then waits for all peers to send their barrier signals.
    // Faster than NCCL barrier (~30us vs ~147us).
    void efa_barrier();

    // Iter 43: Lightweight RDMA immediate-data barrier
    // Zero-byte RDMA writes with imm data — no data transfer overhead.
    // ~30-50us vs NCCL barrier (~147us) or efa_barrier (~50us with 8KB data).
    // Does NOT synchronize CUDA streams — caller must ensure GPU sync.
    // Returns (next_tag, next_baseline) for the subsequent efa_all_to_all
    // to avoid baseline inflation race.
    std::pair<int, int> efa_imm_barrier();

    // Iter 43: Fused RDMA count exchange + data transfer for LL dispatch.
    // Replaces NCCL all_to_all_single for count exchange with RDMA scatter.
    // Flow:
    //   1. Wait for count_event on comm_stream (count kernel completed)
    //   2. D2H copy send_counts from GPU to host
    //   3. H2D copy send_counts into RDMA send buffer count region
    //   4. RDMA scatter counts to all peers (one write per peer, ~64 bytes each)
    //   5. Overlap: wait for pack_done_event (pack kernel on default stream)
    //   6. Wait for all peers' counts to arrive (CQ polling)
    //   7. Read recv_counts from RDMA buffer (NC kernel to avoid L2 stale data)
    //   8. Compute offsets, call efa_all_to_all for data transfer
    // Parameters:
    //   send_counts_gpu: GPU tensor written by count kernel (int32, num_ranks)
    //   count_event_ptr: CUDA event recorded after count kernel (int64 cast)
    //   pack_done_event_ptr: CUDA event recorded after pack kernel (int64 cast)
    //   send_buf: packed data send buffer (in RDMA-registered region)
    //   recv_buf: packed data recv buffer (in RDMA-registered region, recv half)
    //   packed_bytes_per_token: bytes per packed token
    //   slot_size: per-rank recv slot size (num_max_dispatch_tokens_per_rank * packed_bpt)
    // Returns: (send_counts, recv_counts, total_send, total_recv)
    std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
    ll_efa_rdma_dispatch(const torch::Tensor& send_counts_gpu,
                         int64_t count_event_ptr,
                         int64_t pack_done_event_ptr,
                         const torch::Tensor& send_buf,
                         const torch::Tensor& recv_buf,
                         int packed_bytes_per_token,
                         int64_t slot_size);

    // =========================================================================
    // Cooperative kernel wrappers (Iter 48)
    // =========================================================================

    // Allocate scratch buffers for cooperative kernels
    void coop_init(int max_tokens, int num_topk, int num_experts,
                   int max_recv_tokens);

    // Get GDR device pointers for cooperative kernels
    // Returns dict of (name → device_ptr as int64) for all GDR flags + scratch buffers
    std::vector<int64_t> coop_get_ptrs();

    // Launch cooperative dispatch send kernel
    // Returns 0 on success
    int coop_dispatch_send(
        const torch::Tensor& x,                 // [num_tokens, hidden] input tokens
        const std::optional<torch::Tensor>& x_scales, // FP8 scales (optional)
        const torch::Tensor& topk_idx,           // [num_tokens, topk]
        const torch::Tensor& topk_weights,       // [num_tokens, topk]
        int num_experts,
        int num_tokens,
        int hidden_dim,
        int hidden_dim_scale,
        int dp_size,
        int64_t stream_ptr);

    // Launch cooperative dispatch recv kernel
    // Must be called AFTER coop_set_recv_metadata() and after dispatch RDMA completes
    int coop_dispatch_recv(
        torch::Tensor& out_x,                   // [max_recv, hidden] output
        std::optional<torch::Tensor>& out_x_scales, // FP8 scales output (optional)
        torch::Tensor& out_num_tokens,           // [1] total recv tokens
        int num_experts,
        int hidden_dim,
        int src_elemsize,                        // elemsize of source data in buffer (1=FP8, 2=BF16)
        int src_scale_elemsize,                  // elemsize of source scales in buffer (4=float, 0=none)
        int hidden_dim_scale,                    // number of scale elements
        int64_t stream_ptr);

    // Set recv metadata arrays for dispatch_recv kernel
    // Called from Python after computing routing from recv_counts
    void coop_set_recv_metadata(
        const std::vector<uint32_t>& source_rank,
        const std::vector<uint32_t>& source_offset,
        const std::vector<uint32_t>& padded_index,
        const std::vector<uint32_t>& combine_send_offset,
        const std::vector<uint32_t>& tokens_per_expert,
        int total_recv_tokens,
        int efa_recv_tokens);

    // Launch cooperative combine send kernel
    int coop_combine_send(
        const torch::Tensor& expert_x,          // [num_recv, hidden] expert output
        int hidden_dim,
        int dp_size,
        int64_t stream_ptr);

    // Launch cooperative combine recv kernel
    int coop_combine_recv(
        const torch::Tensor& topk_idx,
        const torch::Tensor& topk_weights,
        torch::Tensor& combined_x,              // [num_tokens, hidden] output
        int num_experts,
        int num_tokens,
        int hidden_dim,
        bool accumulate,
        int64_t stream_ptr);

    // Read per-rank send counts from GDR-mapped num_routed
    // Called by Python after dispatch_route_done flag is set
    std::vector<int32_t> coop_read_send_counts(int num_experts, int experts_per_rank);
    // Return GPU tensor view of num_routed (zero-copy)
    torch::Tensor coop_get_num_routed_tensor(int num_experts);

    // Signal tx_ready flag (send buffer available for cooperative kernel)
    void coop_signal_tx_ready();

    // Wait for dispatch_route_done flag (blocking spin)
    void coop_wait_route_done();

    // Signal num_recv_tokens_flag (recv metadata ready for cooperative recv kernel)
    void coop_signal_recv_metadata_ready();

    // Signal dispatch_recv_flag (dispatch RDMA complete)
    void coop_signal_dispatch_recv_ready();

    // Signal combine_recv_flag (combine RDMA complete)
    void coop_signal_combine_recv_ready();

    // Wait for dispatch_recv_done flag (blocking spin)
    void coop_wait_dispatch_recv_done();

    // Wait for combine_send_done flag (blocking spin)
    void coop_wait_combine_send_done();

    // Wait for combine_recv_done flag (blocking spin)
    void coop_wait_combine_recv_done();

    // Reset all cooperative kernel GDR flags for next iteration
    void coop_reset_flags();

    // Iter 61: Reset tag counters between benchmark phases to prevent tag wrapping
    void coop_reset_tags();

    // Cooperative RDMA dispatch transfer (Iter 49)
    // Uses efa_all_to_all with coop token_stride and slot-based recv layout
    void coop_efa_dispatch_rdma(
        const std::vector<int32_t>& send_counts,
        const std::vector<int32_t>& recv_counts,
        int coop_token_stride,
        int max_tokens_per_rank);

    // Cooperative RDMA combine transfer (Iter 49)
    void coop_efa_combine_rdma(
        const std::vector<int32_t>& send_counts,
        const std::vector<int32_t>& recv_counts,
        int combine_token_dim,
        const std::vector<int64_t>& dst_group_offsets_tokens);

    // =========================================================================
    // Iter 50: Cooperative Worker Thread API
    //
    // These methods configure and trigger the LLPipelineWorker to handle the
    // entire cooperative dispatch+combine flow on a dedicated CPU thread,
    // overlapped with cooperative GPU kernels.
    // =========================================================================

    // Configure coop worker with kernel parameters and GPU pointers.
    // Must be called before start_coop_dispatch().
    void coop_worker_init(int num_experts, int num_topk,
                          int max_tokens_per_rank,
                          int coop_token_stride,
                          int combine_token_dim);

    // Start cooperative dispatch on worker thread (returns immediately).
    // Worker will: route exchange → metadata → RDMA → signal.
    void start_coop_dispatch();

    // Wait for cooperative dispatch to complete.
    void wait_coop_dispatch_done();

    // Start cooperative combine on worker thread (returns immediately).
    void start_coop_combine();

    // Wait for cooperative combine to complete.
    void wait_coop_combine_done();

    // Get per-rank send/recv counts computed by the worker (for Python stash).
    // Only valid after wait_coop_dispatch_done().
    std::tuple<std::vector<int32_t>, std::vector<int32_t>, int, int>
    coop_worker_get_counts();
#endif
};

}  // namespace deep_ep
