# DeepEP-EFA

DeepEP-EFA is a reimplementation of [DeepEP](https://github.com/deepseek-ai/DeepEP) that replaces the NVSHMEM/IBGDA transport layer with libfabric/EFA, enabling Mixture-of-Experts (MoE) expert-parallel communication on AWS clusters equipped with Elastic Fabric Adapter (EFA) networking. It preserves the DeepEP Python interface so that existing model code using `deep_ep.Buffer` works without modification.

## Motivation

The original DeepEP relies on NVSHMEM with IBGDA (InfiniBand GPUDirect Async) for internode RDMA, which requires InfiniBand hardware. AWS P5en instances use EFA instead of InfiniBand, and NVSHMEM does not support EFA. DeepEP-EFA solves this by implementing the entire internode and low-latency transport layer using libfabric (the EFA provider) and GDRCopy, following architectural patterns from [pplx-garden](https://github.com/pplx-ai/pplx-garden).

## Approach

### What changed

- **Internode transport**: NVSHMEM symmetric memory and IBGDA device-side RDMA are replaced with CPU-mediated RDMA via libfabric `fi_writemsg` with `FI_REMOTE_CQ_DATA`. A dedicated worker thread on each GPU handles all EFA operations.
- **Low-latency kernels**: The original single-kernel design (where the GPU kernel drives RDMA directly through NVSHMEM) is replaced with a cooperative kernel + worker thread architecture. Four cooperative CUDA kernels handle GPU-side packing/unpacking, while the CPU worker thread manages RDMA transfers in parallel.
- **Signaling**: GDRCopy-mapped flags and counters replace NVSHMEM remote atomics and signals. The CPU worker writes to GDRCopy-mapped GPU memory via MMIO (write-combining), providing immediate visibility to GPU kernels without CUDA API calls.
- **Multi-NIC sharding**: Data RDMA writes are sharded across multiple EFA adapters per GPU (2 NICs per GPU on P5en) for higher aggregate bandwidth. Route scatter uses a single NIC.
- **Immediate data counters**: Each RDMA write carries a 32-bit immediate value encoding `[tag:10 | src_rank:6 | count:16]`, allowing the receiver to track completions per-tag and per-rank without additional signaling messages.

### What stayed the same

- **Intranode code**: The NVLink-based intranode path (`intranode.cu`, `layout.cu`, barrier_block mechanism) is reused as-is. It has zero NVSHMEM/IBGDA dependencies — pure NVLink IPC + CUDA system-scope atomics.
- **Python interface**: `deep_ep.Buffer` API is preserved. `dispatch`, `combine`, `low_latency_dispatch`, `low_latency_combine`, `get_dispatch_layout`, and all configuration methods work unchanged.
- **Config and layout**: `LowLatencyLayout`, `Config`, buffer sizing hints, and the FP8/BF16 data path are unchanged.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Python Layer                         │
│  deep_ep.Buffer  (dispatch / combine / low_latency_*)   │
└──────────────────────────┬──────────────────────────────┘
                           │ pybind11
┌──────────────────────────▼──────────────────────────────┐
│                    C++ Host Layer                         │
│  deep_ep.cpp — Buffer class, coop_dispatch/combine_*     │
│  CoopScratch — GDR flags, counters, grid sync state      │
└─────┬───────────────────────────────────┬───────────────┘
      │                                   │
      ▼                                   ▼
┌─────────────────────┐    ┌──────────────────────────────┐
│  Cooperative CUDA    │    │  LLPipelineWorker Thread      │
│  Kernels (all SMs)   │    │  (pinned to CPU core)         │
│                      │    │                                │
│  dispatch_send_kernel│◄──►│  EFA Transport (libfabric)    │
│  dispatch_recv_kernel│    │  ├─ rdma_write_with_imm()     │
│  combine_send_kernel │    │  ├─ poll_tx_cq / poll_rx_cq   │
│  combine_recv_kernel │    │  ├─ ImmCounterMap tracking     │
│                      │    │  └─ GdrCounter auto-signaling  │
│  GDR flag polling    │    │                                │
│  (ld_volatile_u32)   │    │  GDRCopy MMIO writes           │
└─────────────────────┘    │  (flag set, metadata upload)   │
                           └──────────────────────────────┘
```

### Dispatch flow (low-latency cooperative path)

1. **GPU kernel** counts routes per expert, writes to GDRCopy-mapped `num_routed` array, sets `dispatch_route_done` flag.
2. **Worker thread** reads `num_routed` via `cudaMemcpy` D2H, then RDMA-scatters route info to all peers (small write, ~1 KB per peer).
3. **Worker thread** waits for all peers' route info (via RX CQ immediate data counters) and GPU pack completion (`dispatch_pack_done` flag) in an overlapped loop.
4. **Worker thread** computes send/recv counts from the all-to-all route exchange, uploads per-token metadata to GPU via GDRCopy MMIO, then issues data RDMA writes sharded across all NICs.
5. **Worker thread** arms a `GdrCounter` that auto-sets a GDRCopy flag when all expected RX completions arrive.
6. **GPU kernel** polls the dispatch_recv flag, then unpacks received data.

### Combine flow

Symmetric to dispatch with reversed data direction: what was received during dispatch is sent back during combine.

## Platform

- **Hardware**: AWS P5en instances with NVIDIA H100 GPUs and EFA networking (32 EFA adapters per node, 2 per GPU)
- **Software**: CUDA 12.9.1, PyTorch 2.9.0, libfabric 1.44.0 (EFA provider), GDRCopy 2.5.1
- **Target architecture**: SM 9.0 (Hopper)
- **Build**: Docker container with enroot for Slurm execution

## Performance

### Low-latency kernels (cooperative path, pure EFA RDMA)

Measured on P5en with 2 EFA adapters per GPU (~100 GB/s aggregate per-adapter bandwidth). Test configuration: 128 tokens per batch, 7168 hidden, top-8 experts, FP8 dispatch and BF16 combine. Dispatch+combine latency is measured as the total wall-clock time for one dispatch+combine round-trip divided by 2.

| #EP | Nodes | Dispatch+Combine Latency | Dispatch BW | Combine BW |
|:---:|:-----:|:------------------------:|:-----------:|:----------:|
|  16 |   2   |       ~1145 us           |  6.6 GB/s   | 12.7 GB/s  |
|  32 |   4   |       ~1820 us           |  4.1 GB/s   |  8.0 GB/s  |
|  64 |   8   |       ~2584 us           |  2.9 GB/s   |  5.6 GB/s  |

### Normal kernels with NVLink and EFA forwarding

Measured on P5en with NVLink (~450 GB/s per-GPU) and EFA RDMA. Test configuration: 4096 tokens per batch, 7168 hidden, top-4 groups, top-8 experts, FP8 dispatch and BF16 combine.

|   Type    | Dispatch #EP | RDMA BW  |  NVL BW   | Combine #EP | RDMA BW  |  NVL BW   |
|:---------:|:------------:|:--------:|:---------:|:-----------:|:--------:|:---------:|
| Intranode |      8       |   n/a    | 335 GB/s  |      8      |   n/a    | 321 GB/s  |
| Internode |      16      | 15 GB/s  |  48 GB/s  |     16      | 18 GB/s  |  60 GB/s  |

### Comparison with original DeepEP (InfiniBand)

Original DeepEP numbers are from H800 with CX7 InfiniBand 400 Gb/s RDMA (~50 GB/s per NIC). DeepEP-EFA numbers are from P5en with EFA (~100 GB/s aggregate per adapter).

**Normal kernels:**

|   Type    |      | DeepEP (IB) RDMA BW | DeepEP-EFA RDMA BW | DeepEP (IB) NVL BW | DeepEP-EFA NVL BW |
|:---------:|:----:|:-------------------:|:------------------:|:-------------------:|:-----------------:|
| Intranode | EP8  |         n/a         |        n/a         |    153–158 GB/s     |   321–335 GB/s    |
| Internode | EP16 |      43 GB/s        |      15 GB/s       |       n/a           |     48 GB/s       |

Intranode NVLink bandwidth is ~2x higher on P5en H100 vs H800, as expected from the NVLink generation difference (900 GB/s vs 400 GB/s bidirectional). Internode RDMA bandwidth for normal kernels is lower because the internode kernel path is not yet fully optimized for multi-NIC EFA — current implementation uses a single NIC for the RDMA path in normal mode.

**Low-latency kernels:**

|  #EP  | DeepEP (IB) Dispatch | DeepEP (IB) Combine | DeepEP-EFA Dispatch+Combine |
|:-----:|:--------------------:|:-------------------:|:---------------------------:|
|   16  |       118 us         |       195 us        |          ~1145 us           |
|   32  |       155 us         |       273 us        |          ~1820 us           |
|   64  |       173 us         |       314 us        |          ~2584 us           |

The low-latency path has a significant gap compared to original DeepEP on InfiniBand. Key differences:

- **NVSHMEM IBGDA** allows the GPU to issue RDMA directly from kernel code with ~1 us initiation latency. DeepEP-EFA uses CPU-mediated RDMA (worker thread + libfabric), adding ~100-150 us of CPU-GPU coordination overhead per phase.
- **`cudaLaunchCooperativeKernel`** requires all SMs to be free before launch, adding ~60-80 us of synchronization overhead that NVSHMEM's single-kernel approach avoids.
- **GDRCopy flag polling** adds latency compared to NVSHMEM's native GPU-side signaling primitives.

These overheads are fundamental to the CPU-mediated architecture and represent the cost of running on EFA without GPU-native RDMA support.

## Quick start

### Requirements

- AWS P5en (or P5) instances with EFA networking
- NVIDIA H100 (SM 9.0) GPUs
- Docker and enroot for container-based execution
- CUDA 12.3+, PyTorch 2.1+, libfabric with EFA provider, GDRCopy

### Build

```bash
# Build Docker image
docker build --no-cache -t deepep-efa -f deepep-efa.Dockerfile .

# Create enroot squash file for Slurm
rm -f deepep-efa.sqsh
enroot import -o deepep-efa.sqsh dockerd://deepep-efa
```

### Run tests

Tests are launched via Slurm with enroot containers. Example sbatch files are provided:

```bash
# Intranode test (1 node, 8 GPUs)
sbatch deepep-efa-intranode.sbatch

# Internode test (2 nodes, 16 GPUs)
sbatch deepep-efa-internode.sbatch

# Low-latency test, 16 EP (2 nodes)
sbatch deepep-efa-lowlatency-16ep.sbatch

# Low-latency test, 32 EP (4 nodes)
sbatch deepep-efa-lowlatency-32ep.sbatch

# Low-latency test, 64 EP (8 nodes)
sbatch deepep-efa-lowlatency-64ep.sbatch
```

Inside the container, tests can also be run directly:

```bash
# Intranode
python tests/test_intranode.py

# Internode (normal kernels)
python tests/test_internode.py

# Low-latency (cooperative kernels)
DEEPEP_COOP=1 DEEPEP_PROFILE=1 python tests/test_low_latency.py \
    --num-tokens 128 --hidden 7168 --num-topk 8 --num-experts 288
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `DEEPEP_COOP` | Set to `1` to enable cooperative kernel path (required for multi-node low-latency) |
| `DEEPEP_PROFILE` | Set to `1` to enable per-phase timing breakdowns in low-latency tests |
| `DEEPEP_LL_PIPELINE` | Set to `0` to disable the non-coop LL pipeline path |
| `FI_PROVIDER` | Set to `efa` to use the EFA libfabric provider |
| `FI_EFA_USE_DEVICE_RDMA` | Set to `1` to enable GPU RDMA through EFA |
| `FI_EFA_FORK_SAFE` | Set to `1` for fork safety |

## Source structure

```
deepep_efa/
├── csrc/
│   ├── deep_ep.cpp              # Pybind11 bindings, host-side dispatch/combine
│   ├── deep_ep.hpp              # Buffer class, CoopScratch state
│   ├── config.hpp               # LowLatencyLayout, buffer sizing
│   ├── event.hpp                # EventOverlap (unchanged from DeepEP)
│   ├── kernels/
│   │   ├── intranode.cu         # Intranode NVLink kernels (unchanged)
│   │   ├── layout.cu            # Token layout computation (unchanged)
│   │   ├── internode_efa.cu     # Internode EFA kernels
│   │   ├── efa_kernels.cu       # EFA helper kernels
│   │   ├── coop_dispatch_send.cuh  # Cooperative dispatch send kernel
│   │   ├── coop_dispatch_recv.cuh  # Cooperative dispatch recv kernel
│   │   ├── coop_combine_send.cuh   # Cooperative combine send kernel
│   │   ├── coop_combine_recv.cuh   # Cooperative combine recv kernel
│   │   ├── coop_launch.cuh      # Cooperative kernel launch helpers
│   │   ├── coop_device_utils.cuh   # Device utility functions
│   │   └── ...                  # Shared utilities (utils.cuh, buffer.cuh, etc.)
│   └── transport/
│       ├── efa_transport.cpp/h  # libfabric EFA endpoint management, RDMA ops
│       ├── efa_worker.cpp/h     # EfaWorkerManager, RDMA setup and key exchange
│       ├── ll_pipeline.cpp/h    # LLPipelineWorker: worker thread, dispatch/combine
│       ├── gdr_signal.cpp/h     # GdrFlag, GdrVec, GdrCounter (GDRCopy wrappers)
│       ├── gdr_flags.cuh        # GPU-side MMIO load intrinsics
│       └── imm_counter.cpp/h    # ImmCounterMap: tag/rank/count CQ tracking
├── deep_ep/
│   ├── buffer.py                # Python Buffer class (wraps C++ runtime)
│   └── utils.py                 # Utility functions
├── tests/
│   ├── test_intranode.py        # Intranode correctness and benchmark
│   ├── test_internode.py        # Internode correctness and benchmark
│   ├── test_low_latency.py      # Low-latency correctness and benchmark
│   └── utils.py                 # Test helpers (init_dist, bench, bench_kineto)
├── setup.py                     # Build configuration
└── README.md                    # This file
```

## Key design decisions

1. **CPU-mediated RDMA instead of GPU-initiated RDMA.** EFA does not support NVSHMEM/IBGDA. All RDMA operations are issued by a dedicated CPU worker thread per GPU via libfabric, with GPU-CPU coordination through GDRCopy-mapped flags.

2. **Cooperative kernels for low-latency path.** The dispatch and combine operations are split into four cooperative kernels (send/recv for each). Cooperative launch ensures deterministic SM allocation and enables grid-wide synchronization between pack and unpack phases.

3. **NODE_SIZE=1 for cooperative path.** All traffic (including same-node) goes through EFA in the cooperative kernel path. This simplifies the implementation by eliminating NVLink/EFA routing decisions. Intranode-only workloads (EP8) fall back to the NVLink path.

4. **GDRCopy for CPU-to-GPU signaling.** The worker thread writes flags and metadata to GPU memory via GDRCopy MMIO (write-combining BAR mapping), providing sub-microsecond visibility to GPU kernels without CUDA API calls or stream synchronization.

5. **Tag-based completion tracking.** Each RDMA operation uses a unique tag encoded in the 32-bit immediate data. The `ImmCounterMap` tracks per-tag, per-rank completion counts, supporting overlapped operations across dispatch/combine phases.

## License

This code repository is released under [the MIT License](LICENSE).

## Acknowledgments

- [DeepEP](https://github.com/deepseek-ai/DeepEP) by DeepSeek for the original MoE communication library and kernel designs.
- [pplx-garden](https://github.com/pplx-ai/pplx-garden) by Perplexity for the EFA transport architecture (GDRCopy signaling, worker threads, cooperative kernels, multi-NIC sharding) that this implementation is based on.
