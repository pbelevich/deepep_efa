// efa_kernels.cu — Fused CUDA kernels for EFA internode dispatch/combine
//
// These kernels replace multi-step Python/torch operations with single fused
// GPU kernels, eliminating kernel launch overhead and intermediate allocations.
//
// Kernel 1: moe_routing_sort — fused nonzero + argsort + bincount
// Kernel 2: topk_remap — fused topk remapping for internode dispatch  
// Kernel 3: efa_permute — fused gather/scatter for pack/unpack in _efa_transfer

#ifdef ENABLE_EFA

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <unordered_map>

#include "configs.cuh"
#include "exception.cuh"
#include "../transport/gdr_flags.cuh"

namespace deep_ep {
namespace efa_kernels {

// ============================================================================
// Non-Coherent Memory Access Helpers (inspired by pplx-garden/memory.cuh)
//
// These PTX instructions bypass L1 cache and use L2 streaming hints,
// preventing cache pollution from large scatter/gather operations.
// ============================================================================

__forceinline__ __device__ uint2 ld_nc_uint2(const void* ptr) {
    uint2 v;
    asm volatile(
        "{ ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2]; }"
        : "=r"(v.x), "=r"(v.y)
        : "l"(ptr)
    );
    return v;
}

__forceinline__ __device__ void st_nc_uint2(void* ptr, uint2 v) {
    asm volatile(
        "{ st.global.L1::no_allocate.v2.u32 [%0], {%1, %2}; }"
        : : "l"(ptr), "r"(v.x), "r"(v.y)
    );
}

__forceinline__ __device__ uint32_t ld_nc_u32(const void* ptr) {
    uint32_t v;
    asm volatile(
        "{ ld.global.nc.L1::no_allocate.u32 %0, [%1]; }"
        : "=r"(v)
        : "l"(ptr)
    );
    return v;
}

__forceinline__ __device__ void st_nc_u32(void* ptr, uint32_t v) {
    asm volatile(
        "{ st.global.L1::no_allocate.u32 [%0], %1; }"
        : : "l"(ptr), "r"(v)
    );
}

// ============================================================================
// Iter 41: NC-load gather kernel for RDMA count exchange
// Reads num_ranks int32 values from scattered RDMA positions using non-cached loads.
// Avoids L2 cache pollution that causes stale reads on subsequent calls.
// src[i * stride + self_rank] -> dst[i], for i in [0, num_ranks)
// ============================================================================
__global__ void nc_gather_counts_kernel(const int32_t* __restrict__ src,
                                         int32_t* __restrict__ dst,
                                         int num_ranks,
                                         int stride_int32,
                                         int self_rank) {
    int i = threadIdx.x;
    if (i < num_ranks) {
        // NC load from RDMA buffer (bypasses L2 cache)
        uint32_t val = ld_nc_u32(src + i * stride_int32 + self_rank);
        dst[i] = static_cast<int32_t>(val);
    }
}

// ============================================================================
// Kernel: per_token_cast_to_fp8
//
// Fused per-token FP8 E4M3 quantization with group_size=128.
// Replaces 8-12 separate PyTorch kernel launches (abs, float, amax, clamp,
// div, exp2, ceil, log2, mul, cast) with a single fused kernel.
//
// Input:  x [M, N] bf16  (N must be divisible by 128)
// Output: x_fp8 [M, N] fp8_e4m3, scales [M, N/128] float32
// If round_scale: scales are rounded to nearest power of 2
// If use_ue8m0: scales are packed as UE8M0 (4 exponent bytes per int32)
//   output: x_fp8 [M, N] fp8_e4m3, packed_scales [M, N/128/4] int32
//
// Grid: (num_groups_per_row, M)  i.e. one block per group
// Block: 128 threads (one thread per element in the group)
// ============================================================================

__global__ void per_token_cast_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,     // [M, N]
    __nv_fp8_e4m3* __restrict__ x_fp8,       // [M, N]
    float* __restrict__ scales,              // [M, G] where G = N/128 (only if !use_ue8m0)
    int32_t* __restrict__ packed_scales,     // [M, G/4] (only if use_ue8m0)
    int M, int N,
    bool round_scale,
    bool use_ue8m0)
{
    const int group_idx = blockIdx.x;  // which group within the row
    const int row = blockIdx.y;        // which row (token)
    const int tid = threadIdx.x;       // element within the 128-element group
    const int groups_per_row = N / 128;

    if (row >= M || group_idx >= groups_per_row) return;

    // Load one element
    int global_idx = row * N + group_idx * 128 + tid;
    float val = __bfloat162float(x[global_idx]);
    float abs_val = fabsf(val);

    // Warp-level reduction for max (128 threads = 4 warps)
    // First reduce within each warp
    __shared__ float s_warp_max[4];

    float warp_max = abs_val;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, offset));
    }

    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (lane_id == 0) {
        s_warp_max[warp_id] = warp_max;
    }
    __syncthreads();

    // Thread 0 reduces across 4 warps
    float group_amax;
    if (tid == 0) {
        group_amax = fmaxf(fmaxf(s_warp_max[0], s_warp_max[1]),
                           fmaxf(s_warp_max[2], s_warp_max[3]));
        group_amax = fmaxf(group_amax, 1e-4f);  // clamp min
        s_warp_max[0] = group_amax;  // broadcast back
    }
    __syncthreads();
    group_amax = s_warp_max[0];

    // Compute scale and inv_scale
    float scale, inv_scale;
    if (round_scale) {
        scale = group_amax / 448.0f;
        // Round to nearest power of 2: exp2(ceil(log2(scale)))
        scale = exp2f(ceilf(log2f(scale)));
        inv_scale = 1.0f / scale;
    } else {
        scale = group_amax / 448.0f;
        inv_scale = 448.0f / group_amax;
    }

    // Quantize: multiply by inv_scale and cast to FP8 E4M3
    float scaled_val = val * inv_scale;
    // Clamp to FP8 E4M3 range [-448, 448]
    scaled_val = fminf(fmaxf(scaled_val, -448.0f), 448.0f);
    x_fp8[global_idx] = __nv_fp8_e4m3(scaled_val);

    // Write scale (one thread per group)
    if (tid == 0) {
        int scale_idx = row * groups_per_row + group_idx;
        if (use_ue8m0) {
            // Extract exponent byte from float32 scale
            uint32_t scale_bits = __float_as_uint(scale);
            uint8_t exponent = (scale_bits >> 23) & 0xFF;
            // Pack 4 exponents per int32 — only write if group_idx % 4 == 0
            // Use atomicOr to avoid race conditions between groups in same pack
            int pack_idx = row * (groups_per_row / 4) + group_idx / 4;
            int shift = (group_idx % 4) * 8;
            atomicOr(&packed_scales[pack_idx], static_cast<int32_t>(exponent) << shift);
        } else {
            scales[scale_idx] = scale;
        }
    }
}

// ============================================================================
// Fused combine weighted reduce + f32->bf16 conversion
//
// Same as ll_combine_weighted_reduce_kernel but writes bf16 output directly
// instead of writing f32 that needs a second pass to convert.
//
// NOTE: This cannot be fused into a single pass because multiple send entries
// can map to the same output token (atomicAdd contention). We still need the
// f32 accumulator, but we convert in-place after all atomics are done.
// This kernel replaces the separate f32_to_bf16_kernel pass.
// ============================================================================

// ============================================================================
// Kernel 1: moe_routing_sort
//
// Replaces: is_token_in_rank.nonzero() + argsort(dst_rank) + bincount
//
// Input:
//   is_token_in_rank: [num_tokens, num_ranks] bool
// Output:
//   sorted_token_ids: [total_send] int64 — token indices sorted by dst rank
//   send_counts: [num_ranks] int32 — number of tokens per destination rank
//   total_send: scalar int32 (written to total_send_out[0])
//
// Algorithm:
//   Phase 1: Count tokens per dst rank (parallel histogram)
//   Phase 2: Exclusive prefix sum over counts → write offsets
//   Phase 3: Write token IDs to sorted positions using atomics
// ============================================================================

// Phase 1+3 combined: count and scatter
__global__ void moe_routing_sort_kernel(
    const bool* __restrict__ is_token_in_rank,  // [num_tokens, num_ranks]
    int64_t* __restrict__ sorted_token_ids,      // [capacity] output
    int32_t* __restrict__ send_counts,            // [num_ranks] output
    int64_t* __restrict__ send_cumsum,            // [num_ranks + 1] output
    int32_t* __restrict__ total_send_out,         // [1] output
    int num_tokens,
    int num_ranks)
{
    // Phase 1: Global histogram — each thread processes a tile of tokens
    // We use global atomics on send_counts (which was zero-initialized by host)
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Phase 1: Count — each thread scans rows of is_token_in_rank
    for (int t = tid; t < num_tokens; t += stride) {
        for (int r = 0; r < num_ranks; r++) {
            if (is_token_in_rank[t * num_ranks + r]) {
                atomicAdd(&send_counts[r], 1);
            }
        }
    }
}

// Phase 2: Exclusive prefix sum + total
__global__ void prefix_sum_kernel(
    int32_t* __restrict__ send_counts,   // [num_ranks] input
    int64_t* __restrict__ send_cumsum,   // [num_ranks + 1] output
    int32_t* __restrict__ total_send_out,// [1] output
    int num_ranks)
{
    // Single thread — num_ranks is small (8-160)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t running = 0;
        send_cumsum[0] = 0;
        for (int r = 0; r < num_ranks; r++) {
            running += send_counts[r];
            send_cumsum[r + 1] = running;
        }
        total_send_out[0] = static_cast<int32_t>(running);
    }
}

// Iter 34: int32 -> int32 exclusive prefix sum (for recv_cumsum in ll_recv_unpack_v2)
// Replaces .to(kInt64).cumsum(0).to(kInt32).copy_() which launches 4+ kernels
__global__ void prefix_sum_i32_kernel(
    const int32_t* __restrict__ counts,  // [num_ranks] input
    int32_t* __restrict__ cumsum,        // [num_ranks + 1] output
    int num_ranks)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t running = 0;
        cumsum[0] = 0;
        for (int r = 0; r < num_ranks; r++) {
            running += counts[r];
            cumsum[r + 1] = running;
        }
    }
}

// Phase 3: Scatter token IDs to sorted positions — DETERMINISTIC ORDER
// Each block handles one rank and scans ALL tokens sequentially to guarantee
// ascending token order within each rank's segment.
// This ensures correctness: two calls with the same is_token_in_rank produce
// identical sorted_token_ids ordering (required by the test suite).
__global__ void scatter_token_ids_kernel(
    const bool* __restrict__ is_token_in_rank,  // [num_tokens, num_ranks]
    int64_t* __restrict__ sorted_token_ids,      // [total_send] output
    const int64_t* __restrict__ send_cumsum,     // [num_ranks + 1]
    int32_t* __restrict__ write_counters,         // [num_ranks] temp (unused, kept for API compat)
    int num_tokens,
    int num_ranks)
{
    // One block per rank
    const int rank_id = blockIdx.x;
    if (rank_id >= num_ranks) return;

    const int64_t write_base = send_cumsum[rank_id];
    int write_pos = 0;

    // Each thread in the block handles a strided subset of tokens
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory to collect (token_id, _) pairs from all threads in order
    // We do this in chunks to avoid needing shared memory proportional to num_tokens
    
    // Simple approach: single thread per rank (since num_ranks is small, 8-160)
    // Each block has 256 threads; use thread 0 for serial scan (fast enough for
    // num_tokens <= ~100K which is the typical range)
    if (tid == 0) {
        for (int t = 0; t < num_tokens; t++) {
            if (is_token_in_rank[t * num_ranks + rank_id]) {
                sorted_token_ids[write_base + write_pos] = static_cast<int64_t>(t);
                write_pos++;
            }
        }
    }
}

// ============================================================================
// Kernel 2: topk_remap
//
// Replaces: repeat_interleave + broadcasting comparison + torch.where
//
// For each entry in the sorted send list, remaps topk_idx from global expert
// space to local expert space of the destination rank.
//
// Input:
//   topk_idx: [num_tokens, num_topk] int64 — original global expert indices
//   topk_weights: [num_tokens, num_topk] float — original expert weights
//   sorted_token_ids: [total_send] int64 — token indices sorted by dst rank
//   send_cumsum: [num_ranks + 1] int64 — cumulative send counts
//   num_local_experts: int
// Output:
//   remapped_topk: [total_send, num_topk] int64
//   remapped_weights: [total_send, num_topk] float
// ============================================================================

__global__ void topk_remap_kernel(
    const int64_t* __restrict__ topk_idx,         // [num_tokens, num_topk]
    const float* __restrict__ topk_weights,        // [num_tokens, num_topk]
    const int64_t* __restrict__ sorted_token_ids,  // [total_send]
    const int64_t* __restrict__ send_cumsum,       // [num_ranks + 1]
    int64_t* __restrict__ remapped_topk,            // [total_send, num_topk]
    float* __restrict__ remapped_weights,           // [total_send, num_topk]
    int total_send,
    int num_ranks,
    int num_topk,
    int num_local_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_send * num_topk) return;

    const int send_idx = idx / num_topk;
    const int k = idx % num_topk;

    // Find which dst_rank this send_idx belongs to via binary search on send_cumsum
    int dst_rank = 0;
    {
        int lo = 0, hi = num_ranks;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (send_cumsum[mid + 1] <= send_idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        dst_rank = lo;
    }

    int64_t token_id = sorted_token_ids[send_idx];
    int64_t expert_id = topk_idx[token_id * num_topk + k];
    float weight = topk_weights[token_id * num_topk + k];

    int64_t local_start = static_cast<int64_t>(dst_rank) * num_local_experts;
    int64_t local_end = local_start + num_local_experts;

    int out_offset = send_idx * num_topk + k;
    if (expert_id >= local_start && expert_id < local_end && expert_id >= 0) {
        remapped_topk[out_offset] = expert_id - local_start;
        remapped_weights[out_offset] = weight;
    } else {
        remapped_topk[out_offset] = -1;
        remapped_weights[out_offset] = 0.0f;
    }
}

// ============================================================================
// Kernel 3: efa_permute
//
// Replaces: Python for-loop over ranks for pack/unpack in _efa_transfer.
// Performs a strided gather or scatter between a flat buffer and a packed
// buffer, selecting only ranks matching a mask (intra-node or inter-node).
//
// For pack mode (is_pack=true):
//   For each selected rank r, copies flat[offsets[r]:offsets[r]+counts[r]] → packed[pack_off:...]
// For unpack mode (is_pack=false):
//   For each selected rank r, copies packed[pack_off:...] → flat[offsets[r]:offsets[r]+counts[r]]
//
// Since the rank list is small (8 ranks), we use a simple approach:
// build the pack mapping on GPU and do a single vectorized copy.
// ============================================================================

__global__ void efa_permute_kernel(
    const uint8_t* __restrict__ src,       // source buffer (flat or packed)
    uint8_t* __restrict__ dst,             // destination buffer (packed or flat)
    const int64_t* __restrict__ src_offsets,// [num_selected_ranks] byte offsets into src
    const int64_t* __restrict__ dst_offsets,// [num_selected_ranks] byte offsets into dst
    const int64_t* __restrict__ copy_sizes, // [num_selected_ranks] bytes to copy per rank
    int num_selected_ranks,
    int64_t total_bytes)
{
    // Simple approach: each thread copies one byte at a time from the mapping
    // We iterate over selected ranks and copy their data
    // Since this is called for small rank counts (8), the overhead is minimal
    
    // Find which rank this byte belongs to via linear scan
    // (num_selected_ranks is small, typically 8)
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    
    if (gid >= total_bytes) return;
    
    // Binary search over dst_offsets to find rank
    int64_t acc = 0;
    for (int r = 0; r < num_selected_ranks; r++) {
        int64_t sz = copy_sizes[r];
        if (gid < acc + sz) {
            int64_t within = gid - acc;
            dst[dst_offsets[r] + within] = src[src_offsets[r] + within];
            return;
        }
        acc += sz;
    }
}

// ============================================================================
// Kernel 4: build_recv_src_meta
//
// Replaces: repeat_interleave + division + bitshift to build recv_src_meta
//
// Input:
//   recv_counts: [num_ranks] int32 — number of tokens received from each rank
// Output:
//   recv_src_meta: [num_recv_tokens, 8] uint8 — packed (src_rdma_rank, nvl_bits)
// ============================================================================

__global__ void build_recv_src_meta_kernel(
    const int32_t* __restrict__ recv_counts,  // [num_ranks]
    const int64_t* __restrict__ recv_cumsum,  // [num_ranks + 1]
    uint8_t* __restrict__ recv_src_meta,       // [num_recv_tokens, 8] output
    int num_recv_tokens,
    int num_ranks,
    int num_local_ranks)  // typically 8
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_recv_tokens) return;

    // Find source rank via binary search on recv_cumsum
    int src_rank = 0;
    {
        int lo = 0, hi = num_ranks;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (recv_cumsum[mid + 1] <= idx) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        src_rank = lo;
    }

    int src_rdma_rank = src_rank / num_local_ranks;
    int src_nvl_rank = src_rank % num_local_ranks;
    int nvl_bits = 1 << src_nvl_rank;

    // Pack as two int32s in 8 bytes
    int32_t* meta_int = reinterpret_cast<int32_t*>(recv_src_meta + idx * 8);
    meta_int[0] = src_rdma_rank;
    meta_int[1] = nvl_bits;
}

// ============================================================================
// Kernel 5: ll_dispatch_route_and_pack
//
// FUSED kernel for low-latency dispatch: replaces Python-side
// argsort + bincount + gather + slice-copy with a single 3-phase kernel.
//
// Input:
//   topk_idx:   [num_tokens, num_topk] int64 — global expert ids, -1 = invalid
//   x_data:     [num_tokens, hidden] uint8 — FP8 data (or BF16 reinterpreted)
//   x_scales:   [num_tokens, num_scale_cols] uint8 — FP8 scales (nullptr if !use_fp8)
//   num_tokens, num_topk, num_ranks, num_local_experts, hidden
//   data_bytes_per_token, scale_bytes_per_token, meta_bytes_per_token (=8), packed_bytes_per_token
//
// Output:
//   send_packed: [capacity * packed_bytes_per_token] uint8 — packed send buffer
//   send_counts: [num_ranks] int32
//   send_cumsum: [num_ranks + 1] int64
//   sorted_token_ids:  [capacity] int32 — token indices in send order
//   sorted_local_eids: [capacity] int32 — local expert ids in send order
//   total_send_out: [1] int32
//
// Algorithm:
//   Phase 1: Histogram — count valid (token,topk) pairs per dst_rank
//   Phase 2: Prefix sum (single-thread)
//   Phase 3: Scatter — deterministic order (one block per rank, serial scan)
//            + pack data/scales/meta in one pass
// ============================================================================

__global__ void ll_dispatch_count_kernel(
    const int64_t* __restrict__ topk_idx,  // [num_tokens * num_topk]
    int32_t* __restrict__ send_counts,      // [num_ranks] output (zero-initialized)
    int num_tokens,
    int num_topk,
    int num_local_experts)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total = num_tokens * num_topk;

    for (int i = tid; i < total; i += stride) {
        int64_t expert_id = topk_idx[i];
        if (expert_id >= 0) {
            int dst_rank = static_cast<int>(expert_id / num_local_experts);
            atomicAdd(&send_counts[dst_rank], 1);
        }
    }
}

// Iter 35: Fused count + prefix_sum kernel for dispatch
// All blocks count, last block (detected by atomic counter) does the prefix sum
__global__ void ll_dispatch_count_and_prefix_sum_kernel(
    const int64_t* __restrict__ topk_idx,
    int32_t* __restrict__ send_counts,
    int64_t* __restrict__ send_cumsum,
    int32_t* __restrict__ total_send_out,
    int32_t* __restrict__ block_done_counter,   // [1] — must be zeroed before launch
    int num_tokens,
    int num_topk,
    int num_local_experts,
    int num_ranks,
    int num_blocks)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total = num_tokens * num_topk;

    // Phase 1: Count
    for (int i = tid; i < total; i += stride) {
        int64_t expert_id = topk_idx[i];
        if (expert_id >= 0) {
            int dst_rank = static_cast<int>(expert_id / num_local_experts);
            atomicAdd(&send_counts[dst_rank], 1);
        }
    }

    // Ensure all atomicAdds are globally visible
    __threadfence();
    __syncthreads();

    // Phase 2: Last block does prefix sum
    if (threadIdx.x == 0) {
        int done = atomicAdd(block_done_counter, 1);
        if (done == num_blocks - 1) {
            int64_t running = 0;
            send_cumsum[0] = 0;
            for (int r = 0; r < num_ranks; r++) {
                running += send_counts[r];
                send_cumsum[r + 1] = running;
            }
            total_send_out[0] = static_cast<int32_t>(running);
        }
    }
}

// Phase 3: Scatter + pack — one block per rank, COOPERATIVE thread copying
// Thread 0 scans tokens for this rank and builds a list of (token_id, local_eid) pairs.
// Then ALL threads cooperate to copy data bytes for each matched token.
// This gives ~256x speedup on the data copy vs the old single-thread approach.
__global__ void ll_dispatch_scatter_and_pack_kernel(
    const int64_t* __restrict__ topk_idx,       // [num_tokens * num_topk]
    const uint8_t* __restrict__ x_data,          // [num_tokens, data_bytes_per_token]
    const uint8_t* __restrict__ x_scales,        // [num_tokens, scale_bytes_per_token] (or nullptr)
    uint8_t* __restrict__ send_packed,            // [capacity * packed_bytes_per_token] output
    int64_t* __restrict__ sorted_token_ids,       // [capacity] output (int64 for combine kernel compat)
    int32_t* __restrict__ sorted_local_eids,      // [capacity] output
    const int64_t* __restrict__ send_cumsum,      // [num_ranks + 1]
    int num_tokens,
    int num_topk,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token)
{
    const int rank_id = blockIdx.x;
    const int64_t write_base = send_cumsum[rank_id];
    const int total_entries = num_tokens * num_topk;

    // Shared memory for cooperative processing
    constexpr int BATCH_SIZE = 32;
    __shared__ int s_token_ids[BATCH_SIZE];
    __shared__ int s_local_eids[BATCH_SIZE];
    __shared__ int s_batch_count;
    __shared__ int s_scan_pos;        // shared so all threads see loop progress
    __shared__ int s_global_write_pos;

    if (threadIdx.x == 0) {
        s_scan_pos = 0;
        s_global_write_pos = 0;
        s_batch_count = 0;
    }
    __syncthreads();

    // Loop until thread 0 has scanned all entries
    while (true) {
        // Thread 0: fill batch from current scan position
        if (threadIdx.x == 0) {
            s_batch_count = 0;
            int pos = s_scan_pos;
            while (pos < total_entries && s_batch_count < BATCH_SIZE) {
                int64_t expert_id = topk_idx[pos];
                if (expert_id >= 0) {
                    int dst_rank = static_cast<int>(expert_id / num_local_experts);
                    if (dst_rank == rank_id) {
                        int t = pos / num_topk;
                        int local_eid = static_cast<int>(expert_id % num_local_experts);
                        s_token_ids[s_batch_count] = t;
                        s_local_eids[s_batch_count] = local_eid;
                        s_batch_count++;
                    }
                }
                pos++;
            }
            s_scan_pos = pos;  // update shared scan position
        }
        __syncthreads();

        int batch_count = s_batch_count;
        if (batch_count == 0) {
            // No more entries found — all threads exit together
            break;
        }

        int global_write_pos = s_global_write_pos;

        // All threads: cooperatively copy data for each entry in the batch
        for (int b = 0; b < batch_count; b++) {
            int t = s_token_ids[b];
            int local_eid = s_local_eids[b];
            int64_t out_idx = write_base + global_write_pos + b;

            // Thread 0: write metadata (small, only once per entry)
            if (threadIdx.x == 0) {
                sorted_token_ids[out_idx] = static_cast<int64_t>(t);
                sorted_local_eids[out_idx] = local_eid;
                // Pack meta: [token_idx(i32), local_expert_id(i32)]
                int meta_off = data_bytes_per_token + scale_bytes_per_token;
                int32_t* meta = reinterpret_cast<int32_t*>(send_packed + out_idx * packed_bytes_per_token + meta_off);
                meta[0] = t;
                meta[1] = local_eid;
            }

            // All threads: copy data bytes cooperatively (main bandwidth)
            uint8_t* dst_row = send_packed + out_idx * packed_bytes_per_token;
            const uint8_t* src_data = x_data + static_cast<int64_t>(t) * data_bytes_per_token;

            // Copy data portion — use 8-byte (uint2) copies for 2x throughput over uint32.
            // Safe because: data_bytes_per_token (e.g. 7168) is always 8-byte aligned,
            // src comes from PyTorch tensor (16-byte aligned), and packed_bytes_per_token
            // (e.g. 7400) is 8-byte aligned so dst_row is always 8-byte aligned.
            {
                const uint2* src8 = reinterpret_cast<const uint2*>(src_data);
                uint2* dst8 = reinterpret_cast<uint2*>(dst_row);
                int num_uint2 = data_bytes_per_token / 8;
                for (int i = threadIdx.x; i < num_uint2; i += blockDim.x) {
                    dst8[i] = src8[i];
                }
                // Handle 4-byte remainder (if data_bytes_per_token % 8 == 4)
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < data_bytes_per_token) {
                        // Copy remaining 4 bytes (can't have odd remainders since data is always >=4-byte aligned)
                        *reinterpret_cast<uint32_t*>(dst_row + rem_start) =
                            *reinterpret_cast<const uint32_t*>(src_data + rem_start);
                    }
                }
            }

            // Copy scales portion cooperatively — also use uint2 (8-byte) copies.
            // scale_bytes_per_token (e.g. 224) is 8-byte aligned, and offset
            // data_bytes_per_token (e.g. 7168) is 8-byte aligned.
            if (x_scales != nullptr && scale_bytes_per_token > 0) {
                int off = data_bytes_per_token;
                const uint8_t* src_scale = x_scales + static_cast<int64_t>(t) * scale_bytes_per_token;
                const uint2* src8 = reinterpret_cast<const uint2*>(src_scale);
                uint2* dst8 = reinterpret_cast<uint2*>(dst_row + off);
                int num_uint2 = scale_bytes_per_token / 8;
                for (int i = threadIdx.x; i < num_uint2; i += blockDim.x) {
                    dst8[i] = src8[i];
                }
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < scale_bytes_per_token) {
                        *reinterpret_cast<uint32_t*>(dst_row + off + rem_start) =
                            *reinterpret_cast<const uint32_t*>(src_scale + rem_start);
                    }
                }
            }
        }

        // Thread 0: update global write position
        if (threadIdx.x == 0) {
            s_global_write_pos += batch_count;
        }
        __syncthreads();
    }
}


// ============================================================================
// Iter 38: All-SM dispatch pack kernel
//
// Uses ALL SMs instead of just num_ranks blocks.
// Each block processes entries from topk_idx in grid-stride fashion.
// For each valid entry, atomicAdd to claim a write position, then all
// threads cooperatively copy data+scales+meta using NC uint2 loads/stores.
//
// Grid: (num_blocks, 1) where num_blocks = min(num_SMs, total_entries)
// Block: 256 threads
// ============================================================================
__global__ void ll_dispatch_pack_allsm_kernel(
    const int64_t* __restrict__ topk_idx,       // [num_tokens * num_topk]
    const uint8_t* __restrict__ x_data,          // [num_tokens, data_bytes_per_token]
    const uint8_t* __restrict__ x_scales,        // [num_tokens, scale_bytes_per_token] (or nullptr)
    uint8_t* __restrict__ send_packed,            // [capacity * packed_bytes_per_token] output
    int64_t* __restrict__ sorted_token_ids,       // [capacity] output
    int32_t* __restrict__ sorted_local_eids,      // [capacity] output
    const int64_t* __restrict__ send_cumsum,      // [num_ranks + 1]
    int32_t* __restrict__ rank_write_pos,         // [num_ranks] — atomic write counters, zero-initialized
    int total_entries,                            // num_tokens * num_topk
    int num_topk,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token)
{
    // Shared memory for batching: thread 0 finds valid entries, all threads copy data
    constexpr int BATCH_SIZE = 16;  // smaller batch for more SM occupancy
    __shared__ int s_token_ids[BATCH_SIZE];
    __shared__ int s_local_eids[BATCH_SIZE];
    __shared__ int64_t s_out_idx[BATCH_SIZE];
    __shared__ int s_batch_count;

    // Grid-stride loop over topk_idx entries
    for (int base = blockIdx.x * BATCH_SIZE; base < total_entries; base += gridDim.x * BATCH_SIZE) {
        // Thread 0: scan this chunk for valid entries destined for any rank
        if (threadIdx.x == 0) {
            s_batch_count = 0;
            int limit = min(base + BATCH_SIZE, total_entries);
            for (int i = base; i < limit; i++) {
                int64_t expert_id = topk_idx[i];
                if (expert_id >= 0) {
                    int dst_rank = static_cast<int>(expert_id / num_local_experts);
                    int t = i / num_topk;
                    int local_eid = static_cast<int>(expert_id % num_local_experts);
                    // Atomically claim a write position within this rank's output range
                    int pos = atomicAdd(&rank_write_pos[dst_rank], 1);
                    int64_t out_idx = send_cumsum[dst_rank] + pos;
                    int b = s_batch_count;
                    s_token_ids[b] = t;
                    s_local_eids[b] = local_eid;
                    s_out_idx[b] = out_idx;
                    s_batch_count++;
                }
            }
        }
        __syncthreads();

        int batch_count = s_batch_count;
        if (batch_count == 0) {
            __syncthreads();  // need barrier before next iteration modifies shared mem
            continue;
        }

        // All threads: cooperatively copy data for each entry in the batch
        for (int b = 0; b < batch_count; b++) {
            int t = s_token_ids[b];
            int local_eid = s_local_eids[b];
            int64_t out_idx = s_out_idx[b];

            // Thread 0: write metadata
            if (threadIdx.x == 0) {
                sorted_token_ids[out_idx] = static_cast<int64_t>(t);
                sorted_local_eids[out_idx] = local_eid;
                int meta_off = data_bytes_per_token + scale_bytes_per_token;
                int32_t* meta = reinterpret_cast<int32_t*>(send_packed + out_idx * packed_bytes_per_token + meta_off);
                meta[0] = t;
                meta[1] = local_eid;
            }

            // All threads: copy data bytes using NC uint2 loads/stores
            uint8_t* dst_row = send_packed + out_idx * packed_bytes_per_token;
            const uint8_t* src_data = x_data + static_cast<int64_t>(t) * data_bytes_per_token;

            {
                const uint2* src8 = reinterpret_cast<const uint2*>(src_data);
                uint2* dst8 = reinterpret_cast<uint2*>(dst_row);
                int num_uint2 = data_bytes_per_token / 8;
                for (int i = threadIdx.x; i < num_uint2; i += blockDim.x) {
                    uint2 val = ld_nc_uint2(&src8[i]);
                    st_nc_uint2(&dst8[i], val);
                }
                // Handle 4-byte remainder
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < data_bytes_per_token) {
                        st_nc_u32(dst_row + rem_start, ld_nc_u32(src_data + rem_start));
                    }
                }
            }

            // Copy scales using NC uint2
            if (x_scales != nullptr && scale_bytes_per_token > 0) {
                int off = data_bytes_per_token;
                const uint8_t* src_scale = x_scales + static_cast<int64_t>(t) * scale_bytes_per_token;
                const uint2* src8 = reinterpret_cast<const uint2*>(src_scale);
                uint2* dst8 = reinterpret_cast<uint2*>(dst_row + off);
                int num_uint2 = scale_bytes_per_token / 8;
                for (int i = threadIdx.x; i < num_uint2; i += blockDim.x) {
                    uint2 val = ld_nc_uint2(&src8[i]);
                    st_nc_uint2(&dst8[i], val);
                }
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < scale_bytes_per_token) {
                        st_nc_u32(dst_row + off + rem_start, ld_nc_u32(src_scale + rem_start));
                    }
                }
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// Kernel 6: ll_recv_unpack
//
// FUSED kernel for low-latency dispatch recv side: replaces Python-side
// argsort(sort_key) + bincount + scatter + layout_range construction.
//
// Input:
//   recv_packed: [total_recv * packed_bytes_per_token] uint8
//   recv_counts: [num_ranks] int32 — tokens received from each rank
//   total_recv, num_ranks, num_local_experts
//   data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token
//   hidden, N (= num_max_dispatch_tokens_per_rank * num_ranks)
//
// Phase 1 (histogram): Count tokens per (expert, src_rank) pair
// Phase 2 (prefix sum): Compute positions
// Phase 3 (scatter): Write data into packed_recv_x, build src_info + layout_range
//
// Output (all pre-allocated by host):
//   packed_recv_x:      [num_local_experts, N, hidden] — scattered data
//   packed_recv_scales: [num_local_experts, N, num_scale_cols] — scattered scales (or nullptr)
//   packed_recv_count:  [num_local_experts] int32
//   packed_recv_src_info: [num_local_experts, N] bf16
//   packed_recv_layout_range: [num_local_experts, num_ranks] int64
//   recv_expert_ids:     [total_recv] int32 — sorted expert ids (for combine)
//   recv_expert_pos:     [total_recv] int64 — position within expert (for combine)
//   sort_order_recv:     [total_recv] int64 — the sort permutation (for inverse_sort)
// ============================================================================

// Phase 1: Count (expert, src_rank) pairs
__global__ void ll_recv_count_kernel(
    const uint8_t* __restrict__ recv_packed,       // [total_recv * packed_bytes] or RDMA slot buffer
    const int32_t* __restrict__ recv_cumsum,        // [num_ranks + 1]
    int32_t* __restrict__ pair_counts,              // [num_local_experts * num_ranks] output
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t slot_size)   // >0: RDMA slot-based addressing, 0: contiguous flat
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < total_recv; i += stride) {
        // Find src_rank via binary search on recv_cumsum
        int src_rank = 0;
        {
            int lo = 0, hi = num_ranks;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (recv_cumsum[mid + 1] <= i) lo = mid + 1;
                else hi = mid;
            }
            src_rank = lo;
        }

        // Compute source address: slot-based or flat
        int64_t src_byte_offset;
        if (slot_size > 0) {
            int within_rank = i - recv_cumsum[src_rank];
            src_byte_offset = static_cast<int64_t>(src_rank) * slot_size +
                              static_cast<int64_t>(within_rank) * packed_bytes_per_token;
        } else {
            src_byte_offset = static_cast<int64_t>(i) * packed_bytes_per_token;
        }

        // Extract local_expert_id from metadata
        int meta_off = data_bytes_per_token + scale_bytes_per_token;
        const int32_t* meta = reinterpret_cast<const int32_t*>(
            recv_packed + src_byte_offset + meta_off);
        int local_eid = meta[1];

        atomicAdd(&pair_counts[local_eid * num_ranks + src_rank], 1);
    }
}

// Iter 35: Fused count + build_positions kernel
// Phase 1: All blocks count (expert, src_rank) pairs via atomicAdd
// Phase 2: Last block (detected by atomic grid-wide counter) does build_positions
// Eliminates one <<<1,1>>> kernel launch
__global__ void ll_recv_count_and_build_kernel(
    const uint8_t* __restrict__ recv_packed,
    const int32_t* __restrict__ recv_cumsum,
    int32_t* __restrict__ pair_counts,
    int32_t* __restrict__ packed_recv_count,
    int64_t* __restrict__ packed_recv_layout_range,
    int32_t* __restrict__ expert_cumsum,
    int32_t* __restrict__ pair_cumsum,
    int32_t* __restrict__ block_done_counter,     // [1] — must be zeroed before launch
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t slot_size,
    int num_blocks)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Phase 1: Count — same as ll_recv_count_kernel
    for (int i = tid; i < total_recv; i += stride) {
        int src_rank = 0;
        {
            int lo = 0, hi = num_ranks;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (recv_cumsum[mid + 1] <= i) lo = mid + 1;
                else hi = mid;
            }
            src_rank = lo;
        }

        int64_t src_byte_offset;
        if (slot_size > 0) {
            int within_rank = i - recv_cumsum[src_rank];
            src_byte_offset = static_cast<int64_t>(src_rank) * slot_size +
                              static_cast<int64_t>(within_rank) * packed_bytes_per_token;
        } else {
            src_byte_offset = static_cast<int64_t>(i) * packed_bytes_per_token;
        }

        int meta_off = data_bytes_per_token + scale_bytes_per_token;
        const int32_t* meta = reinterpret_cast<const int32_t*>(
            recv_packed + src_byte_offset + meta_off);
        int local_eid = meta[1];

        atomicAdd(&pair_counts[local_eid * num_ranks + src_rank], 1);
    }

    // Ensure all atomicAdds to pair_counts are globally visible
    __threadfence();
    __syncthreads();

    // Thread 0 of each block atomically increments the done counter
    if (threadIdx.x == 0) {
        int done = atomicAdd(block_done_counter, 1);
        // Last block: do the build_positions work (Phase 2)
        if (done == num_blocks - 1) {
            int global_pos = 0;
            expert_cumsum[0] = 0;

            for (int e = 0; e < num_local_experts; e++) {
                int expert_total = 0;
                int within_expert_pos = 0;
                for (int r = 0; r < num_ranks; r++) {
                    int cnt = pair_counts[e * num_ranks + r];
                    pair_cumsum[e * num_ranks + r] = within_expert_pos;

                    int64_t layout_val = 0;
                    if (cnt > 0) {
                        layout_val = (static_cast<int64_t>(within_expert_pos) << 32) | static_cast<int64_t>(cnt);
                    }
                    packed_recv_layout_range[e * num_ranks + r] = layout_val;

                    within_expert_pos += cnt;
                    expert_total += cnt;
                }
                packed_recv_count[e] = expert_total;
                global_pos += expert_total;
                expert_cumsum[e + 1] = global_pos;
            }
        }
    }
}

// Phase 2: Build expert counts, positions, layout_range — single-thread kernel
__global__ void ll_recv_build_positions_kernel(
    const int32_t* __restrict__ pair_counts,        // [num_local_experts * num_ranks]
    int32_t* __restrict__ packed_recv_count,         // [num_local_experts] output
    int64_t* __restrict__ packed_recv_layout_range,  // [num_local_experts, num_ranks] output
    int32_t* __restrict__ expert_cumsum,             // [num_local_experts + 1] output
    int32_t* __restrict__ pair_cumsum,               // [num_local_experts * num_ranks] output — within-expert position offset
    int num_local_experts,
    int num_ranks)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int global_pos = 0;
    expert_cumsum[0] = 0;

    for (int e = 0; e < num_local_experts; e++) {
        int expert_total = 0;
        int within_expert_pos = 0;
        for (int r = 0; r < num_ranks; r++) {
            int cnt = pair_counts[e * num_ranks + r];
            pair_cumsum[e * num_ranks + r] = within_expert_pos;

            // Layout range: upper 32 bits = begin_idx, lower 32 bits = count
            int64_t layout_val = 0;
            if (cnt > 0) {
                layout_val = (static_cast<int64_t>(within_expert_pos) << 32) | static_cast<int64_t>(cnt);
            }
            packed_recv_layout_range[e * num_ranks + r] = layout_val;

            within_expert_pos += cnt;
            expert_total += cnt;
        }
        packed_recv_count[e] = expert_total;
        global_pos += expert_total;
        expert_cumsum[e + 1] = global_pos;
    }
}

// Phase 3: Scatter data + scales + src_info, build sort_order + recv_expert_ids + recv_expert_pos
// One block per rank, COOPERATIVE thread copying (same pattern as ll_dispatch_scatter_and_pack_kernel)
// Thread 0 scans tokens and builds batches of (token_id, local_eid) pairs.
// Then ALL threads cooperate to copy data bytes for each token.
// This gives ~256x speedup on data copy vs the old single-thread approach.
__global__ void ll_recv_scatter_kernel(
    const uint8_t* __restrict__ recv_packed,            // [total_recv * packed_bytes] or RDMA slot buffer
    const int32_t* __restrict__ recv_cumsum,             // [num_ranks + 1]
    const int32_t* __restrict__ pair_cumsum,             // [num_local_experts * num_ranks]
    const int32_t* __restrict__ expert_cumsum,           // [num_local_experts + 1]
    uint8_t* __restrict__ packed_recv_x,                 // [num_local_experts, N, hidden_bytes]
    uint8_t* __restrict__ packed_recv_scales,            // [num_local_experts, N, scale_cols_bytes] (or nullptr)
    __nv_bfloat16* __restrict__ packed_recv_src_info,    // [num_local_experts, N]
    int32_t* __restrict__ recv_expert_ids_out,           // [total_recv] output
    int64_t* __restrict__ recv_expert_pos_out,           // [total_recv] output
    int64_t* __restrict__ sort_order_recv_out,           // [total_recv] output
    int32_t* __restrict__ pair_write_counters,           // [num_local_experts * num_ranks] temp (zero-init)
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t N,            // num_max_dispatch_tokens_per_rank * num_ranks
    int hidden_bytes,     // hidden * element_size for data
    int scale_cols_bytes, // num_scale_cols * element_size for scales
    int64_t slot_size)    // >0: RDMA slot-based addressing, 0: contiguous flat
{
    const int rank_id = blockIdx.x;
    if (rank_id >= num_ranks) return;

    const int recv_start = recv_cumsum[rank_id];
    const int recv_end = recv_cumsum[rank_id + 1];
    const int meta_off = data_bytes_per_token + scale_bytes_per_token;

    // Base address for this rank's data
    // slot_size > 0: data starts at rank_id * slot_size
    // slot_size == 0: data starts at recv_start * packed_bytes_per_token (contiguous)
    const int64_t rank_data_base = (slot_size > 0) ?
        static_cast<int64_t>(rank_id) * slot_size :
        static_cast<int64_t>(recv_start) * packed_bytes_per_token;

    // Shared memory: batch of entries found by thread 0
    constexpr int BATCH_SIZE = 32;
    __shared__ int s_recv_idx[BATCH_SIZE];       // original recv index
    __shared__ int s_local_eid[BATCH_SIZE];      // local expert id
    __shared__ int s_token_idx[BATCH_SIZE];      // original token index (for src_info)
    __shared__ int s_sorted_idx[BATCH_SIZE];     // sorted index (for output arrays)
    __shared__ int s_pos_in_expert[BATCH_SIZE];  // position within expert
    __shared__ int s_batch_count;
    __shared__ int s_scan_pos;                   // shared so all threads see loop progress

    if (threadIdx.x == 0) {
        s_batch_count = 0;
        s_scan_pos = recv_start;
    }
    __syncthreads();

    while (true) {
        // Thread 0: extract metadata, compute positions, fill batch
        if (threadIdx.x == 0) {
            s_batch_count = 0;
            int pos = s_scan_pos;
            int limit = min(pos + BATCH_SIZE, recv_end);
            for (int i = pos; i < limit; i++) {
                // Compute source byte offset for token i
                int64_t src_off = rank_data_base + static_cast<int64_t>(i - recv_start) * packed_bytes_per_token;
                const int32_t* meta = reinterpret_cast<const int32_t*>(
                    recv_packed + src_off + meta_off);
                int token_idx = meta[0];
                int local_eid = meta[1];

                // Compute position within expert
                int pair_idx = local_eid * num_ranks + rank_id;
                int pos_in_pair = atomicAdd(&pair_write_counters[pair_idx], 1);
                int pos_in_exp = pair_cumsum[pair_idx] + pos_in_pair;
                int sorted_idx = expert_cumsum[local_eid] + pos_in_exp;

                int b = s_batch_count;
                s_recv_idx[b] = i;
                s_local_eid[b] = local_eid;
                s_token_idx[b] = token_idx;
                s_sorted_idx[b] = sorted_idx;
                s_pos_in_expert[b] = pos_in_exp;
                s_batch_count++;
            }
            s_scan_pos = limit;
        }
        __syncthreads();

        int batch_count = s_batch_count;
        if (batch_count == 0) {
            // No more entries — all threads exit together
            break;
        }

        // All threads: cooperatively process each entry in the batch
        for (int b = 0; b < batch_count; b++) {
            int i = s_recv_idx[b];
            int local_eid = s_local_eid[b];
            int token_idx = s_token_idx[b];
            int sorted_idx = s_sorted_idx[b];
            int pos_in_exp = s_pos_in_expert[b];

            // Thread 0: write small metadata outputs
            if (threadIdx.x == 0) {
                sort_order_recv_out[sorted_idx] = static_cast<int64_t>(i);
                recv_expert_ids_out[sorted_idx] = local_eid;
                recv_expert_pos_out[sorted_idx] = static_cast<int64_t>(pos_in_exp);
                packed_recv_src_info[local_eid * N + pos_in_exp] = __float2bfloat16(static_cast<float>(token_idx));
            }

            // All threads: copy data bytes cooperatively (main bandwidth)
            // Use 8-byte (uint2) copies: packed_bytes_per_token (7400) is 8-byte aligned,
            // hidden_bytes (7168 for FP8, 14336 for BF16) is 8-byte aligned.
            {
                int64_t src_off = rank_data_base + static_cast<int64_t>(i - recv_start) * packed_bytes_per_token;
                const uint8_t* src = recv_packed + src_off;
                uint8_t* dst = packed_recv_x + (static_cast<int64_t>(local_eid) * N + pos_in_exp) * hidden_bytes;
                const uint2* src8 = reinterpret_cast<const uint2*>(src);
                uint2* dst8 = reinterpret_cast<uint2*>(dst);
                int num_uint2 = data_bytes_per_token / 8;
                for (int j = threadIdx.x; j < num_uint2; j += blockDim.x) {
                    dst8[j] = src8[j];
                }
                // Handle 4-byte remainder (if data_bytes_per_token % 8 == 4)
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < data_bytes_per_token) {
                        *reinterpret_cast<uint32_t*>(dst + rem_start) =
                            *reinterpret_cast<const uint32_t*>(src + rem_start);
                    }
                }
            }

            // All threads: copy scales cooperatively — use uint2 (8-byte) copies.
            // scale_bytes_per_token (224) and data_bytes_per_token (7168) are both 8-byte aligned,
            // scale_cols_bytes (224) is 8-byte aligned.
            if (packed_recv_scales != nullptr && scale_bytes_per_token > 0) {
                int64_t src_off = rank_data_base + static_cast<int64_t>(i - recv_start) * packed_bytes_per_token + data_bytes_per_token;
                const uint8_t* src_s = recv_packed + src_off;
                uint8_t* dst_s = packed_recv_scales + (static_cast<int64_t>(local_eid) * N + pos_in_exp) * scale_cols_bytes;
                const uint2* src8 = reinterpret_cast<const uint2*>(src_s);
                uint2* dst8 = reinterpret_cast<uint2*>(dst_s);
                int num_uint2 = scale_bytes_per_token / 8;
                for (int j = threadIdx.x; j < num_uint2; j += blockDim.x) {
                    dst8[j] = src8[j];
                }
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < scale_bytes_per_token) {
                        *reinterpret_cast<uint32_t*>(dst_s + rem_start) =
                            *reinterpret_cast<const uint32_t*>(src_s + rem_start);
                    }
                }
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// Iter 38: All-SM recv scatter kernel
//
// Uses ALL SMs instead of just num_ranks blocks.
// Each block processes recv tokens in grid-stride fashion.
// For each token, extracts metadata, atomicAdd to get write position,
// then all threads cooperatively copy data+scales using NC uint2.
//
// Grid: (num_blocks, 1) where num_blocks = min(num_SMs, total_recv)
// Block: 256 threads
// ============================================================================
__global__ void ll_recv_scatter_allsm_kernel(
    const uint8_t* __restrict__ recv_packed,            // recv buffer (RDMA slots or contiguous)
    const int32_t* __restrict__ recv_cumsum,             // [num_ranks + 1]
    const int32_t* __restrict__ pair_cumsum,             // [num_local_experts * num_ranks]
    const int32_t* __restrict__ expert_cumsum,           // [num_local_experts + 1]
    uint8_t* __restrict__ packed_recv_x,                 // [num_local_experts, N, hidden_bytes]
    uint8_t* __restrict__ packed_recv_scales,            // [num_local_experts, N, scale_cols_bytes] (or nullptr)
    __nv_bfloat16* __restrict__ packed_recv_src_info,    // [num_local_experts, N]
    int32_t* __restrict__ recv_expert_ids_out,           // [total_recv] output
    int64_t* __restrict__ recv_expert_pos_out,           // [total_recv] output
    int64_t* __restrict__ sort_order_recv_out,           // [total_recv] output
    int32_t* __restrict__ pair_write_counters,           // [num_local_experts * num_ranks] temp (zero-init)
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t N,
    int hidden_bytes,
    int scale_cols_bytes,
    int64_t slot_size)
{
    const int meta_off = data_bytes_per_token + scale_bytes_per_token;

    // Shared memory for batching
    constexpr int BATCH_SIZE = 16;
    __shared__ int s_src_rank[BATCH_SIZE];
    __shared__ int s_local_eid[BATCH_SIZE];
    __shared__ int s_token_idx[BATCH_SIZE];
    __shared__ int s_sorted_idx[BATCH_SIZE];
    __shared__ int s_pos_in_expert[BATCH_SIZE];
    __shared__ int s_recv_idx[BATCH_SIZE];
    __shared__ int s_batch_count;

    // Grid-stride loop over recv tokens
    for (int base = blockIdx.x * BATCH_SIZE; base < total_recv; base += gridDim.x * BATCH_SIZE) {
        // Thread 0: process metadata for this chunk
        if (threadIdx.x == 0) {
            s_batch_count = 0;
            int limit = min(base + BATCH_SIZE, total_recv);
            for (int i = base; i < limit; i++) {
                // Binary search for src_rank
                int src_rank = 0;
                {
                    int lo = 0, hi = num_ranks;
                    while (lo < hi) {
                        int mid = (lo + hi) / 2;
                        if (recv_cumsum[mid + 1] <= i) lo = mid + 1;
                        else hi = mid;
                    }
                    src_rank = lo;
                }

                // Compute source byte offset (slot-based or flat)
                int64_t src_byte_offset;
                if (slot_size > 0) {
                    int within_rank = i - recv_cumsum[src_rank];
                    src_byte_offset = static_cast<int64_t>(src_rank) * slot_size +
                                      static_cast<int64_t>(within_rank) * packed_bytes_per_token;
                } else {
                    src_byte_offset = static_cast<int64_t>(i) * packed_bytes_per_token;
                }

                // Extract metadata
                const int32_t* meta = reinterpret_cast<const int32_t*>(
                    recv_packed + src_byte_offset + meta_off);
                int token_idx = meta[0];
                int local_eid = meta[1];

                // Claim position within expert
                int pair_idx = local_eid * num_ranks + src_rank;
                int pos_in_pair = atomicAdd(&pair_write_counters[pair_idx], 1);
                int pos_in_exp = pair_cumsum[pair_idx] + pos_in_pair;
                int sorted_idx = expert_cumsum[local_eid] + pos_in_exp;

                int b = s_batch_count;
                s_src_rank[b] = src_rank;
                s_local_eid[b] = local_eid;
                s_token_idx[b] = token_idx;
                s_sorted_idx[b] = sorted_idx;
                s_pos_in_expert[b] = pos_in_exp;
                s_recv_idx[b] = i;
                s_batch_count++;
            }
        }
        __syncthreads();

        int batch_count = s_batch_count;
        if (batch_count == 0) {
            __syncthreads();
            continue;
        }

        // All threads: cooperatively process each entry
        for (int b = 0; b < batch_count; b++) {
            int i = s_recv_idx[b];
            int src_rank = s_src_rank[b];
            int local_eid = s_local_eid[b];
            int token_idx = s_token_idx[b];
            int sorted_idx = s_sorted_idx[b];
            int pos_in_exp = s_pos_in_expert[b];

            // Thread 0: write small metadata outputs
            if (threadIdx.x == 0) {
                sort_order_recv_out[sorted_idx] = static_cast<int64_t>(i);
                recv_expert_ids_out[sorted_idx] = local_eid;
                recv_expert_pos_out[sorted_idx] = static_cast<int64_t>(pos_in_exp);
                packed_recv_src_info[local_eid * N + pos_in_exp] = __float2bfloat16(static_cast<float>(token_idx));
            }

            // Compute source address
            int64_t src_off;
            if (slot_size > 0) {
                int within_rank = i - recv_cumsum[src_rank];
                src_off = static_cast<int64_t>(src_rank) * slot_size +
                          static_cast<int64_t>(within_rank) * packed_bytes_per_token;
            } else {
                src_off = static_cast<int64_t>(i) * packed_bytes_per_token;
            }

            // All threads: copy data bytes using NC uint2
            {
                const uint8_t* src = recv_packed + src_off;
                uint8_t* dst = packed_recv_x + (static_cast<int64_t>(local_eid) * N + pos_in_exp) * hidden_bytes;
                const uint2* src8 = reinterpret_cast<const uint2*>(src);
                uint2* dst8 = reinterpret_cast<uint2*>(dst);
                int num_uint2 = data_bytes_per_token / 8;
                for (int j = threadIdx.x; j < num_uint2; j += blockDim.x) {
                    uint2 val = ld_nc_uint2(&src8[j]);
                    st_nc_uint2(&dst8[j], val);
                }
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < data_bytes_per_token) {
                        st_nc_u32(dst + rem_start, ld_nc_u32(src + rem_start));
                    }
                }
            }

            // All threads: copy scales using NC uint2
            if (packed_recv_scales != nullptr && scale_bytes_per_token > 0) {
                int64_t src_off2 = src_off + data_bytes_per_token;
                const uint8_t* src_s = recv_packed + src_off2;
                uint8_t* dst_s = packed_recv_scales + (static_cast<int64_t>(local_eid) * N + pos_in_exp) * scale_cols_bytes;
                const uint2* src8 = reinterpret_cast<const uint2*>(src_s);
                uint2* dst8 = reinterpret_cast<uint2*>(dst_s);
                int num_uint2 = scale_bytes_per_token / 8;
                for (int j = threadIdx.x; j < num_uint2; j += blockDim.x) {
                    uint2 val = ld_nc_uint2(&src8[j]);
                    st_nc_uint2(&dst8[j], val);
                }
                if (threadIdx.x == 0) {
                    int rem_start = num_uint2 * 8;
                    if (rem_start < scale_bytes_per_token) {
                        st_nc_u32(dst_s + rem_start, ld_nc_u32(src_s + rem_start));
                    }
                }
            }
        }
        __syncthreads();
    }
}


// ============================================================================
// Kernel 7: ll_combine_weighted_reduce
//
// FUSED kernel for low-latency combine step 3: replaces Python-side
// gather + unsort + topk match + multiply + index_add
//
// Input:
//   combine_recv_flat: [total_send, hidden] bf16 — received expert outputs
//   sorted_send_token_ids: [total_send] int64
//   global_expert_ids: [total_send] int64
//   topk_idx: [num_tokens, num_topk] int64
//   topk_weights: [num_tokens, num_topk] float32
//   total_send, num_tokens, hidden, num_topk
//
// Output:
//   combined_x: [num_tokens, hidden] bf16 — weighted sum output
//
// Each thread handles one (send_entry, hidden_dim_chunk) pair.
// We use atomicAdd in f32 for the reduction across tokens.
// ============================================================================

__global__ void ll_combine_weighted_reduce_kernel(
    const __nv_bfloat16* __restrict__ combine_recv_flat, // [total_send, hidden] or RDMA slot buffer
    const int64_t* __restrict__ sorted_send_token_ids,    // [total_send]
    const int64_t* __restrict__ global_expert_ids,        // [total_send]
    const int64_t* __restrict__ topk_idx,                 // [num_tokens, num_topk]
    const float* __restrict__ topk_weights,               // [num_tokens, num_topk]
    float* __restrict__ combined_x_f32,                   // [num_tokens, hidden] output (f32 accumulator)
    const int32_t* __restrict__ send_cumsum,              // [num_ranks + 1] — for slot addressing (nullptr if flat)
    int total_send,
    int hidden,
    int num_topk,
    int num_ranks,
    int64_t slot_size)   // >0: RDMA slot-based (bytes), 0: contiguous flat
{
    // Each thread processes one element: (send_idx, h)
    // Grid: (total_send, ceil(hidden/blockDim.x))
    const int send_idx = blockIdx.x;
    if (send_idx >= total_send) return;

    // Find weight for this send entry
    int64_t token_id = sorted_send_token_ids[send_idx];
    int64_t expert_id = global_expert_ids[send_idx];

    float weight = 0.0f;
    for (int k = 0; k < num_topk; k++) {
        if (topk_idx[token_id * num_topk + k] == expert_id) {
            weight += topk_weights[token_id * num_topk + k];
        }
    }

    if (weight == 0.0f) return;

    // Compute source row address
    const __nv_bfloat16* src_row;
    if (slot_size > 0 && send_cumsum != nullptr) {
        // Find which rank this send_idx belongs to via binary search on send_cumsum
        int src_rank = 0;
        {
            int lo = 0, hi = num_ranks;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (send_cumsum[mid + 1] <= send_idx) lo = mid + 1;
                else hi = mid;
            }
            src_rank = lo;
        }
        int within_rank = send_idx - send_cumsum[src_rank];
        // slot_size is in bytes, hidden is in bf16 elements (2 bytes each)
        int64_t byte_offset = static_cast<int64_t>(src_rank) * slot_size +
                              static_cast<int64_t>(within_rank) * hidden * 2;
        src_row = reinterpret_cast<const __nv_bfloat16*>(
            reinterpret_cast<const uint8_t*>(combine_recv_flat) + byte_offset);
    } else {
        src_row = combine_recv_flat + static_cast<int64_t>(send_idx) * hidden;
    }
    float* dst_row = combined_x_f32 + static_cast<int64_t>(token_id) * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        float val = __bfloat162float(src_row[h]) * weight;
        atomicAdd(&dst_row[h], val);
    }
}

// Simple kernel to convert f32 to bf16
__global__ void f32_to_bf16_kernel(
    const float* __restrict__ src,       // [N, hidden]
    __nv_bfloat16* __restrict__ dst,     // [N, hidden]
    int64_t total_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}


// ============================================================================
// Kernel 8: ll_compute_inverse_sort
//
// Computes inverse_sort[sort_order[i]] = i for i in [0, total)
// One thread per element.
// ============================================================================
__global__ void ll_compute_inverse_sort_kernel(
    const int64_t* __restrict__ sort_order,   // [total]
    int64_t* __restrict__ inverse_sort,        // [total] output
    int total)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        inverse_sort[sort_order[idx]] = static_cast<int64_t>(idx);
    }
}

// ============================================================================
// Kernel 9: ll_compute_global_expert_ids
//
// Computes dst_rank_per_sent and global_expert_ids from send_counts + sorted_local_eids.
// dst_rank_per_sent[i] = rank that sent token i (via prefix sum over send_counts)
// global_expert_ids[i] = dst_rank_per_sent[i] * num_local_experts + sorted_local_eids[i]
//
// One thread per total_send entry.
// ============================================================================
__global__ void ll_compute_global_expert_ids_kernel(
    const int64_t* __restrict__ send_cumsum,        // [num_ranks + 1] int64
    const int32_t* __restrict__ sorted_local_eids,  // [total_send]
    int64_t* __restrict__ global_expert_ids,         // [total_send] output
    int total_send,
    int num_ranks,
    int num_local_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_send) {
        // Binary search to find rank
        int rank = 0;
        {
            int lo = 0, hi = num_ranks;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (send_cumsum[mid + 1] <= static_cast<int64_t>(idx)) lo = mid + 1;
                else hi = mid;
            }
            rank = lo;
        }
        int local_eid = sorted_local_eids[idx];
        global_expert_ids[idx] = static_cast<int64_t>(rank) * num_local_experts + static_cast<int64_t>(local_eid);
    }
}

}  // namespace efa_kernels

// Iter 36: Static per-device block counter for fused kernels
// Avoids per-call torch::zeros({1}) allocation overhead
// Each GPU gets its own counter. Safe because dispatch and unpack are sequential
// on the same device, and the counter is zeroed via cudaMemsetAsync before each use.
namespace {
    static std::unordered_map<int, int32_t*> s_block_counters;

    int32_t* get_block_counter(int device_id) {
        auto it = s_block_counters.find(device_id);
        if (it != s_block_counters.end()) {
            return it->second;
        }
        int32_t* ptr = nullptr;
        int prev_device;
        cudaGetDevice(&prev_device);
        if (prev_device != device_id) cudaSetDevice(device_id);
        cudaMalloc(&ptr, sizeof(int32_t));
        if (prev_device != device_id) cudaSetDevice(prev_device);
        s_block_counters[device_id] = ptr;
        return ptr;
    }

    // Iter 38: Static per-device rank_write_pos buffer for all-SM pack kernel
    // Max 64 ranks, zeroed via cudaMemsetAsync before each use
    static constexpr int MAX_RANKS = 64;
    static std::unordered_map<int, int32_t*> s_rank_write_pos;

    int32_t* get_rank_write_pos(int device_id) {
        auto it = s_rank_write_pos.find(device_id);
        if (it != s_rank_write_pos.end()) {
            return it->second;
        }
        int32_t* ptr = nullptr;
        int prev_device;
        cudaGetDevice(&prev_device);
        if (prev_device != device_id) cudaSetDevice(device_id);
        cudaMalloc(&ptr, MAX_RANKS * sizeof(int32_t));
        if (prev_device != device_id) cudaSetDevice(prev_device);
        s_rank_write_pos[device_id] = ptr;
        return ptr;
    }
}  // anonymous namespace

// ============================================================================
// Host-callable wrappers (exposed via pybind11 as free functions)
// ============================================================================

// moe_routing_sort: fused nonzero + argsort + bincount
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>
moe_routing_sort(const torch::Tensor& is_token_in_rank)
{
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2);
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.is_cuda());

    const int num_tokens = is_token_in_rank.size(0);
    const int num_ranks = is_token_in_rank.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Allocate outputs
    // Upper bound for total_send: num_tokens * num_ranks (worst case: every token goes to every rank)
    // But typically total_send << num_tokens * num_ranks
    // We'll allocate at max capacity and truncate later
    int max_send = num_tokens * num_ranks;

    auto send_counts = torch::zeros({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(is_token_in_rank.device()));
    auto send_cumsum = torch::zeros({num_ranks + 1}, torch::TensorOptions().dtype(torch::kInt64).device(is_token_in_rank.device()));
    auto total_send_out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(is_token_in_rank.device()));
    auto write_counters = torch::zeros({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(is_token_in_rank.device()));

    // Allocate sorted_token_ids at max capacity
    auto sorted_token_ids = torch::empty({max_send}, torch::TensorOptions().dtype(torch::kInt64).device(is_token_in_rank.device()));

    const int threads = 256;
    const int blocks = std::min((num_tokens + threads - 1) / threads, 1024);

    // Phase 1: Count
    efa_kernels::moe_routing_sort_kernel<<<blocks, threads, 0, stream>>>(
        is_token_in_rank.data_ptr<bool>(),
        sorted_token_ids.data_ptr<int64_t>(),
        send_counts.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        total_send_out.data_ptr<int32_t>(),
        num_tokens, num_ranks);

    // Phase 2: Prefix sum (single thread)
    efa_kernels::prefix_sum_kernel<<<1, 1, 0, stream>>>(
        send_counts.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        total_send_out.data_ptr<int32_t>(),
        num_ranks);

    // Phase 3: Scatter — one block per rank, deterministic token ordering
    efa_kernels::scatter_token_ids_kernel<<<num_ranks, threads, 0, stream>>>(
        is_token_in_rank.data_ptr<bool>(),
        sorted_token_ids.data_ptr<int64_t>(),
        send_cumsum.data_ptr<int64_t>(),
        write_counters.data_ptr<int32_t>(),
        num_tokens, num_ranks);

    // Sync to get total_send
    int total_send = total_send_out.item<int32_t>();

    // Truncate sorted_token_ids to actual size
    sorted_token_ids = sorted_token_ids.slice(0, 0, total_send);

    return std::make_tuple(sorted_token_ids, send_counts, send_cumsum, total_send);
}

// topk_remap: fused topk remapping
std::tuple<torch::Tensor, torch::Tensor>
topk_remap(const torch::Tensor& topk_idx,
           const torch::Tensor& topk_weights,
           const torch::Tensor& sorted_token_ids,
           const torch::Tensor& send_cumsum,
           int total_send,
           int num_ranks,
           int num_local_experts)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous() && topk_idx.is_cuda());
    EP_HOST_ASSERT(topk_weights.dim() == 2);
    EP_HOST_ASSERT(topk_weights.is_contiguous() && topk_weights.is_cuda());

    const int num_topk = topk_idx.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto remapped_topk = torch::empty({total_send, num_topk}, topk_idx.options());
    auto remapped_weights = torch::empty({total_send, num_topk}, torch::TensorOptions().dtype(torch::kFloat32).device(topk_idx.device()));

    if (total_send == 0) {
        return std::make_tuple(remapped_topk, remapped_weights);
    }

    const int total_work = total_send * num_topk;
    const int threads = 256;
    const int blocks = (total_work + threads - 1) / threads;

    efa_kernels::topk_remap_kernel<<<blocks, threads, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        topk_weights.data_ptr<float>(),
        sorted_token_ids.data_ptr<int64_t>(),
        send_cumsum.data_ptr<int64_t>(),
        remapped_topk.data_ptr<int64_t>(),
        remapped_weights.data_ptr<float>(),
        total_send, num_ranks, num_topk, num_local_experts);

    return std::make_tuple(remapped_topk, remapped_weights);
}

// efa_permute: fused gather/scatter for pack/unpack
void efa_permute(const torch::Tensor& src,
                 torch::Tensor& dst,
                 const torch::Tensor& src_offsets,
                 const torch::Tensor& dst_offsets,
                 const torch::Tensor& copy_sizes,
                 int64_t total_bytes)
{
    if (total_bytes == 0) return;

    EP_HOST_ASSERT(src.is_contiguous() && src.is_cuda());
    EP_HOST_ASSERT(dst.is_contiguous() && dst.is_cuda());
    EP_HOST_ASSERT(src_offsets.is_contiguous() && src_offsets.is_cuda());
    EP_HOST_ASSERT(dst_offsets.is_contiguous() && dst_offsets.is_cuda());
    EP_HOST_ASSERT(copy_sizes.is_contiguous() && copy_sizes.is_cuda());

    int num_selected = src_offsets.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    const int threads = 256;
    const int blocks = std::min(static_cast<int64_t>((total_bytes + threads - 1) / threads), static_cast<int64_t>(65535));

    efa_kernels::efa_permute_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<uint8_t>(),
        dst.data_ptr<uint8_t>(),
        src_offsets.data_ptr<int64_t>(),
        dst_offsets.data_ptr<int64_t>(),
        copy_sizes.data_ptr<int64_t>(),
        num_selected,
        total_bytes);
}

// build_recv_src_meta: fused metadata construction
torch::Tensor build_recv_src_meta(
    const torch::Tensor& recv_counts,
    const torch::Tensor& recv_cumsum,
    int num_recv_tokens,
    int num_ranks,
    int num_local_ranks)
{
    auto meta = torch::zeros({num_recv_tokens, 8}, torch::TensorOptions().dtype(torch::kUInt8).device(recv_counts.device()));

    if (num_recv_tokens == 0) return meta;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    const int blocks = (num_recv_tokens + threads - 1) / threads;

    efa_kernels::build_recv_src_meta_kernel<<<blocks, threads, 0, stream>>>(
        recv_counts.data_ptr<int32_t>(),
        recv_cumsum.data_ptr<int64_t>(),
        meta.data_ptr<uint8_t>(),
        num_recv_tokens, num_ranks, num_local_ranks);

    return meta;
}

// ll_dispatch_route_and_pack: fused routing + packing for LL dispatch
// Returns: (send_packed, send_counts, sorted_token_ids, sorted_local_eids, total_send)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
ll_dispatch_route_and_pack(
    const torch::Tensor& topk_idx,       // [num_tokens, num_topk] int64
    const torch::Tensor& x_data,         // [num_tokens, hidden] (viewed as uint8, shape [num_tokens, data_bytes_per_token])
    const c10::optional<torch::Tensor>& x_scales_opt,  // [num_tokens, num_scale_cols] (viewed as uint8)
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous() && topk_idx.is_cuda());
    EP_HOST_ASSERT(x_data.dim() == 2 && x_data.is_contiguous() && x_data.is_cuda());

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = topk_idx.device();

    // Max possible sends: each token can be sent to up to num_topk experts
    int max_send = num_tokens * num_topk;

    auto send_counts = torch::zeros({num_ranks}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto send_cumsum = torch::zeros({num_ranks + 1}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto total_send_out = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Phase 1: Count
    {
        const int threads = 256;
        const int blocks = std::min((num_tokens * num_topk + threads - 1) / threads, 1024);
        efa_kernels::ll_dispatch_count_kernel<<<blocks, threads, 0, stream>>>(
            topk_idx.data_ptr<int64_t>(),
            send_counts.data_ptr<int32_t>(),
            num_tokens, num_topk, num_local_experts);
    }

    // Phase 2: Prefix sum
    efa_kernels::prefix_sum_kernel<<<1, 1, 0, stream>>>(
        send_counts.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        total_send_out.data_ptr<int32_t>(),
        num_ranks);

    // Sync to get total_send (needed for allocation)
    int total_send = total_send_out.item<int32_t>();

    if (total_send == 0) {
        auto send_packed = torch::empty({0}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
        auto sorted_token_ids = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto sorted_local_eids = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        return std::make_tuple(send_packed, send_counts, sorted_token_ids, sorted_local_eids, 0);
    }

    auto send_packed = torch::empty({static_cast<int64_t>(total_send) * packed_bytes_per_token},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto sorted_token_ids = torch::empty({total_send}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto sorted_local_eids = torch::empty({total_send}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // Phase 3: Scatter + pack — Iter 38: Use all-SM kernel with NC memory access
    const uint8_t* x_scales_ptr = x_scales_opt.has_value() ? x_scales_opt->data_ptr<uint8_t>() : nullptr;
    int32_t* rank_write_pos = get_rank_write_pos(device.index());
    cudaMemsetAsync(rank_write_pos, 0, num_ranks * sizeof(int32_t), stream);

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device.index());
    const int total_entries = num_tokens * num_topk;
    const int blocks = std::min(num_sms, (total_entries + 15) / 16);

    efa_kernels::ll_dispatch_pack_allsm_kernel<<<blocks, 256, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        x_data.data_ptr<uint8_t>(),
        x_scales_ptr,
        send_packed.data_ptr<uint8_t>(),
        sorted_token_ids.data_ptr<int64_t>(),
        sorted_local_eids.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        rank_write_pos,
        total_entries, num_topk, num_local_experts,
        data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token);

    return std::make_tuple(send_packed, send_counts, sorted_token_ids, sorted_local_eids, total_send);
}

// ll_recv_unpack: fused unpack + scatter for LL dispatch receive side
// Returns: (packed_recv_count, packed_recv_src_info, packed_recv_layout_range,
//           recv_expert_ids, recv_expert_pos, sort_order_recv)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ll_recv_unpack(
    const torch::Tensor& recv_packed,           // [total_recv * packed_bytes_per_token] uint8
    const torch::Tensor& recv_counts_tensor,    // [num_ranks] int32
    torch::Tensor& packed_recv_x,               // [num_local_experts, N, hidden] — pre-allocated, zero-init
    c10::optional<torch::Tensor> packed_recv_scales_opt,  // [num_local_experts, N, scale_cols] or nullopt
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t N,
    int hidden_bytes,
    int scale_cols_bytes)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = recv_counts_tensor.device();

    // Build recv_cumsum on GPU
    auto recv_cumsum = torch::zeros({num_ranks + 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    {
        // Simple prefix sum kernel (reuse existing pattern but for int32)
        // We can compute this on CPU since recv_counts_tensor is small
        // Actually, let's build it with a tiny kernel
        auto recv_counts_int64 = recv_counts_tensor.to(torch::kInt64);
        auto cumsum = recv_counts_int64.cumsum(0);
        recv_cumsum.slice(0, 1, num_ranks + 1).copy_(cumsum.to(torch::kInt32));
    }

    // Phase 1: Count (expert, src_rank) pairs
    int total_pairs = num_local_experts * num_ranks;
    auto pair_counts = torch::zeros({total_pairs}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (total_recv > 0) {
        const int threads = 256;
        const int blocks = std::min((total_recv + threads - 1) / threads, 1024);
        efa_kernels::ll_recv_count_kernel<<<blocks, threads, 0, stream>>>(
            recv_packed.data_ptr<uint8_t>(),
            recv_cumsum.data_ptr<int32_t>(),
            pair_counts.data_ptr<int32_t>(),
            total_recv, num_ranks, num_local_experts,
            data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token,
            0);  // slot_size=0 for v1 (contiguous flat layout)
    }

    // Phase 2: Build positions + layout_range
    auto packed_recv_count = torch::zeros({num_local_experts}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto packed_recv_layout_range = torch::zeros({num_local_experts, num_ranks}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto expert_cumsum = torch::zeros({num_local_experts + 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto pair_cumsum = torch::zeros({total_pairs}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    efa_kernels::ll_recv_build_positions_kernel<<<1, 1, 0, stream>>>(
        pair_counts.data_ptr<int32_t>(),
        packed_recv_count.data_ptr<int32_t>(),
        packed_recv_layout_range.data_ptr<int64_t>(),
        expert_cumsum.data_ptr<int32_t>(),
        pair_cumsum.data_ptr<int32_t>(),
        num_local_experts, num_ranks);

    // Phase 3: Scatter
    auto packed_recv_src_info = torch::zeros({num_local_experts, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(device));
    auto recv_expert_ids = torch::empty({std::max(total_recv, 1)}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto recv_expert_pos = torch::empty({std::max(total_recv, 1)}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto sort_order_recv = torch::empty({std::max(total_recv, 1)}, torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto pair_write_counters = torch::zeros({total_pairs}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (total_recv > 0) {
        uint8_t* scales_ptr = packed_recv_scales_opt.has_value() ?
            reinterpret_cast<uint8_t*>(packed_recv_scales_opt->data_ptr()) : nullptr;

        efa_kernels::ll_recv_scatter_kernel<<<num_ranks, 256, 0, stream>>>(
            recv_packed.data_ptr<uint8_t>(),
            recv_cumsum.data_ptr<int32_t>(),
            pair_cumsum.data_ptr<int32_t>(),
            expert_cumsum.data_ptr<int32_t>(),
            reinterpret_cast<uint8_t*>(packed_recv_x.data_ptr()),
            scales_ptr,
            reinterpret_cast<__nv_bfloat16*>(packed_recv_src_info.data_ptr()),
            recv_expert_ids.data_ptr<int32_t>(),
            recv_expert_pos.data_ptr<int64_t>(),
            sort_order_recv.data_ptr<int64_t>(),
            pair_write_counters.data_ptr<int32_t>(),
            total_recv, num_ranks, num_local_experts,
            data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token,
            N, hidden_bytes, scale_cols_bytes,
            0);  // slot_size=0 for v1 (contiguous flat layout)
    }

    return std::make_tuple(packed_recv_count, packed_recv_src_info, packed_recv_layout_range,
                           recv_expert_ids.slice(0, 0, total_recv),
                           recv_expert_pos.slice(0, 0, total_recv),
                           sort_order_recv.slice(0, 0, total_recv));
}

// ll_combine_weighted_reduce: fused combine reduction
void ll_combine_weighted_reduce(
    const torch::Tensor& combine_recv_flat,       // [total_send, hidden] bf16
    const torch::Tensor& sorted_send_token_ids,   // [total_send] int32
    const torch::Tensor& global_expert_ids,        // [total_send] int64
    const torch::Tensor& topk_idx,                 // [num_tokens, num_topk] int64
    const torch::Tensor& topk_weights,             // [num_tokens, num_topk] float32
    torch::Tensor& combined_x,                     // [num_tokens, hidden] bf16 output
    int total_send,
    int num_tokens,
    int hidden,
    int num_topk)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = combined_x.device();

    if (total_send == 0) return;

    // Allocate f32 accumulator
    auto combined_x_f32 = torch::zeros({num_tokens, hidden}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    // Weighted reduction kernel
    {
        const int threads = 256;
        efa_kernels::ll_combine_weighted_reduce_kernel<<<total_send, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(combine_recv_flat.data_ptr()),
            sorted_send_token_ids.data_ptr<int64_t>(),
            global_expert_ids.data_ptr<int64_t>(),
            topk_idx.data_ptr<int64_t>(),
            topk_weights.data_ptr<float>(),
            combined_x_f32.data_ptr<float>(),
            nullptr,  // send_cumsum (not used for flat mode)
            total_send, hidden, num_topk, 0, 0);  // num_ranks=0, slot_size=0
    }

    // Convert f32 -> bf16
    {
        int64_t total_elements = static_cast<int64_t>(num_tokens) * hidden;
        const int threads = 256;
        const int blocks = (total_elements + threads - 1) / threads;
        efa_kernels::f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            combined_x_f32.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(combined_x.data_ptr()),
            total_elements);
    }
}

// ll_combine_weighted_reduce_v2: same as above but accepts pre-allocated f32 accumulator
void ll_combine_weighted_reduce_v2(
    const torch::Tensor& combine_recv_flat,       // [total_send, hidden] bf16 or RDMA slot buffer
    const torch::Tensor& sorted_send_token_ids,   // [total_send] int64
    const torch::Tensor& global_expert_ids,        // [total_send] int64
    const torch::Tensor& topk_idx,                 // [num_tokens, num_topk] int64
    const torch::Tensor& topk_weights,             // [num_tokens, num_topk] float32
    torch::Tensor& combined_x,                     // [num_tokens, hidden] bf16 output
    torch::Tensor& combined_x_f32,                 // [num_tokens, hidden] f32 accumulator (pre-allocated, will be zeroed)
    const c10::optional<torch::Tensor>& send_cumsum_opt,  // [num_ranks + 1] int32 — for slot addressing (nullopt if flat)
    int total_send,
    int num_tokens,
    int hidden,
    int num_topk,
    int num_ranks,
    int64_t slot_size)   // >0: RDMA slot-based (bytes), 0: contiguous flat
{
    auto stream = at::cuda::getCurrentCUDAStream();

    if (total_send == 0) return;

    // Zero the f32 accumulator
    combined_x_f32.zero_();

    // Weighted reduction kernel
    {
        const int threads = 256;
        const int32_t* cumsum_ptr = send_cumsum_opt.has_value() ?
            send_cumsum_opt->data_ptr<int32_t>() : nullptr;
        efa_kernels::ll_combine_weighted_reduce_kernel<<<total_send, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(combine_recv_flat.data_ptr()),
            sorted_send_token_ids.data_ptr<int64_t>(),
            global_expert_ids.data_ptr<int64_t>(),
            topk_idx.data_ptr<int64_t>(),
            topk_weights.data_ptr<float>(),
            combined_x_f32.data_ptr<float>(),
            cumsum_ptr,
            total_send, hidden, num_topk, num_ranks, slot_size);
    }

    // Convert f32 -> bf16
    {
        int64_t total_elements = static_cast<int64_t>(num_tokens) * hidden;
        const int threads = 256;
        const int blocks = (total_elements + threads - 1) / threads;
        efa_kernels::f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
            combined_x_f32.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(combined_x.data_ptr()),
            total_elements);
    }
}

// ll_dispatch_route_and_pack_v2: same as v1 but accepts pre-allocated buffers
// Pre-allocated buffers: send_counts, send_cumsum, total_send_out, send_packed_max, sorted_token_ids_max, sorted_local_eids_max
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
ll_dispatch_route_and_pack_v2(
    const torch::Tensor& topk_idx,       // [num_tokens, num_topk] int64
    const torch::Tensor& x_data,         // [num_tokens, data_bpt] uint8
    const c10::optional<torch::Tensor>& x_scales_opt,  // [num_tokens, scale_bpt] uint8 or nullopt
    torch::Tensor& send_counts,          // [num_ranks] int32 — pre-allocated, will be zeroed
    torch::Tensor& send_cumsum,          // [num_ranks + 1] int64 — pre-allocated
    torch::Tensor& total_send_out,       // [1] int32 — pre-allocated
    torch::Tensor& send_packed_max,      // [max_send * packed_bpt] uint8 — pre-allocated max-size
    torch::Tensor& sorted_token_ids_max, // [max_send] int64 — pre-allocated max-size
    torch::Tensor& sorted_local_eids_max,// [max_send] int32 — pre-allocated max-size
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous() && topk_idx.is_cuda());
    EP_HOST_ASSERT(x_data.dim() == 2 && x_data.is_contiguous() && x_data.is_cuda());

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Zero send_counts (reuse pre-allocated)
    send_counts.zero_();

    // Phase 1: Count
    {
        const int threads = 256;
        const int blocks = std::min((num_tokens * num_topk + threads - 1) / threads, 1024);
        efa_kernels::ll_dispatch_count_kernel<<<blocks, threads, 0, stream>>>(
            topk_idx.data_ptr<int64_t>(),
            send_counts.data_ptr<int32_t>(),
            num_tokens, num_topk, num_local_experts);
    }

    // Phase 2: Prefix sum
    efa_kernels::prefix_sum_kernel<<<1, 1, 0, stream>>>(
        send_counts.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        total_send_out.data_ptr<int32_t>(),
        num_ranks);

    // Sync to get total_send
    int total_send = total_send_out.item<int32_t>();

    if (total_send == 0) {
        auto send_packed = send_packed_max.slice(0, 0, 0);
        auto sorted_token_ids = sorted_token_ids_max.slice(0, 0, 0);
        auto sorted_local_eids = sorted_local_eids_max.slice(0, 0, 0);
        return std::make_tuple(send_packed, send_counts, sorted_token_ids, sorted_local_eids, 0);
    }

    // Use pre-allocated max-size buffers, sliced to actual size
    auto send_packed = send_packed_max.slice(0, 0, static_cast<int64_t>(total_send) * packed_bytes_per_token);
    auto sorted_token_ids = sorted_token_ids_max.slice(0, 0, total_send);
    auto sorted_local_eids = sorted_local_eids_max.slice(0, 0, total_send);

    // Phase 3: Scatter + pack — Iter 38: Use all-SM kernel with NC memory access
    const uint8_t* x_scales_ptr = x_scales_opt.has_value() ? x_scales_opt->data_ptr<uint8_t>() : nullptr;
    auto device = topk_idx.device();
    int32_t* rank_write_pos = get_rank_write_pos(device.index());
    cudaMemsetAsync(rank_write_pos, 0, num_ranks * sizeof(int32_t), stream);

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device.index());
    const int total_entries = num_tokens * num_topk;
    const int blocks = std::min(num_sms, (total_entries + 15) / 16);

    efa_kernels::ll_dispatch_pack_allsm_kernel<<<blocks, 256, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        x_data.data_ptr<uint8_t>(),
        x_scales_ptr,
        send_packed.data_ptr<uint8_t>(),
        sorted_token_ids.data_ptr<int64_t>(),
        sorted_local_eids.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        rank_write_pos,
        total_entries, num_topk, num_local_experts,
        data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token);

    return std::make_tuple(send_packed, send_counts, sorted_token_ids, sorted_local_eids, total_send);
}

// ll_recv_unpack_v2: same as v1 but accepts pre-allocated scratch buffers
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ll_recv_unpack_v2(
    const torch::Tensor& recv_packed,           // [total_recv * packed_bpt] uint8
    const torch::Tensor& recv_counts_tensor,    // [num_ranks] int32
    torch::Tensor& packed_recv_x,               // [num_local_experts, N, hidden] — pre-allocated, zero-init
    c10::optional<torch::Tensor> packed_recv_scales_opt,  // [num_local_experts, N, scale_cols] or nullopt
    // Pre-allocated scratch buffers:
    torch::Tensor& recv_cumsum,                 // [num_ranks + 1] int32
    torch::Tensor& pair_counts,                 // [num_local_experts * num_ranks] int32
    torch::Tensor& packed_recv_count,           // [num_local_experts] int32
    torch::Tensor& packed_recv_layout_range,    // [num_local_experts, num_ranks] int64
    torch::Tensor& expert_cumsum,               // [num_local_experts + 1] int32
    torch::Tensor& pair_cumsum,                 // [num_local_experts * num_ranks] int32
    torch::Tensor& packed_recv_src_info,        // [num_local_experts, N] bf16
    torch::Tensor& recv_expert_ids_max,         // [max_recv] int32
    torch::Tensor& recv_expert_pos_max,         // [max_recv] int64
    torch::Tensor& sort_order_recv_max,         // [max_recv] int64
    torch::Tensor& pair_write_counters,         // [num_local_experts * num_ranks] int32
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t N,
    int hidden_bytes,
    int scale_cols_bytes,
    int64_t slot_size)    // >0: RDMA slot-based addressing, 0: contiguous flat
{
    auto stream = at::cuda::getCurrentCUDAStream();

    // Zero scratch buffers that use atomicAdd (must be zero-initialized)
    pair_counts.zero_();
    pair_write_counters.zero_();
    // Iter 34: packed_recv_count and packed_recv_layout_range are fully overwritten
    // by ll_recv_build_positions_kernel — no need to zero them.
    // packed_recv_src_info: scatter kernel only writes valid positions, but downstream
    // only reads via layout_range indices, so no need to zero.

    // Iter 34: Build recv_cumsum using simple prefix_sum_i32_kernel (1 launch)
    // Replaces .to(kInt64).cumsum(0).to(kInt32).copy_() which was 4+ kernel launches
    efa_kernels::prefix_sum_i32_kernel<<<1, 1, 0, stream>>>(
        recv_counts_tensor.data_ptr<int32_t>(),
        recv_cumsum.data_ptr<int32_t>(),
        num_ranks);

    // Iter 35: Fused Phase 1 (count) + Phase 2 (build_positions) in one kernel
    // Uses grid-wide atomic counter to detect last block, which does the prefix sum
    // Iter 36: Use static block counter to avoid per-call torch::zeros allocation
    if (total_recv > 0) {
        const int threads = 256;
        const int blocks = std::min((total_recv + threads - 1) / threads, 1024);

        int32_t* block_counter = get_block_counter(recv_packed.device().index());
        cudaMemsetAsync(block_counter, 0, sizeof(int32_t), stream);

        efa_kernels::ll_recv_count_and_build_kernel<<<blocks, threads, 0, stream>>>(
            recv_packed.data_ptr<uint8_t>(),
            recv_cumsum.data_ptr<int32_t>(),
            pair_counts.data_ptr<int32_t>(),
            packed_recv_count.data_ptr<int32_t>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            expert_cumsum.data_ptr<int32_t>(),
            pair_cumsum.data_ptr<int32_t>(),
            block_counter,
            total_recv, num_ranks, num_local_experts,
            data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token,
            slot_size, blocks);
    } else {
        // total_recv == 0: still need to initialize build_positions outputs
        efa_kernels::ll_recv_build_positions_kernel<<<1, 1, 0, stream>>>(
            pair_counts.data_ptr<int32_t>(),
            packed_recv_count.data_ptr<int32_t>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            expert_cumsum.data_ptr<int32_t>(),
            pair_cumsum.data_ptr<int32_t>(),
            num_local_experts, num_ranks);
    }

    // Phase 3: Scatter — Iter 38: Use all-SM kernel with NC memory access
    if (total_recv > 0) {
        uint8_t* scales_ptr = packed_recv_scales_opt.has_value() ?
            reinterpret_cast<uint8_t*>(packed_recv_scales_opt->data_ptr()) : nullptr;

        // Query SM count for optimal grid size
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, recv_packed.device().index());
        const int blocks = std::min(num_sms, (total_recv + 15) / 16);  // BATCH_SIZE=16

        efa_kernels::ll_recv_scatter_allsm_kernel<<<blocks, 256, 0, stream>>>(
            recv_packed.data_ptr<uint8_t>(),
            recv_cumsum.data_ptr<int32_t>(),
            pair_cumsum.data_ptr<int32_t>(),
            expert_cumsum.data_ptr<int32_t>(),
            reinterpret_cast<uint8_t*>(packed_recv_x.data_ptr()),
            scales_ptr,
            reinterpret_cast<__nv_bfloat16*>(packed_recv_src_info.data_ptr()),
            recv_expert_ids_max.data_ptr<int32_t>(),
            recv_expert_pos_max.data_ptr<int64_t>(),
            sort_order_recv_max.data_ptr<int64_t>(),
            pair_write_counters.data_ptr<int32_t>(),
            total_recv, num_ranks, num_local_experts,
            data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token,
            N, hidden_bytes, scale_cols_bytes, slot_size);
    }

    return std::make_tuple(packed_recv_count, packed_recv_src_info, packed_recv_layout_range,
                           recv_expert_ids_max.slice(0, 0, total_recv),
                           recv_expert_pos_max.slice(0, 0, total_recv),
                           sort_order_recv_max.slice(0, 0, total_recv));
}

// ll_compute_combine_helpers: compute inverse_sort + global_expert_ids in a single call
// Replaces:
//   inverse_sort[sort_order] = arange(total_recv)          (~3 kernel launches)
//   repeat_interleave + sorted_local_eids -> global_expert_ids (~6 kernel launches)
// With: 2 simple CUDA kernels
std::tuple<torch::Tensor, torch::Tensor>
ll_compute_combine_helpers(
    const torch::Tensor& sort_order_recv,       // [total_recv] int64
    const torch::Tensor& send_cumsum,           // [num_ranks + 1] int64 (from dispatch)
    const torch::Tensor& sorted_local_eids,     // [total_send] int32
    torch::Tensor& inverse_sort_max,            // [max_recv] int64 — pre-allocated
    torch::Tensor& global_expert_ids_max,       // [max_send] int64 — pre-allocated
    int total_recv,
    int total_send,
    int num_ranks,
    int num_local_experts)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    // Compute inverse_sort
    if (total_recv > 0) {
        const int threads = 256;
        const int blocks = (total_recv + threads - 1) / threads;
        efa_kernels::ll_compute_inverse_sort_kernel<<<blocks, threads, 0, stream>>>(
            sort_order_recv.data_ptr<int64_t>(),
            inverse_sort_max.data_ptr<int64_t>(),
            total_recv);
    }

    // Compute global_expert_ids
    if (total_send > 0) {
        const int threads = 256;
        const int blocks = (total_send + threads - 1) / threads;
        efa_kernels::ll_compute_global_expert_ids_kernel<<<blocks, threads, 0, stream>>>(
            send_cumsum.data_ptr<int64_t>(),
            sorted_local_eids.data_ptr<int32_t>(),
            global_expert_ids_max.data_ptr<int64_t>(),
            total_send, num_ranks, num_local_experts);
    }

    return std::make_tuple(
        inverse_sort_max.slice(0, 0, total_recv),
        global_expert_ids_max.slice(0, 0, total_send));
}

// per_token_cast_to_fp8: Fused FP8 E4M3 quantization (replaces 8-12 PyTorch ops)
// Input:  x [M, N] bf16
// Output: x_fp8 [M, N] fp8_e4m3, scales [M, G] float32 or packed_scales [M, G/4] int32 (UE8M0)
std::tuple<torch::Tensor, torch::Tensor>
per_token_cast_to_fp8(
    const torch::Tensor& x,   // [M, N] bf16
    bool round_scale,
    bool use_ue8m0)
{
    EP_HOST_ASSERT(x.dim() == 2 && x.is_contiguous() && x.is_cuda());
    EP_HOST_ASSERT(x.scalar_type() == torch::kBFloat16);

    const int M = x.size(0);
    const int N = x.size(1);
    const int group_size = 128;
    EP_HOST_ASSERT(N % group_size == 0);
    const int G = N / group_size;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = x.device();

    auto x_fp8 = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device));

    torch::Tensor scales_out;
    float* scales_ptr = nullptr;
    int32_t* packed_scales_ptr = nullptr;

    if (use_ue8m0) {
        EP_HOST_ASSERT(G % 4 == 0);
        scales_out = torch::zeros({M, G / 4}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        packed_scales_ptr = scales_out.data_ptr<int32_t>();
    } else {
        scales_out = torch::empty({M, G}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        scales_ptr = scales_out.data_ptr<float>();
    }

    if (M > 0) {
        dim3 grid(G, M);
        dim3 block(128);
        efa_kernels::per_token_cast_to_fp8_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(x_fp8.data_ptr()),
            scales_ptr,
            packed_scales_ptr,
            M, N,
            round_scale, use_ue8m0);
    }

    return std::make_tuple(x_fp8, scales_out);
}

// ll_dispatch_route_and_pack_v3: no .item() GPU sync — uses max capacity for send_packed
// Returns total_send via the pre-allocated total_send_out tensor (not as int)
// Caller reads total_send from total_send_out after GPU sync (or accepts max capacity)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ll_dispatch_route_and_pack_v3(
    const torch::Tensor& topk_idx,       // [num_tokens, num_topk] int64
    const torch::Tensor& x_data,         // [num_tokens, data_bpt] uint8
    const c10::optional<torch::Tensor>& x_scales_opt,  // [num_tokens, scale_bpt] uint8 or nullopt
    torch::Tensor& send_counts,          // [num_ranks] int32 — pre-allocated, will be zeroed
    torch::Tensor& send_cumsum,          // [num_ranks + 1] int64 — pre-allocated
    torch::Tensor& total_send_out,       // [1] int32 — pre-allocated
    torch::Tensor& send_packed_max,      // [max_send * packed_bpt] uint8 — pre-allocated max-size
    torch::Tensor& sorted_token_ids_max, // [max_send] int64 — pre-allocated max-size
    torch::Tensor& sorted_local_eids_max,// [max_send] int32 — pre-allocated max-size
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous() && topk_idx.is_cuda());
    EP_HOST_ASSERT(x_data.dim() == 2 && x_data.is_contiguous() && x_data.is_cuda());

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Zero send_counts (reuse pre-allocated)
    send_counts.zero_();

    // Phase 1: Count
    {
        const int threads = 256;
        const int blocks = std::min((num_tokens * num_topk + threads - 1) / threads, 1024);
        efa_kernels::ll_dispatch_count_kernel<<<blocks, threads, 0, stream>>>(
            topk_idx.data_ptr<int64_t>(),
            send_counts.data_ptr<int32_t>(),
            num_tokens, num_topk, num_local_experts);
    }

    // Phase 2: Prefix sum
    efa_kernels::prefix_sum_kernel<<<1, 1, 0, stream>>>(
        send_counts.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        total_send_out.data_ptr<int32_t>(),
        num_ranks);

    // Phase 3: Scatter + pack — Iter 38: Use all-SM kernel with NC memory access
    const uint8_t* x_scales_ptr = x_scales_opt.has_value() ? x_scales_opt->data_ptr<uint8_t>() : nullptr;
    auto device = topk_idx.device();
    int32_t* rank_write_pos = get_rank_write_pos(device.index());
    cudaMemsetAsync(rank_write_pos, 0, num_ranks * sizeof(int32_t), stream);

    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device.index());
    const int total_entries = num_tokens * num_topk;
    const int blocks = std::min(num_sms, (total_entries + 15) / 16);

    efa_kernels::ll_dispatch_pack_allsm_kernel<<<blocks, 256, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        x_data.data_ptr<uint8_t>(),
        x_scales_ptr,
        send_packed_max.data_ptr<uint8_t>(),
        sorted_token_ids_max.data_ptr<int64_t>(),
        sorted_local_eids_max.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        rank_write_pos,
        total_entries, num_topk, num_local_experts,
        data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token);

    // Return max-capacity tensors — caller will slice after reading total_send from send_counts.tolist()
    return std::make_tuple(send_packed_max, send_counts, sorted_token_ids_max, sorted_local_eids_max);
}

// ll_dispatch_count_only: Phase 1+2 only — count + prefix sum (fused in one kernel)
// Returns send_counts (for NCCL exchange). Does NOT read total_send from GPU.
// Caller can start NCCL count exchange immediately after this returns.
void ll_dispatch_count_only(
    const torch::Tensor& topk_idx,       // [num_tokens, num_topk] int64
    torch::Tensor& send_counts,          // [num_ranks] int32 — pre-allocated, will be zeroed
    torch::Tensor& send_cumsum,          // [num_ranks + 1] int64 — pre-allocated
    torch::Tensor& total_send_out,       // [1] int32 — pre-allocated
    int num_ranks,
    int num_local_experts)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous() && topk_idx.is_cuda());

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();

    send_counts.zero_();

    // Iter 35: Fused count + prefix_sum in one kernel launch
    // Iter 36: Use static block counter to avoid per-call torch::zeros allocation
    const int threads = 256;
    const int blocks = std::min((num_tokens * num_topk + threads - 1) / threads, 1024);
    int32_t* block_counter = get_block_counter(topk_idx.device().index());
    cudaMemsetAsync(block_counter, 0, sizeof(int32_t), stream);
    efa_kernels::ll_dispatch_count_and_prefix_sum_kernel<<<blocks, threads, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        send_counts.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        total_send_out.data_ptr<int32_t>(),
        block_counter,
        num_tokens, num_topk, num_local_experts,
        num_ranks, blocks);
}

// ll_dispatch_pack_only: Phase 3 only — scatter + pack
// Uses pre-computed send_cumsum from count_only. Requires x_data ready on GPU.
void ll_dispatch_pack_only(
    const torch::Tensor& topk_idx,              // [num_tokens, num_topk] int64
    const torch::Tensor& x_data,                // [num_tokens, data_bpt] uint8
    const c10::optional<torch::Tensor>& x_scales_opt,  // [num_tokens, scale_bpt] uint8 or nullopt
    const torch::Tensor& send_cumsum,           // [num_ranks + 1] int64 — from count_only
    torch::Tensor& send_packed_max,             // [max_send * packed_bpt] uint8 — pre-allocated
    torch::Tensor& sorted_token_ids_max,        // [max_send] int64 — pre-allocated
    torch::Tensor& sorted_local_eids_max,       // [max_send] int32 — pre-allocated
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token)
{
    EP_HOST_ASSERT(topk_idx.dim() == 2 && topk_idx.is_contiguous() && topk_idx.is_cuda());
    EP_HOST_ASSERT(x_data.dim() == 2 && x_data.is_contiguous() && x_data.is_cuda());

    const int num_tokens = topk_idx.size(0);
    const int num_topk = topk_idx.size(1);
    const int total_entries = num_tokens * num_topk;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto device = topk_idx.device();

    const uint8_t* x_scales_ptr = x_scales_opt.has_value() ? x_scales_opt->data_ptr<uint8_t>() : nullptr;

    // Iter 38: Use all-SM kernel with NC memory access
    // Need rank_write_pos counters (atomicAdd targets, must be zeroed)
    int32_t* rank_write_pos = get_rank_write_pos(device.index());
    cudaMemsetAsync(rank_write_pos, 0, num_ranks * sizeof(int32_t), stream);

    // Query SM count for optimal grid size
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device.index());
    const int blocks = std::min(num_sms, (total_entries + 15) / 16);  // BATCH_SIZE=16

    efa_kernels::ll_dispatch_pack_allsm_kernel<<<blocks, 256, 0, stream>>>(
        topk_idx.data_ptr<int64_t>(),
        x_data.data_ptr<uint8_t>(),
        x_scales_ptr,
        send_packed_max.data_ptr<uint8_t>(),
        sorted_token_ids_max.data_ptr<int64_t>(),
        sorted_local_eids_max.data_ptr<int32_t>(),
        send_cumsum.data_ptr<int64_t>(),
        rank_write_pos,
        total_entries, num_topk, num_local_experts,
        data_bytes_per_token, scale_bytes_per_token, packed_bytes_per_token);
}

}  // namespace deep_ep

// ============================================================================
// GDR signal kernel: copy send_counts to GDR memory and signal pack_done
// ============================================================================
namespace deep_ep {
namespace efa_kernels {

// Small kernel: copy send_counts[num_ranks] from GPU tensor to GDR-mapped memory,
// then signal pack_done flag via MMIO store.
// Launch: <<<1, 1>>>
__global__ void gdr_signal_counts_kernel(
    const int32_t* __restrict__ send_counts_src,  // GPU tensor with send counts
    int32_t* __restrict__ gdr_send_counts,          // GDR-mapped memory for counts
    uint8_t* __restrict__ gdr_pack_done,            // GDR-mapped flag
    int num_ranks)
{
    // Copy counts (use regular stores — these go to GDR-pinned GPU memory
    // which the CPU reads via BAR1 mapping)
    for (int i = 0; i < num_ranks; ++i) {
        gdr_send_counts[i] = send_counts_src[i];
    }
    // System-scope fence ensures all count stores are visible before the flag
    deep_ep::gdr::fence_release_system();
    // Signal pack_done via MMIO store (bypasses GPU caches, ~1us to CPU)
    deep_ep::gdr::st_mmio_b8(gdr_pack_done, 1);
}

}  // namespace efa_kernels

// Host wrapper
void gdr_signal_counts(
    const torch::Tensor& send_counts_tensor,
    int64_t gdr_send_counts_ptr,
    int64_t gdr_pack_done_ptr,
    int num_ranks)
{
    EP_HOST_ASSERT(send_counts_tensor.is_cuda() && send_counts_tensor.is_contiguous());
    EP_HOST_ASSERT(send_counts_tensor.scalar_type() == torch::kInt32);

    auto stream = at::cuda::getCurrentCUDAStream();
    efa_kernels::gdr_signal_counts_kernel<<<1, 1, 0, stream>>>(
        send_counts_tensor.data_ptr<int32_t>(),
        reinterpret_cast<int32_t*>(gdr_send_counts_ptr),
        reinterpret_cast<uint8_t*>(gdr_pack_done_ptr),
        num_ranks);
}

// Iter 41: NC-load gather for reading RDMA-received counts
// Reads scattered int32 values from RDMA buffer using non-cached loads,
// avoiding L2 cache stale data issues.
// src_rdma: pointer to RDMA recv buffer count region
// dst: regular GPU tensor to write gathered values
// stride_int32: stride between peer entries in int32 units (= num_ranks)
// self_rank: our rank (column offset in the scattered layout)
void nc_gather_counts(
    const torch::Tensor& src_rdma,
    const torch::Tensor& dst,
    int num_ranks,
    int stride_int32,
    int self_rank)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    efa_kernels::nc_gather_counts_kernel<<<1, num_ranks, 0, stream>>>(
        reinterpret_cast<const int32_t*>(src_rdma.data_ptr()),
        dst.data_ptr<int32_t>(),
        num_ranks, stride_int32, self_rank);
}

// Iter 43: Raw-pointer version for use from C++ (not pybind11)
namespace efa_kernels {
void nc_gather_counts(
    const int32_t* src, int32_t* dst,
    int num_ranks, int stride_int32, int self_rank,
    cudaStream_t stream)
{
    nc_gather_counts_kernel<<<1, num_ranks, 0, stream>>>(
        src, dst, num_ranks, stride_int32, self_rank);
}
}  // namespace efa_kernels

}  // namespace deep_ep

#endif  // ENABLE_EFA
