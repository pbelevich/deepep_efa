// ============================================================================
// Cooperative Dispatch Send Kernel
//
// Replaces the split-kernel dispatch path (ll_dispatch_count + prefix_sum +
// ll_dispatch_pack_allsm + efa_all_to_all) with a single cooperative kernel.
//
// Architecture (matching pplx-garden):
//   Phase 1 (block 0): Count tokens per expert, compute expert_offsets prefix
//                       sum, assign per-token offsets (token_offset). Write
//                       num_routed to GDR. Signal dispatch_route_done via MMIO.
//   Phase 2 (block 0): Wait for NVLink barrier + tx_ready from CPU worker.
//   Phase 3:           grid.sync() — all blocks proceed.
//   Phase 4 (all blocks): Pack token data + scale + metadata into send buffer.
//                          For intra-node NVLink peers: write to recv_ptrs.
//                          Use grid_counter atomic to detect last block done,
//                          then signal dispatch_send_done.
//   Phase 5:           grid.sync(), NVLink barrier signaling.
//
// Buffer layout per packed token (matches deepep existing layout):
//   [0, data_bpt)                     : token data (FP8: hidden bytes, BF16: hidden*2)
//   [data_bpt, data_bpt + scale_bpt)  : scales (FP8 only, else 0)
//   [data_bpt + scale_bpt, +8)        : metadata [token_id:i32, local_expert_id:i32]
//
// Send buffer layout (contiguous, grouped by expert prefix sum position):
//   expert_offsets[e-1]..expert_offsets[e] are tokens for expert e
//   token at position = (expert > 0 ? expert_offsets[expert-1] : 0) + token_offset
//   This gives a flat global position; for per-rank slicing:
//     rank_offset = dst_rank > 0 ? expert_offsets[dst_rank * experts_per_rank - 1] : 0
//     within_rank_offset = position - rank_offset
//
// Template parameters:
//   QUICK: bool — true when num_blocks >= num_tokens (1 block per token)
//   NUM_WARPS: int — warps per block (16 = 512 threads)
//   NODE_SIZE: int — GPUs per node (8 for P5en)
//   TokenDimTy: Fixed<N> or NotFixed — data bytes per token (rounded to int4)
//   HiddenDimScaleTy: Fixed<N> or NotFixed — number of scale elements
//   NumExpertsPerTokenTy: Fixed<N> or NotFixed — topk per token
// ============================================================================

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "../transport/gdr_flags.cuh"
#include "coop_device_utils.cuh"
#include "coop_launch.cuh"

namespace deep_ep {
namespace coop {

// ============================================================================
// Kernel arguments — passed as struct to avoid parameter count limits
// ============================================================================
struct DispatchSendArgs {
    // Input tokens
    const uint8_t* x_data;           // [num_tokens, data_bytes_per_token]
    const uint8_t* x_scales;         // [num_tokens, scale_bytes_per_token] (null if BF16)
    const int32_t* topk_idx;         // [num_tokens, num_experts_per_token]
    const float*   topk_weights;     // [num_tokens, num_experts_per_token]

    // Per-token offset within its expert (assigned in Phase 1)
    uint32_t* token_offset;          // [num_tokens * num_experts_per_token] — written by block 0

    // Expert prefix sums (computed in Phase 1)
    uint32_t* expert_offsets;        // [num_experts] — inclusive prefix sum of per-expert counts

    // Output: packed send buffer (in RDMA-registered memory)
    uint8_t* send_buffer;            // Flat packed buffer, token_stride per row

    // NVLink peer recv buffer pointers (for intra-node direct writes)
    uint8_t** recv_ptrs;             // [NODE_SIZE] array of recv buffer pointers

    // GDR flag device pointers (written by GPU, read by CPU worker)
    uint8_t* dispatch_route_done_flag;
    uint8_t* dispatch_send_done_flag;
    uint8_t* tx_ready_flag;

    // GDR vec: per-expert routing counts (written by GPU, read by CPU)
    uint32_t* num_routed;            // [num_dp_groups * num_experts]

    // Grid-wide atomic counter for tracking packing completion
    uint32_t* grid_counter;          // single uint32, reset to 0 each call

    // Sync counter (monotonically increasing epoch)
    uint32_t* sync_counter;          // pointer to persistent counter

    // NVLink synchronization matrix
    uint32_t** sync_ptrs;            // [NODE_SIZE][2*NODE_SIZE] epoch counters

    // Configuration
    int num_tokens;
    int num_experts;
    int num_experts_per_token;
    int experts_per_rank;            // ceil(num_experts / world_size)

    int rank;
    int dp_rank;
    int dp_group;
    int dp_size;
    int world_size;

    // Token packing sizes
    size_t token_dim;                // data bytes, rounded to int4 alignment
    size_t token_scale_dim;          // scale bytes, rounded to int4 alignment
    size_t token_stride;             // token_dim + token_scale_dim + 16 (metadata)

    // Input strides
    size_t x_stride;                 // stride in bytes between input token rows
    size_t x_scale_stride_elem;      // stride between scale elements (in floats)
    size_t x_scale_stride_token;     // stride between token scale rows (in bytes)
    size_t indices_stride;           // stride of indices array (may differ from num_experts_per_token)
    size_t weights_stride;           // stride of weights array

    // NVLink private buffer
    size_t max_private_tokens;       // max tokens per dp_group for NVLink private writes

    // Bound M (optional dynamic token count)
    const int32_t* bound_m_ptr;      // if non-null, actual num_tokens = *bound_m_ptr
};

// ============================================================================
// ExpertAndOffset: routing info for a single (token, expert) pair
// ============================================================================
static constexpr uint32_t INVALID_EXPERT = 0xFFFFFFFF;

struct ExpertAndOffset {
    uint32_t expert;       // global expert ID (INVALID_EXPERT if masked)
    uint32_t offset;       // within-rank offset (position - rank_offset)
    uint32_t position;     // global position in prefix-sum order
    float    weight;       // topk weight

    __forceinline__ __device__ bool is_valid() const { return expert != INVALID_EXPERT; }
};

// ============================================================================
// ExpertIterator: efficiently look up routing info for a token's experts
// Generic version (runtime num_experts_per_token)
// ============================================================================
template <typename NumExpertsPerTokenTy>
class ExpertIterator {
public:
    __forceinline__ __device__ ExpertIterator(
        NumExpertsPerTokenTy num_experts_per_token,
        const int32_t* indices,
        size_t indices_stride,
        const float* weights,
        size_t weights_stride,
        const uint32_t* token_offset,
        const uint32_t* expert_offsets,
        unsigned token,
        unsigned experts_per_rank
    ) : num_experts_per_token_(num_experts_per_token),
        indices_(indices), indices_stride_(indices_stride),
        weights_(weights), weights_stride_(weights_stride),
        token_offset_(token_offset), expert_offsets_(expert_offsets),
        token_(token), experts_per_rank_(experts_per_rank)
    {}

    __forceinline__ __device__ ExpertAndOffset operator[](unsigned i) {
        const int32_t expert_signed = indices_[token_ * indices_stride_ + i];
        if (expert_signed < 0) {
            return {INVALID_EXPERT, 0, 0, 0.0f};
        }
        const uint32_t expert = static_cast<uint32_t>(expert_signed);
        const float weight = weights_[token_ * weights_stride_ + i];
        const uint32_t offset = token_offset_[token_ * num_experts_per_token_ + i];
        const uint32_t position = (expert > 0 ? expert_offsets_[expert - 1] : 0) + offset;
        const uint32_t dst_rank = expert / experts_per_rank_;
        const uint32_t rank_offset = dst_rank > 0 ? expert_offsets_[dst_rank * experts_per_rank_ - 1] : 0;
        return {expert, position - rank_offset, position, weight};
    }

private:
    NumExpertsPerTokenTy num_experts_per_token_;
    const int32_t* indices_;
    size_t indices_stride_;
    const float* weights_;
    size_t weights_stride_;
    const uint32_t* token_offset_;
    const uint32_t* expert_offsets_;
    unsigned token_;
    unsigned experts_per_rank_;
};

// Fixed<N> specialization: unrolls expert loop at compile time
template <size_t N>
class ExpertIterator<Fixed<N>> {
public:
    __forceinline__ __device__ ExpertIterator(
        Fixed<N> /*num_experts_per_token*/,
        const int32_t* indices,
        size_t indices_stride,
        const float* weights,
        size_t weights_stride,
        const uint32_t* token_offset,
        const uint32_t* expert_offsets,
        unsigned token,
        unsigned experts_per_rank
    ) {
        #pragma unroll(N)
        for (unsigned i = 0; i < N; i++) {
            const auto expert_signed = indices[token * indices_stride + i];
            if (expert_signed < 0) {
                experts_[i] = INVALID_EXPERT;
                weights_[i] = 0.0f;
                offsets_[i] = 0;
                positions_[i] = 0;
                continue;
            }
            const uint32_t expert = static_cast<uint32_t>(expert_signed);
            const auto weight = weights[token * weights_stride + i];
            const auto off = token_offset[token * N + i];
            const uint32_t position = (expert > 0 ? expert_offsets[expert - 1] : 0) + off;
            const uint32_t dst_rank = expert / experts_per_rank;
            const uint32_t rank_offset = dst_rank > 0 ? expert_offsets[dst_rank * experts_per_rank - 1] : 0;
            experts_[i] = expert;
            weights_[i] = weight;
            offsets_[i] = position - rank_offset;
            positions_[i] = position;
        }
    }

    __forceinline__ __device__ ExpertAndOffset operator[](unsigned i) {
        return {experts_[i], offsets_[i], positions_[i], weights_[i]};
    }

private:
    uint32_t experts_[N];
    float weights_[N];
    uint32_t offsets_[N];
    uint32_t positions_[N];
};

// ============================================================================
// Helper: copy one token's data+scale to a destination row.
// Uses uint4 (16-byte) loads/stores with non-cached access for DMA-BUF compat.
//
// dst_row points to the start of the packed token row.
// Layout:  [data | scale | metadata]
// metadata = [token_id:i32, local_expert_id:i32] — written by thread 0 only.
// ============================================================================
template <typename TokenDimTy, typename HiddenDimScaleTy, int NUM_THREADS>
__forceinline__ __device__ void copy_token_to_dst(
    const uint8_t* x_ptr,
    const float* x_scale_ptr,
    size_t x_stride,
    size_t x_scale_stride_elem,
    size_t x_scale_stride_token,
    TokenDimTy token_dim_bound,
    HiddenDimScaleTy hidden_dim_scale_bound,
    uint8_t* dst_row,
    unsigned token,
    uint32_t token_id_meta,
    uint32_t local_eid_meta,
    int tid)
{
    const uint4* src = reinterpret_cast<const uint4*>(x_ptr + token * x_stride);
    uint4* dst = reinterpret_cast<uint4*>(dst_row);

    // Copy token data (uint4 = 16 bytes at a time)
    for (unsigned i = tid; i * sizeof(uint4) < (size_t)token_dim_bound; i += NUM_THREADS) {
        uint4 val = gdr::ld_global_uint4(&src[i]);

        // Also copy scale in the same loop iteration if applicable
        const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;

        gdr::st_global_uint4(&dst[i], val);

        if (has_scale) {
            const float* scale_src = reinterpret_cast<const float*>(
                reinterpret_cast<const uint8_t*>(x_scale_ptr) + token * x_scale_stride_token);
            float scale_val = *(scale_src + i * x_scale_stride_elem);
            *(reinterpret_cast<float*>(dst_row + (size_t)token_dim_bound) + i) = scale_val;
        }
    }

    // Thread 0 writes metadata at the end of the row
    if (tid == 0) {
        size_t meta_offset = (size_t)token_dim_bound;
        if (x_scale_ptr) {
            // scale_dim is after data, metadata is after scale
            // In deepep layout: meta at data_bpt + scale_bpt
            // But token_dim_bound is already the data part (rounded to int4)
            // We need the actual position: after data + scale
            // For pplx-garden compat: metadata is at token_dim + token_scale_dim
            // Since we don't have token_scale_dim in template, compute from hidden_dim_scale_bound
            // Actually, metadata is at the fixed offset within token_stride
            // The simplest: data_end + scale_end
            meta_offset = (size_t)token_dim_bound +
                          (size_t)hidden_dim_scale_bound * sizeof(float);
            // Round up to be safe — actually pplx-garden puts it at token_dim + scale_bytes
        }
        int32_t* meta_ptr = reinterpret_cast<int32_t*>(dst_row + meta_offset);
        meta_ptr[0] = static_cast<int32_t>(token_id_meta);
        meta_ptr[1] = static_cast<int32_t>(local_eid_meta);
    }
}

// ============================================================================
// Cooperative Dispatch Send Kernel
// ============================================================================
template <bool QUICK, int NUM_WARPS, int NODE_SIZE,
          typename TokenDimTy, typename HiddenDimScaleTy,
          typename NumExpertsPerTokenTy>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1)
a2a_dispatch_send_kernel(
    const size_t token_dim,
    const size_t token_scale_dim,
    const size_t token_stride,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t max_private_tokens,
    size_t rank,
    size_t dp_size,
    size_t node_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t* __restrict__ bound_m_ptr,
    const uint8_t* __restrict__ x_ptr,
    size_t x_stride,
    const float* __restrict__ x_scale_ptr,
    size_t x_scale_stride_elem,
    size_t x_scale_stride_token,
    const int32_t* __restrict__ indices,
    size_t indices_stride,
    const float* __restrict__ weights,
    size_t weights_stride,
    uint32_t* __restrict__ token_offset,
    uint32_t* __restrict__ num_routed,
    uint32_t* __restrict__ expert_offsets,
    uint8_t* __restrict__ dispatch_route_done,
    uint8_t* __restrict__ dispatch_send_done,
    uint8_t* __restrict__ tx_ready,
    uint8_t* __restrict__ send_buffer,
    uint32_t* __restrict__ grid_counter,
    uint32_t* __restrict__ sync_counter,
    uint32_t** __restrict__ sync_ptrs,
    uint8_t** __restrict__ recv_ptrs,
    size_t hidden_dim_scale
)
{
    TokenDimTy token_dim_bound(token_dim);
    HiddenDimScaleTy hidden_dim_scale_bound(hidden_dim_scale);
    NumExpertsPerTokenTy num_experts_per_token_bound(num_experts_per_token);

    namespace cg = cooperative_groups;
    auto grid = cg::this_grid();

    extern __shared__ uint8_t shared_memory[];
    constexpr size_t NUM_THREADS = NUM_WARPS * 32;
    const size_t warp_id = threadIdx.x / 32;
    const size_t lane_id = gdr::get_lane_id();

    uint32_t counter = *sync_counter;

    const size_t node_rank = rank / NODE_SIZE;
    const size_t dp_group = rank / dp_size;
    const size_t experts_per_rank = (num_experts + world_size - 1) / world_size;

    const size_t num_send_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens;

    // ========================================================================
    // Phase 1: Count tokens per expert, assign token_offset, compute
    //          expert_offsets prefix sum, write num_routed, signal route_done.
    //          (Block 0 only)
    // ========================================================================
    if (blockIdx.x == 0) {
        uint32_t* tokens_per_expert = reinterpret_cast<uint32_t*>(shared_memory);

        // Initialize counts to 0
        for (uint32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
            tokens_per_expert[i] = 0;
        }
        __syncthreads();

        // Count tokens and assign per-expert offset atomically
        for (uint32_t i = threadIdx.x; i < num_send_tokens * num_experts_per_token_bound;
             i += blockDim.x) {
            const uint32_t token = i / num_experts_per_token_bound;
            const uint32_t index = i % num_experts_per_token_bound;
            const int32_t expert_signed = __ldg(reinterpret_cast<const int32_t*>(&indices[token * indices_stride + index]));

            // Skip masked entries (expert == -1)
            if (expert_signed >= 0 && static_cast<uint32_t>(expert_signed) < num_experts) {
                const uint32_t expert = static_cast<uint32_t>(expert_signed);
                // Assign offset within expert's range
                token_offset[i] = atomicAdd(&tokens_per_expert[expert], 1);
            } else {
                token_offset[i] = 0;  // Placeholder; will be skipped in packing phase
            }
        }
        __syncthreads();

        // Write per-expert counts to GDR-mapped num_routed (CPU worker reads this)
        uint32_t* local_num_routed = num_routed + dp_group * num_experts;
        const uint32_t i = threadIdx.x;
        uint32_t expert_offset = 0;
        if (i < num_experts) {
            expert_offset = tokens_per_expert[i];
            local_num_routed[i] = expert_offset;
        }
        __syncthreads();

        // Signal dispatch_route_done ASAP (CPU worker starts route exchange)
        if (threadIdx.x == 0) {
            gdr::st_mmio_b8(dispatch_route_done, 1);
        }

        // Compute inclusive prefix sum over expert counts → expert_offsets
        // Step 1: Intra-warp prefix sum using shuffle
        const uint32_t num_warps_needed = (num_experts + 31) / 32;
        uint32_t* expert_sums = reinterpret_cast<uint32_t*>(shared_memory);

        for (int offset = 1; offset < 32; offset <<= 1) {
            unsigned warp_sum = __shfl_up_sync(0xFFFFFFFF, expert_offset, offset);
            if (lane_id >= (unsigned)offset) {
                expert_offset += warp_sum;
            }
        }
        if (lane_id == 31) {
            expert_sums[warp_id] = expert_offset;
        }
        __syncthreads();

        // Step 2: Sum warp totals in warp 0
        if (warp_id == 0) {
            uint32_t total = (lane_id < num_warps_needed) ? expert_sums[lane_id] : 0;
            for (int offset = 1; offset < (int)num_warps_needed; offset <<= 1) {
                unsigned s = __shfl_up_sync(0xFFFFFFFF, total, offset);
                if (lane_id >= (unsigned)offset) {
                    total += s;
                }
            }
            if (lane_id < num_warps_needed) {
                expert_sums[lane_id] = total;
            }
        }
        __syncthreads();

        // Step 3: Write inclusive expert_offsets
        if (i < num_experts) {
            if (warp_id > 0) {
                expert_offsets[i] = expert_sums[warp_id - 1] + expert_offset;
            } else {
                expert_offsets[i] = expert_offset;
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // Phase 2: Wait for NVLink barrier from previous combine + tx_ready
    //          (Block 0 only)
    // ========================================================================
    if (NODE_SIZE > 1) {
        if (blockIdx.x == 0) {
            auto local_rank = rank % NODE_SIZE;
            for (unsigned peer = threadIdx.x; peer < (unsigned)NODE_SIZE; peer += blockDim.x) {
                while (gdr::ld_volatile_u32(&sync_ptrs[local_rank][peer]) != counter)
                    ;
            }
        }
    }

    // Wait for tx_ready (CPU worker sets this when send buffer is available)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (gdr::ld_mmio_b8(tx_ready) == 0)
            ;
    }

    // ========================================================================
    // Phase 3: Grid sync — all blocks have expert_offsets, token_offset ready
    // ========================================================================
    grid.sync();

    // Update sync_counter for next iteration
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *sync_counter = counter + 1;
    }

    // ========================================================================
    // Phase 4: Pack tokens into send buffer (all blocks participate)
    //
    // Strategy follows pplx-garden exactly:
    //   - QUICK mode (Fixed TokenDim): read token data into registers, then
    //     write to all destinations. For NVLink peers: write after a second
    //     grid.sync() (ensures send buffer is fully populated for CPU worker
    //     before NVLink writes which may overlap with RDMA).
    //   - Non-QUICK mode: iterate tokens with grid stride, write to send_buffer
    //     first (all non-NVLink destinations), then second pass for NVLink.
    //
    // In both modes: dispatch_send_done is signaled as soon as the send buffer
    // is complete (before NVLink writes), so CPU worker can start RDMA early.
    // ========================================================================
    if constexpr (QUICK) {
        unsigned token = blockIdx.x;
        if (token < num_send_tokens) {
            const uint4* x_token_src = reinterpret_cast<const uint4*>(x_ptr + token * x_stride);
            const float* x_scale_src = x_scale_ptr ?
                reinterpret_cast<const float*>(
                    reinterpret_cast<const uint8_t*>(x_scale_ptr) + token * x_scale_stride_token)
                : nullptr;

            ExpertIterator<NumExpertsPerTokenTy> expert_iterator(
                num_experts_per_token_bound,
                indices, indices_stride,
                weights, weights_stride,
                token_offset, expert_offsets,
                token, experts_per_rank
            );

            if constexpr (std::is_same_v<TokenDimTy, NotFixed>) {
                // ---- NotFixed token dim: single-pass, read+write inline ----
                for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_dim_bound;
                     i += NUM_THREADS) {
                    const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;

                    uint4 val = gdr::ld_global_uint4(&x_token_src[i]);
                    float scale_val;
                    if (has_scale) {
                        scale_val = *(x_scale_src + i * x_scale_stride_elem);
                    }

                    #pragma unroll
                    for (unsigned e = 0; e < (unsigned)num_experts_per_token_bound; e++) {
                        auto route = expert_iterator[e];
                        if (!route.is_valid()) continue;
                        const uint32_t dst_rank = route.expert / experts_per_rank;
                        const uint32_t dst_node = dst_rank / NODE_SIZE;

                        // NVLink intra-node peers with space in private buffer
                        if (dst_node == node_rank && dst_rank != rank &&
                            route.offset < max_private_tokens) {
                            if (dst_rank % dp_size == rank % dp_size) {
                                const uint32_t local_peer = dst_rank % NODE_SIZE;
                                uint8_t* token_ptr = recv_ptrs[local_peer] +
                                    (dp_group * max_private_tokens + route.offset) * token_stride;
                                uint4* dst = reinterpret_cast<uint4*>(token_ptr);
                                gdr::st_global_uint4(&dst[i], val);
                                if (has_scale) {
                                    *(reinterpret_cast<float*>(token_ptr + (size_t)token_dim_bound) + i) = scale_val;
                                }
                            }
                        } else {
                            // Send buffer (EFA remote, self, or overflow NVLink)
                            uint8_t* token_ptr = send_buffer + route.position * token_stride;
                            uint4* dst = reinterpret_cast<uint4*>(token_ptr);
                            gdr::st_global_uint4(&dst[i], val);
                            if (has_scale) {
                                *(reinterpret_cast<float*>(token_ptr + (size_t)token_dim_bound) + i) = scale_val;
                            }
                        }
                    }
                }

                // Signal dispatch_send_done when last block finishes
                if (threadIdx.x == 0) {
                    auto cnt = gdr::add_release_sys_u32(grid_counter, 1) + 1;
                    if (cnt == num_send_tokens) {
                        gdr::st_mmio_b8(dispatch_send_done, 1);
                        *grid_counter = 0;
                    }
                }
            } else {
                // ---- Fixed token dim: two-pass (registers → send buffer, then NVLink) ----
                constexpr size_t TOKEN_DIM = TokenDimTy::Value;
                constexpr size_t NUM_STEPS = (TOKEN_DIM + NUM_THREADS * sizeof(uint4) - 1) /
                                              (NUM_THREADS * sizeof(uint4));

                uint4 vals[NUM_STEPS];
                float scales[NUM_STEPS];

                // Read token data into registers
                #pragma unroll(NUM_STEPS)
                for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM;
                     i += NUM_THREADS, s++) {
                    const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;
                    vals[s] = gdr::ld_global_uint4(&x_token_src[i]);
                    if (has_scale) {
                        scales[s] = *(x_scale_src + i * x_scale_stride_elem);
                    }
                }

                // Pass 1: Write to send buffer (for EFA/self/overflow)
                #pragma unroll
                for (unsigned e = 0; e < (unsigned)num_experts_per_token_bound; e++) {
                    auto route = expert_iterator[e];
                    if (!route.is_valid()) continue;
                    const uint32_t dst_rank = route.expert / experts_per_rank;
                    const uint32_t dst_node = dst_rank / NODE_SIZE;

                    if (dst_node != node_rank || dst_rank == rank ||
                        route.offset >= max_private_tokens) {
                        uint8_t* token_ptr = send_buffer + route.position * token_stride;
                        uint4* dst = reinterpret_cast<uint4*>(token_ptr);
                        for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM;
                             i += NUM_THREADS, s++) {
                            const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;
                            gdr::st_global_uint4(&dst[i], vals[s]);
                            if (has_scale) {
                                *(reinterpret_cast<float*>(token_ptr + (size_t)token_dim_bound) + i) = scales[s];
                            }
                        }
                    }
                }

                __syncthreads();

                // Signal send_done when all blocks have written send buffer
                if (threadIdx.x == 0) {
                    auto cnt = gdr::add_release_sys_u32(grid_counter, 1) + 1;
                    if (cnt == num_send_tokens) {
                        gdr::st_mmio_b8(dispatch_send_done, 1);
                        *grid_counter = 0;
                    }
                }

                // grid.sync() so CPU worker can start RDMA on send buffer
                // while GPU does NVLink writes below
                grid.sync();

                // Pass 2: Write to NVLink peers
                #pragma unroll
                for (unsigned e = 0; e < (unsigned)num_experts_per_token_bound; e++) {
                    auto route = expert_iterator[e];
                    if (!route.is_valid()) continue;
                    const uint32_t dst_rank = route.expert / experts_per_rank;
                    const uint32_t dst_node = dst_rank / NODE_SIZE;

                    if (dst_node == node_rank && dst_rank != rank &&
                        route.offset < max_private_tokens) {
                        if (dst_rank % dp_size == rank % dp_size) {
                            const uint32_t local_peer = dst_rank % NODE_SIZE;
                            uint8_t* token_ptr = recv_ptrs[local_peer] +
                                (dp_group * max_private_tokens + route.offset) * token_stride;
                            uint4* dst = reinterpret_cast<uint4*>(token_ptr);
                            for (unsigned i = threadIdx.x, s = 0; i * sizeof(uint4) < TOKEN_DIM;
                                 i += NUM_THREADS, s++) {
                                const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;
                                gdr::st_global_uint4(&dst[i], vals[s]);
                                if (has_scale) {
                                    *(reinterpret_cast<float*>(token_ptr + (size_t)token_dim_bound) + i) = scales[s];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Blocks beyond num_send_tokens: still need to participate in grid.sync()
            if constexpr (!std::is_same_v<TokenDimTy, NotFixed>) {
                grid.sync();
            }
        }
    } else {
        // ================================================================
        // Non-QUICK mode: blocks iterate over tokens with grid stride
        // ================================================================

        // Pass 1: Write to send buffer (non-NVLink destinations)
        unsigned num_local_tokens = 0;
        for (unsigned token = blockIdx.x; token < num_send_tokens;
             token += gridDim.x, num_local_tokens++) {
            const uint4* x_token_src = reinterpret_cast<const uint4*>(x_ptr + token * x_stride);
            const float* x_scale_src = x_scale_ptr ?
                reinterpret_cast<const float*>(
                    reinterpret_cast<const uint8_t*>(x_scale_ptr) + token * x_scale_stride_token)
                : nullptr;

            ExpertIterator<NumExpertsPerTokenTy> expert_iterator(
                num_experts_per_token_bound,
                indices, indices_stride,
                weights, weights_stride,
                token_offset, expert_offsets,
                token, experts_per_rank
            );

            for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_dim_bound;
                 i += blockDim.x) {
                const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;

                uint4 val = gdr::ld_global_uint4(&x_token_src[i]);
                float scale_val;
                if (has_scale) {
                    scale_val = *(x_scale_src + i * x_scale_stride_elem);
                }

                #pragma unroll
                for (unsigned e = 0; e < (unsigned)num_experts_per_token_bound; e++) {
                    auto route = expert_iterator[e];
                    if (!route.is_valid()) continue;
                    const uint32_t dst_rank = route.expert / experts_per_rank;
                    const uint32_t dst_node = dst_rank / NODE_SIZE;

                    // Skip NVLink peers (handled in pass 2)
                    if (dst_node == node_rank && dst_rank != rank &&
                        route.offset < max_private_tokens) {
                        continue;
                    }

                    uint8_t* token_ptr = send_buffer + route.position * token_stride;
                    uint4* dst = reinterpret_cast<uint4*>(token_ptr);
                    gdr::st_global_uint4(&dst[i], val);
                    if (has_scale) {
                        *(reinterpret_cast<float*>(token_ptr + (size_t)token_dim_bound) + i) = scale_val;
                    }
                }
            }
        }
        __syncthreads();

        // Signal send_done when all send-buffer writes are complete
        if (threadIdx.x == 0) {
            auto cnt = gdr::add_release_sys_u32(grid_counter, num_local_tokens) + num_local_tokens;
            if (cnt == num_send_tokens) {
                gdr::st_mmio_b8(dispatch_send_done, 1);
                *grid_counter = 0;
            }
        }

        // Pass 2: NVLink writes (only for intra-node, non-self peers)
        if (NODE_SIZE > 1) {
            for (unsigned token = blockIdx.x; token < num_send_tokens; token += gridDim.x) {
                const uint4* x_token_src = reinterpret_cast<const uint4*>(x_ptr + token * x_stride);
                const float* x_scale_src = x_scale_ptr ?
                    reinterpret_cast<const float*>(
                        reinterpret_cast<const uint8_t*>(x_scale_ptr) + token * x_scale_stride_token)
                    : nullptr;

                ExpertIterator<NumExpertsPerTokenTy> expert_iterator(
                    num_experts_per_token_bound,
                    indices, indices_stride,
                    weights, weights_stride,
                    token_offset, expert_offsets,
                    token, experts_per_rank
                );

                for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_dim_bound;
                     i += blockDim.x) {
                    const bool has_scale = x_scale_ptr && i < (size_t)hidden_dim_scale_bound;

                    uint4 val = gdr::ld_global_uint4(&x_token_src[i]);
                    float scale_val;
                    if (has_scale) {
                        scale_val = *(x_scale_src + i * x_scale_stride_elem);
                    }

                    #pragma unroll
                    for (unsigned e = 0; e < (unsigned)num_experts_per_token_bound; e++) {
                        auto route = expert_iterator[e];
                        if (!route.is_valid()) continue;
                        const uint32_t dst_rank = route.expert / experts_per_rank;
                        const uint32_t dst_node = dst_rank / NODE_SIZE;

                        if (dst_node == node_rank && dst_rank != rank &&
                            route.offset < max_private_tokens) {
                            if (dst_rank % dp_size == rank % dp_size) {
                                const uint32_t local_peer = dst_rank % NODE_SIZE;
                                uint8_t* token_ptr = recv_ptrs[local_peer] +
                                    (dp_group * max_private_tokens + route.offset) * token_stride;
                                uint4* dst = reinterpret_cast<uint4*>(token_ptr);
                                gdr::st_global_uint4(&dst[i], val);
                                if (has_scale) {
                                    *(reinterpret_cast<float*>(token_ptr + (size_t)token_dim_bound) + i) = scale_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ========================================================================
    // Phase 5: NVLink barrier — signal peers that dispatch_send is complete
    // ========================================================================
    if (NODE_SIZE > 1) {
        grid.sync();

        if (blockIdx.x == 0) {
            auto local_rank = rank % NODE_SIZE;
            if (threadIdx.x < (unsigned)NODE_SIZE) {
                auto* flag = &sync_ptrs[threadIdx.x][local_rank + NODE_SIZE];
                gdr::st_release_u32(flag, counter + 1);
            }
        }
    }
}

// ============================================================================
// Host-side launch function declaration
// ============================================================================
int a2a_dispatch_send(
    size_t num_blocks,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t max_private_tokens,
    size_t rank,
    size_t dp_size,
    size_t node_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t* bound_m_ptr,
    const uint8_t* x_ptr,
    size_t x_elemsize,
    size_t x_stride,
    const uint8_t* x_scale_ptr,
    size_t x_scale_elemsize,
    size_t x_scale_stride_elem,
    size_t x_scale_stride_token,
    const int32_t* indices,
    size_t indices_stride,
    const float* weights,
    size_t weights_stride,
    uint32_t* token_offset,
    uint32_t* num_routed,
    uint32_t* expert_offsets,
    uint8_t* dispatch_route_done,
    uint8_t* dispatch_send_done,
    uint8_t* tx_ready,
    uint8_t* send_buffer,
    uint32_t* grid_counter,
    uint32_t* sync_counter,
    uint32_t** sync_ptrs,
    uint8_t** recv_ptrs,
    uint64_t stream);

}  // namespace coop
}  // namespace deep_ep
