// ============================================================================
// Cooperative Dispatch Recv Kernel
//
// Architecture (matching pplx-garden):
//   1. Wait for num_recv_tokens_flag (CPU signals metadata is ready) and
//      NVLink barrier from dispatch_send.
//   2. Process NVLink/local tokens while waiting for EFA data — copy from
//      recv_buffer/send_buffer to output buffer using source_offset/padded_index.
//   3. NVLink barrier (protect NVLink buffers before combine overwrites).
//   4. Wait for dispatch_recv_flag (CPU signals all EFA data arrived).
//   5. Process EFA tokens using 8-stage pipeline for latency hiding.
//   6. Write out_num_tokens_ptr (per-expert counts for MoE layer).
//   7. Signal dispatch_recv_done via MMIO when last block finishes.
//
// CPU-written GDR arrays consumed by this kernel:
//   - num_recv_tokens_ptr[0] = total recv tokens, [1] = EFA recv tokens
//   - source_rank[i]    = which rank sent token i
//   - source_offset[i]  = position in recv buffer (bit 31 set = NVLink send_ptr)
//   - padded_index[i]   = destination index in output buffer (with expert padding)
//   - tokens_per_expert[e] = per-expert token counts for this rank
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
// Cooperative Dispatch Recv Kernel
// ============================================================================
template <unsigned NUM_WARPS, unsigned NODE_SIZE,
          typename TokenDimTy, typename HiddenDimScaleTy>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1)
a2a_dispatch_recv_kernel(
    const size_t token_dim,
    const size_t token_scale_dim,
    const size_t token_stride,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t x_elemsize,
    size_t x_scale_elemsize,
    size_t num_experts,
    size_t rank,
    size_t world_size,
    int32_t* __restrict__ out_num_tokens_ptr,
    uint8_t* __restrict__ out_x_ptr,
    size_t out_x_stride,
    float* __restrict__ out_x_scale_ptr,
    size_t out_x_scale_stride_elem,
    size_t out_x_scale_stride_token,
    uint32_t* __restrict__ tokens_per_expert,
    uint8_t* __restrict__ send_buffer,
    uint8_t* __restrict__ recv_buffer,
    uint32_t* __restrict__ source_rank,
    uint32_t* __restrict__ source_offset,
    uint32_t* __restrict__ padded_index,
    uint32_t* __restrict__ num_routed,
    uint32_t* __restrict__ num_recv_tokens_ptr,
    uint8_t* __restrict__ num_recv_tokens_flag,
    uint8_t* __restrict__ dispatch_recv_flag,
    uint8_t* __restrict__ dispatch_recv_done,
    uint32_t* __restrict__ grid_counter,
    uint32_t* __restrict__ sync_counter,
    uint32_t** __restrict__ sync_ptrs,
    uint8_t** send_ptrs
)
{
    TokenDimTy token_dim_bound(token_dim);
    HiddenDimScaleTy hidden_dim_scale_bound(hidden_dim_scale);

    // 8-stage pipeline structures for EFA token copy
    struct SharedStage {
        uint32_t src_index;
        uint32_t dst_index;
    };
    struct LocalStage {
        uint4* x_token_src;
        uint4* x_token_dst;
        float* x_scale_src;
        float* x_scale_dst;
    };
    constexpr size_t NUM_STAGES = 8;

    __shared__ SharedStage shared_stage[NUM_STAGES];
    LocalStage local_stage[NUM_STAGES];

    // Lambda: convert shared stage indices to local pointers
    auto shared_to_local = [&] {
        #pragma unroll(NUM_STAGES)
        for (unsigned i = 0; i < NUM_STAGES; ++i) {
            uint32_t src_index = shared_stage[i].src_index;
            uint32_t dst_index = shared_stage[i].dst_index;
            local_stage[i].x_token_src = reinterpret_cast<uint4*>(recv_buffer + src_index * token_stride);
            local_stage[i].x_scale_src = reinterpret_cast<float*>(recv_buffer + src_index * token_stride + (size_t)token_dim_bound);
            local_stage[i].x_token_dst = reinterpret_cast<uint4*>(out_x_ptr + dst_index * out_x_stride);
            local_stage[i].x_scale_dst = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(out_x_scale_ptr) + dst_index * out_x_scale_stride_token);
        }
        __syncthreads();
    };

    auto grid = cooperative_groups::this_grid();
    const unsigned warp_id = threadIdx.x / 32;
    const unsigned lane_id = gdr::get_lane_id();

    const size_t experts_per_rank = (num_experts + world_size - 1) / world_size;
    const size_t first_expert = rank * experts_per_rank;
    const size_t last_expert = min(first_expert + experts_per_rank, num_experts);

    // ========================================================================
    // Phase 1: Wait for CPU to provide routing metadata + NVLink barrier
    // ========================================================================
    auto counter = *sync_counter;

    // Warp 0: wait for num_recv_tokens_flag (CPU wrote metadata)
    if (warp_id == 0) {
        if (lane_id == 0) {
            while (gdr::ld_mmio_b8(num_recv_tokens_flag) == 0)
                ;
        }
    }
    // Warp 1: wait for NVLink barrier from dispatch_send
    else if (warp_id == 1) {
        if constexpr (NODE_SIZE > 1) {
            auto local_rank = rank % NODE_SIZE;
            if (lane_id < NODE_SIZE) {
                auto* flag_ptr = &sync_ptrs[local_rank][lane_id + NODE_SIZE];
                while (gdr::ld_acquire_u32(flag_ptr) != counter)
                    ;
            }
        }
    }
    __syncthreads();

    // System-scope fence to ensure GDRCopy-written metadata (source_offset,
    // padded_index, etc.) is visible to subsequent regular loads. The MMIO
    // flag read above acts as a signal but doesn't guarantee L1 coherence
    // for CPU-originated PCIe BAR writes.
    __threadfence_system();

    // Read metadata written by CPU worker
    const unsigned num_recv_tokens = gdr::ld_volatile_u32(num_recv_tokens_ptr);
    const unsigned num_efa_tokens = gdr::ld_volatile_u32(num_recv_tokens_ptr + 1);

    // ========================================================================
    // Phase 2: Process NVLink/local tokens while waiting for EFA data
    //          Tokens [num_efa_tokens, num_recv_tokens) are from NVLink/local
    // ========================================================================
    // Pre-populate first stages of pipeline (for EFA tokens later)
    // Use volatile loads for GDRCopy-written metadata arrays
    auto next_token = blockIdx.x + threadIdx.x * gridDim.x;
    if (threadIdx.x < NUM_STAGES && next_token < num_efa_tokens) {
        shared_stage[threadIdx.x].src_index = gdr::ld_volatile_u32(&source_offset[next_token]);
        shared_stage[threadIdx.x].dst_index = gdr::ld_volatile_u32(&padded_index[next_token]);
    }
    __syncthreads();

    // Copy NVLink/local tokens into output buffer
    for (unsigned token = num_efa_tokens + blockIdx.x; token < num_recv_tokens;
         token += gridDim.x) {
        auto padded_token = gdr::ld_volatile_u32(&padded_index[token]);
        auto token_rank = gdr::ld_volatile_u32(&source_rank[token]);

        auto local_rank = token_rank % NODE_SIZE;
        auto position = gdr::ld_volatile_u32(&source_offset[token]);
        uint4* x_token_src;
        if (token_rank == rank) {
            // Self: read from send buffer
            x_token_src = reinterpret_cast<uint4*>(send_buffer + position * token_stride);
        } else if (position & (1u << 31)) {
            // NVLink private buffer (bit 31 set)
            x_token_src = reinterpret_cast<uint4*>(
                send_ptrs[local_rank] + (position & ~(1u << 31)) * token_stride);
        } else {
            // NVLink overflow in recv buffer
            x_token_src = reinterpret_cast<uint4*>(recv_buffer + position * token_stride);
        }

        uint4* x_token_dst = reinterpret_cast<uint4*>(out_x_ptr + padded_token * out_x_stride);
        float* x_scale_src = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(x_token_src) + (size_t)token_dim);
        float* x_scale_dst = reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(out_x_scale_ptr) + padded_token * out_x_scale_stride_token);

        for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_dim; i += blockDim.x) {
            const bool has_scale = out_x_scale_ptr && i < (size_t)hidden_dim_scale_bound;
            auto val = gdr::ld_global_uint4(&x_token_src[i]);
            float scale;
            if (has_scale) {
                scale = x_scale_src[i];
            }
            gdr::st_global_uint4(&x_token_dst[i], val);
            if (has_scale) {
                x_scale_dst[i * out_x_scale_stride_elem] = scale;
            }
        }
    }

    // ========================================================================
    // Phase 3: NVLink barrier (protect NVLink buffers before combine)
    // ========================================================================
    if constexpr (NODE_SIZE > 1) {
        grid.sync();
        if (blockIdx.x == 0) {
            if (threadIdx.x == 0) {
                *sync_counter = counter + 1;
            }
            auto local_rank = rank % NODE_SIZE;
            for (unsigned peer = threadIdx.x; peer < (unsigned)NODE_SIZE; peer += blockDim.x) {
                gdr::st_volatile_u32(&sync_ptrs[peer][local_rank], counter + 1);
            }
        }
    }

    shared_to_local();

    // ========================================================================
    // Phase 4: Wait for all EFA data to arrive
    // ========================================================================
    if (warp_id == 0) {
        if (lane_id == 0) {
            while (gdr::ld_mmio_b8(dispatch_recv_flag) == 0)
                ;
        }
    }
    __syncthreads();

    // System fence: ensure RDMA-written recv buffer data (PCIe writes from
    // remote NICs) and any GDRCopy metadata writes are visible to L1.
    __threadfence_system();

    // ========================================================================
    // Phase 5: Process EFA tokens using 8-stage pipeline
    // ========================================================================
    unsigned num_local_tokens = 0;
    unsigned token = blockIdx.x;
    while (token < num_efa_tokens) {
        // Pre-fetch next batch of stages
        auto next_token2 = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
        if (threadIdx.x < NUM_STAGES && next_token2 < num_efa_tokens) {
            shared_stage[threadIdx.x].src_index = gdr::ld_volatile_u32(&source_offset[next_token2]);
            shared_stage[threadIdx.x].dst_index = gdr::ld_volatile_u32(&padded_index[next_token2]);
        }
        __syncthreads();

        for (unsigned s = 0; s < NUM_STAGES && token < num_efa_tokens; s++) {
            uint4* x_token_src = local_stage[s].x_token_src;
            uint4* x_token_dst = local_stage[s].x_token_dst;
            float* x_scale_src = local_stage[s].x_scale_src;
            float* x_scale_dst = local_stage[s].x_scale_dst;

            for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_dim_bound;
                 i += blockDim.x) {
                const bool has_scale = out_x_scale_ptr && i < (size_t)hidden_dim_scale_bound;
                auto val = gdr::ld_global_uint4(&x_token_src[i]);
                float scale;
                if (has_scale) {
                    scale = x_scale_src[i];
                }
                gdr::st_global_uint4(&x_token_dst[i], val);
                if (has_scale) {
                    x_scale_dst[i * out_x_scale_stride_elem] = scale;
                }
            }

            token += gridDim.x;
            num_local_tokens++;
        }

        shared_to_local();
    }

    // ========================================================================
    // Phase 6: Write per-expert token counts for MoE layer
    // ========================================================================
    if (blockIdx.x == 0) {
        for (unsigned expert = threadIdx.x; expert < last_expert - first_expert;
             expert += blockDim.x) {
            out_num_tokens_ptr[expert] = tokens_per_expert[expert];
        }
    }

    // ========================================================================
    // Phase 7: Signal dispatch_recv_done when last block finishes EFA copies
    // ========================================================================
    if (threadIdx.x == 0) {
        auto cnt = gdr::add_release_sys_u32(grid_counter, num_local_tokens) + num_local_tokens;
        if (cnt == num_efa_tokens) {
            gdr::st_mmio_b8(dispatch_recv_done, 1);
            // Reset state for next iteration
            *num_recv_tokens_flag = 0;
            *dispatch_recv_flag = 0;
            *grid_counter = 0;
        }
    }
}

// ============================================================================
// Host-side launch function declaration
// ============================================================================
int a2a_dispatch_recv(
    size_t num_blocks,
    size_t hidden_dim,
    size_t hidden_dim_scale,
    size_t x_elemsize,
    size_t x_scale_elemsize,
    size_t num_experts,
    size_t rank,
    size_t node_size,
    size_t world_size,
    int32_t* out_num_tokens_ptr,
    uint8_t* out_x_ptr,
    size_t out_x_stride,
    uint8_t* out_x_scale_ptr,
    size_t out_x_scale_stride_elem,
    size_t out_x_scale_stride_token,
    uint32_t* tokens_per_expert,
    uint8_t* send_buffer,
    uint8_t* recv_buffer,
    uint32_t* source_rank,
    uint32_t* source_offset,
    uint32_t* padded_index,
    uint32_t* num_routed,
    uint32_t* num_recv_tokens_ptr,
    uint8_t* num_recv_tokens_flag,
    uint8_t* dispatch_recv_flag,
    uint8_t* dispatch_recv_done,
    uint32_t* grid_counter,
    uint32_t* sync_counter,
    uint32_t** sync_ptrs,
    uint8_t** send_ptrs,
    uint64_t stream);

}  // namespace coop
}  // namespace deep_ep
