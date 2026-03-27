// ============================================================================
// Cooperative Combine Send Kernel
//
// Architecture (matching pplx-garden):
//   1. Wait for tx_ready (CPU signals send buffer available) + NVLink barrier
//   2. Pre-fetch EFA token routing info (source_rank, combine_send_offset,
//      padded_index) into 8-stage pipeline
//   3. Copy EFA-destined tokens from expert output to send buffer (pipelined)
//   4. Signal combine_send_done when all EFA tokens written
//   5. grid.sync()
//   6. Copy NVLink-destined tokens to peer recv buffers via NVLink (pipelined)
//   7. grid.sync(), NVLink barrier, reset state
//
// Token layout: combine uses token_dim = round_up(hidden_dim * x_elemsize, 16)
// (no scale, no metadata — just BF16 expert output packed to int4 alignment)
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
// Cooperative Combine Send Kernel
// ============================================================================
template <unsigned NUM_WARPS, unsigned NODE_SIZE, unsigned DP_SIZE,
          typename TokenDimTy>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1)
a2a_combine_send_kernel(
    const size_t token_dim,
    const size_t rank,
    const uint8_t* __restrict__ expert_x_ptr,
    size_t expert_x_stride,
    uint8_t* __restrict__ tx_ready,
    uint8_t* __restrict__ send_buffer,
    uint8_t* __restrict__ recv_buffer,
    uint32_t* __restrict__ source_rank,
    uint32_t* __restrict__ combine_send_offset,
    uint32_t* __restrict__ padded_index,
    const uint32_t* __restrict__ num_recv_tokens_ptr,
    uint8_t* __restrict__ combine_send_done,
    uint32_t* __restrict__ token_counter,
    uint32_t* __restrict__ sync_counter,
    uint32_t** __restrict__ sync_ptrs,
    uint8_t** __restrict__ recv_ptrs
)
{
    TokenDimTy token_bound(token_dim);
    constexpr size_t NUM_THREADS = NUM_WARPS * 32;

    constexpr size_t NUM_STAGES = 8;
    struct Stage {
        uint32_t offset;
        uint32_t index;
        uint32_t rank;
    };
    __shared__ Stage shared_stages[NUM_STAGES];
    Stage local_stages[NUM_STAGES];

    // Local copy of peer recv ptrs
    uint8_t* recv_ptrs_local[NODE_SIZE];
    #pragma unroll
    for (unsigned i = 0; i < NODE_SIZE; i++) {
        recv_ptrs_local[i] = recv_ptrs[i];
    }

    auto grid = cooperative_groups::this_grid();
    const unsigned rank_node = rank / NODE_SIZE;
    const unsigned warp_id = threadIdx.x / 32;
    const unsigned lane_id = gdr::get_lane_id();

    // Use volatile loads to ensure we read the current iteration's values
    // (GDRCopy-written from CPU; __ldg could serve stale texture cache entries)
    const unsigned num_recv_tokens = gdr::ld_volatile_u32(const_cast<uint32_t*>(num_recv_tokens_ptr));
    const unsigned num_efa_tokens = gdr::ld_volatile_u32(const_cast<uint32_t*>(num_recv_tokens_ptr + 1));

    unsigned token = blockIdx.x;

    auto counter = *sync_counter;

    // ========================================================================
    // Phase 1: Wait for tx_ready + NVLink barrier + pre-fetch first stage
    // ========================================================================
    if (warp_id == 0) {
        if (lane_id == 0) {
            while (gdr::ld_mmio_b8(tx_ready) == 0)
                ;
            // If no EFA tokens, signal done immediately
            if (num_efa_tokens == 0) {
                gdr::st_mmio_b8(combine_send_done, 1);
            }
        }
    } else if (warp_id == 1) {
        if constexpr (NODE_SIZE > 1) {
            auto local_rank = rank % NODE_SIZE;
            if (lane_id < NODE_SIZE) {
                auto* flag = &sync_ptrs[lane_id][local_rank];
                while (gdr::ld_volatile_u32(flag) != counter)
                    ;
            }
        }
    } else if (warp_id == 2) {
        unsigned next_token = token + lane_id * gridDim.x;
        if (next_token < num_recv_tokens && lane_id < NUM_STAGES) {
            // Use volatile loads to bypass L1 cache for GDRCopy-written metadata
            shared_stages[lane_id].offset = gdr::ld_volatile_u32(&combine_send_offset[next_token]);
            shared_stages[lane_id].index = gdr::ld_volatile_u32(&padded_index[next_token]);
            shared_stages[lane_id].rank = gdr::ld_volatile_u32(&source_rank[next_token]);
        }
    }
    __syncthreads();

    auto shared_to_local = [&] {
        #pragma unroll(NUM_STAGES)
        for (unsigned s = 0; s < NUM_STAGES; s++) {
            local_stages[s] = shared_stages[s];
        }
        __syncthreads();
    };

    shared_to_local();

    // ========================================================================
    // Phase 2: Copy EFA-destined tokens to send buffer (pipelined)
    // ========================================================================
    unsigned num_local_efa_tokens = 0;
    while (token < num_efa_tokens) {
        // Pre-fetch next batch
        unsigned next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
        if (threadIdx.x < NUM_STAGES && next_token < num_efa_tokens) {
            shared_stages[threadIdx.x].offset = gdr::ld_volatile_u32(&combine_send_offset[next_token]);
            shared_stages[threadIdx.x].index = gdr::ld_volatile_u32(&padded_index[next_token]);
            shared_stages[threadIdx.x].rank = gdr::ld_volatile_u32(&source_rank[next_token]);
        }
        __syncthreads();

        // Pipelined copy: read from expert output, write to send buffer
        uint4 values[NUM_STAGES];
        for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_bound; i += NUM_THREADS) {
            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_efa_tokens; s++) {
                auto* ptr = reinterpret_cast<const uint4*>(
                    expert_x_ptr + expert_x_stride * local_stages[s].index);
                values[s] = gdr::ld_global_uint4(&ptr[i]);
            }

            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_efa_tokens; s++) {
                unsigned offset = local_stages[s].offset;
                auto token_rank = local_stages[s].rank;
                auto token_node = token_rank / NODE_SIZE;
                // NODE_SIZE==1: write ALL tokens (including self-rank) to send buffer,
                // because the CPU worker handles self via D2D copy from send buffer.
                if (NODE_SIZE == 1 || token_node != rank_node) {
                    auto* x_token_dst = reinterpret_cast<uint4*>(
                        send_buffer + offset * (size_t)token_bound);
                    gdr::st_global_uint4(&x_token_dst[i], values[s]);
                }
            }
        }

        #pragma unroll(NUM_STAGES)
        for (unsigned s = 0; s < NUM_STAGES && token < num_efa_tokens; s++) {
            if (token < num_efa_tokens) {
                num_local_efa_tokens++;
            }
            token += gridDim.x;
        }

        shared_to_local();
    }

    // Signal combine_send_done when all EFA tokens are in send buffer
    if (threadIdx.x == 0) {
        auto num_tokens_done = gdr::add_release_sys_u32(token_counter, num_local_efa_tokens)
                               + num_local_efa_tokens;
        if (num_tokens_done == num_efa_tokens) {
            gdr::st_mmio_b8(combine_send_done, 1);
        }
    }

    // Pre-fetch first NVLink batch
    if (warp_id == 0) {
        unsigned next_token = token + lane_id * gridDim.x;
        if (next_token < num_recv_tokens && lane_id < NUM_STAGES) {
            shared_stages[lane_id].offset = gdr::ld_volatile_u32(&combine_send_offset[next_token]);
            shared_stages[lane_id].index = gdr::ld_volatile_u32(&padded_index[next_token]);
            shared_stages[lane_id].rank = gdr::ld_volatile_u32(&source_rank[next_token]);
        }
    }
    __syncthreads();

    shared_to_local();

    // ========================================================================
    // Phase 3: grid.sync() then copy NVLink-destined tokens
    // ========================================================================
    grid.sync();

    while (token < num_recv_tokens) {
        unsigned next_token = token + (NUM_STAGES + threadIdx.x) * gridDim.x;
        if (threadIdx.x < NUM_STAGES && next_token < num_recv_tokens) {
            shared_stages[threadIdx.x].offset = gdr::ld_volatile_u32(&combine_send_offset[next_token]);
            shared_stages[threadIdx.x].index = gdr::ld_volatile_u32(&padded_index[next_token]);
            shared_stages[threadIdx.x].rank = gdr::ld_volatile_u32(&source_rank[next_token]);
        }
        __syncthreads();

        // Pipelined copy: read from expert output, write to NVLink peer recv buffers
        uint4 values[NUM_STAGES];
        for (unsigned i = threadIdx.x; i * sizeof(uint4) < (size_t)token_bound; i += NUM_THREADS) {
            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_recv_tokens; s++) {
                auto* ptr = reinterpret_cast<const uint4*>(
                    expert_x_ptr + expert_x_stride * local_stages[s].index);
                values[s] = gdr::ld_global_uint4(&ptr[i]);
            }

            #pragma unroll(NUM_STAGES)
            for (unsigned s = 0; s < NUM_STAGES && token + s * gridDim.x < num_recv_tokens; s++) {
                unsigned offset = local_stages[s].offset;
                auto token_rank = local_stages[s].rank;
                auto token_node = token_rank / NODE_SIZE;
                if (token_node == rank_node) {
                    unsigned first_peer = (token_rank / DP_SIZE) * DP_SIZE;
                    #pragma unroll(DP_SIZE)
                    for (unsigned dp_peer = 0; dp_peer < DP_SIZE; dp_peer++) {
                        auto token_peer = (first_peer + dp_peer) % NODE_SIZE;
                        auto* x_token_dst = reinterpret_cast<uint4*>(
                            recv_ptrs_local[token_peer] + offset * (size_t)token_bound);
                        gdr::st_global_uint4(&x_token_dst[i], values[s]);
                    }
                }
            }
        }

        #pragma unroll(NUM_STAGES)
        for (unsigned s = 0; s < NUM_STAGES && token < num_recv_tokens; s++) {
            token += gridDim.x;
        }

        shared_to_local();
    }

    // ========================================================================
    // Phase 4: grid.sync(), NVLink barrier, reset state
    // ========================================================================
    grid.sync();

    if (blockIdx.x == 0) {
        if (warp_id == 0) {
            if (lane_id == 0) {
                *sync_counter = counter + 1;
                *token_counter = 0;
                *tx_ready = 0;
            }
        } else if (warp_id == 1) {
            if constexpr (NODE_SIZE > 1) {
                auto local_rank = rank % NODE_SIZE;
                if (lane_id < NODE_SIZE) {
                    gdr::st_release_u32(&sync_ptrs[lane_id][local_rank + NODE_SIZE],
                                        counter + 1);
                }
            }
        }
    }
}

// ============================================================================
// Host-side launch function declaration
// ============================================================================
int a2a_combine_send(
    size_t num_blocks,
    size_t hidden_dim,
    size_t x_elemsize,
    size_t rank,
    size_t node_size,
    size_t dp_size,
    const uint8_t* expert_x_ptr,
    size_t expert_x_stride,
    uint8_t* tx_ready,
    uint8_t* send_buffer,
    uint8_t* recv_buffer,
    uint32_t* source_rank,
    uint32_t* combine_send_offset,
    uint32_t* padded_index,
    uint32_t* num_recv_tokens_ptr,
    uint8_t* combine_send_done,
    uint32_t* token_counter,
    uint32_t* sync_counter,
    uint32_t** sync_ptrs,
    uint8_t** recv_ptrs,
    uint64_t stream);

}  // namespace coop
}  // namespace deep_ep
