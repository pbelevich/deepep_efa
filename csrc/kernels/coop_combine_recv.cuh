// ============================================================================
// Cooperative Combine Recv Kernel
//
// Architecture (matching pplx-garden):
//   1. Pre-compute token positions (expert_offsets + token_offset) into smem.
//   2. Wait for combine_recv_flag (CPU signals all combine data arrived) +
//      NVLink barrier from combine_send.
//   3. Weighted reduction: for each output token, accumulate
//      weight[k] * recv_buffer[position[k]] for k in 0..num_experts_per_token.
//   4. grid.sync(), signal combine_recv_done, NVLink barrier for next dispatch.
//
// This kernel operates on BF16 expert outputs, accumulates in FP32, and
// writes the result as BF16 (matching deepep's existing combine path).
// ============================================================================

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "../transport/gdr_flags.cuh"
#include "coop_device_utils.cuh"
#include "coop_launch.cuh"

namespace deep_ep {
namespace coop {

// ============================================================================
// Vectorized BF16→FP32 conversion helpers for combine reduction
// ============================================================================

// Load uint4 (16 bytes = 8 BF16 values), convert to 8 floats
__forceinline__ __device__ void bf16x8_to_fp32x8(
    const uint4& data, float* out)
{
    auto from_u32 = [](uint32_t v) -> float2 {
        union { uint32_t u; __nv_bfloat162 b; } tmp;
        tmp.u = v;
        return __bfloat1622float2(tmp.b);
    };
    auto v0 = from_u32(data.x); out[0] = v0.x; out[1] = v0.y;
    auto v1 = from_u32(data.y); out[2] = v1.x; out[3] = v1.y;
    auto v2 = from_u32(data.z); out[4] = v2.x; out[5] = v2.y;
    auto v3 = from_u32(data.w); out[6] = v3.x; out[7] = v3.y;
}

// Pack 8 FP32 values back to uint4 (8 BF16)
__forceinline__ __device__ uint4 fp32x8_to_bf16x8(const float* in)
{
    auto pack = [](float a, float b) -> uint32_t {
        uint32_t v;
        asm volatile("{ cvt.rn.bf16x2.f32 %0, %1, %2; }"
                     : "=r"(v) : "f"(b), "f"(a));
        return v;
    };
    return make_uint4(
        pack(in[0], in[1]),
        pack(in[2], in[3]),
        pack(in[4], in[5]),
        pack(in[6], in[7])
    );
}

// ============================================================================
// Cooperative Combine Recv Kernel
// ============================================================================
template <unsigned NUM_WARPS, unsigned NODE_SIZE,
          typename NumExpertsPerTokenTy>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1)
a2a_combine_recv_kernel(
    const size_t token_dim,
    size_t hidden_dim,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t rank,
    size_t world_size,
    size_t num_tokens,
    const int32_t* bound_m_ptr,
    const int32_t* __restrict__ indices_ptr,
    const size_t indices_stride,
    const float* __restrict__ weights_ptr,
    const size_t weights_stride,
    __nv_bfloat16* __restrict__ out_tokens_ptr,
    size_t out_tokens_stride,
    uint8_t accumulate,
    uint8_t* __restrict__ recv_buffer,
    uint32_t* __restrict__ token_offset,
    uint32_t* __restrict__ expert_offsets,
    uint8_t* __restrict__ combine_recv_flag,
    uint8_t* __restrict__ combine_recv_done,
    uint32_t* __restrict__ sync_counter,
    uint32_t** __restrict__ sync_ptrs
)
{
    extern __shared__ uint8_t shared_memory[];

    auto grid = cooperative_groups::this_grid();
    const unsigned warp_id = threadIdx.x / 32;
    const unsigned lane_id = gdr::get_lane_id();

    const size_t num_send_tokens = bound_m_ptr ? *bound_m_ptr : num_tokens;

    NumExpertsPerTokenTy num_experts_per_token_bound(num_experts_per_token);

    // Pre-compute token positions into shared memory
    // INVALID_POSITION sentinel for masked (-1) expert indices
    constexpr uint32_t INVALID_POSITION = 0xFFFFFFFF;
    uint32_t* positions = reinterpret_cast<uint32_t*>(shared_memory);
    {
        uint32_t i = threadIdx.x;
        for (;;) {
            const uint32_t local_token = i / num_experts_per_token_bound;
            const uint32_t token = blockIdx.x + local_token * gridDim.x;
            const uint32_t route = i % num_experts_per_token_bound;
            if (token >= num_send_tokens) break;

            const uint32_t global_slot = token * num_experts_per_token_bound + route;
            const uint32_t local_slot = local_token * num_experts_per_token_bound + route;

            const int32_t expert_signed = reinterpret_cast<const int32_t*>(indices_ptr)[token * indices_stride + route];
            if (expert_signed < 0 || static_cast<uint32_t>(expert_signed) >= num_experts) {
                positions[local_slot] = INVALID_POSITION;
            } else {
                const uint32_t expert = static_cast<uint32_t>(expert_signed);
                const uint32_t offset = token_offset[global_slot];
                const uint32_t position = (expert > 0 ? expert_offsets[expert - 1] : 0) + offset;
                positions[local_slot] = position;
            }
            i += blockDim.x;
        }
        __syncthreads();
    }

    // ========================================================================
    // Wait for combine_recv_flag + NVLink barrier
    // ========================================================================
    auto counter = *sync_counter;
    if (warp_id == 0) {
        if (lane_id == 0) {
            while (gdr::ld_mmio_b8(combine_recv_flag) == 0)
                ;
        }
    } else if (warp_id == 1 && NODE_SIZE > 1) {
        auto local_rank = rank % NODE_SIZE;
        if (lane_id < NODE_SIZE) {
            auto* flag_ptr = &sync_ptrs[local_rank][lane_id + NODE_SIZE];
            while (gdr::ld_acquire_u32(flag_ptr) != counter)
                ;
        }
    }
    __syncthreads();

    // System fence: ensure RDMA-written combine recv buffer data is visible
    __threadfence_system();

    // ========================================================================
    // Weighted reduction: accumulate expert outputs into output token buffer
    // ========================================================================
    constexpr unsigned VEC_SIZE = 8;  // 8 BF16 = 16 bytes = uint4

    for (unsigned token = blockIdx.x, local_token = 0;
         token < num_send_tokens;
         token += gridDim.x, local_token++) {

        __nv_bfloat16* dst_ptr = out_tokens_ptr + token * out_tokens_stride;

        if constexpr (std::is_same_v<NumExpertsPerTokenTy, NotFixed>) {
            // Runtime num_experts_per_token
            for (unsigned j = threadIdx.x * VEC_SIZE; j < hidden_dim;
                 j += blockDim.x * VEC_SIZE) {
                float acc[VEC_SIZE];
                if (accumulate) {
                    // Load existing output
                    uint4 existing = gdr::ld_global_uint4(
                        reinterpret_cast<uint4*>(dst_ptr + j));
                    bf16x8_to_fp32x8(existing, acc);
                } else {
                    #pragma unroll
                    for (unsigned v = 0; v < VEC_SIZE; v++) acc[v] = 0.0f;
                }

                #pragma unroll(8)
                for (unsigned k = 0; k < (unsigned)num_experts_per_token_bound; ++k) {
                    const uint32_t position = positions[local_token * num_experts_per_token + k];
                    if (position == INVALID_POSITION) continue;
                    const float weight = weights_ptr[token * weights_stride + k];

                    auto* buffer = reinterpret_cast<__nv_bfloat16*>(
                        recv_buffer + position * token_dim);
                    uint4 src_data = gdr::ld_global_uint4(
                        reinterpret_cast<uint4*>(buffer + j));
                    float src_fp32[VEC_SIZE];
                    bf16x8_to_fp32x8(src_data, src_fp32);

                    #pragma unroll
                    for (unsigned v = 0; v < VEC_SIZE; v++) {
                        acc[v] += src_fp32[v] * weight;
                    }
                }

                uint4 result = fp32x8_to_bf16x8(acc);
                gdr::st_global_uint4(reinterpret_cast<uint4*>(dst_ptr + j), result);
            }
        } else {
            // Fixed num_experts_per_token: unroll expert loop
            static constexpr size_t NUM_EXPERTS = NumExpertsPerTokenTy::Value;
            __nv_bfloat16* tokens_buf[NUM_EXPERTS];
            float weights_buf[NUM_EXPERTS];
            bool valid[NUM_EXPERTS];

            #pragma unroll(NUM_EXPERTS)
            for (unsigned k = 0; k < NUM_EXPERTS; ++k) {
                const uint32_t position = positions[local_token * num_experts_per_token + k];
                valid[k] = (position != INVALID_POSITION);
                tokens_buf[k] = valid[k] ? reinterpret_cast<__nv_bfloat16*>(
                    recv_buffer + position * token_dim) : nullptr;
                weights_buf[k] = weights_ptr[token * weights_stride + k];
            }

            for (unsigned j = threadIdx.x * VEC_SIZE; j < hidden_dim;
                 j += blockDim.x * VEC_SIZE) {
                float acc[VEC_SIZE];
                if (accumulate) {
                    uint4 existing = gdr::ld_global_uint4(
                        reinterpret_cast<uint4*>(dst_ptr + j));
                    bf16x8_to_fp32x8(existing, acc);
                } else {
                    #pragma unroll
                    for (unsigned v = 0; v < VEC_SIZE; v++) acc[v] = 0.0f;
                }

                // Load all expert sources first (better memory pipelining)
                uint4 srcs[NUM_EXPERTS];
                #pragma unroll(NUM_EXPERTS)
                for (unsigned k = 0; k < NUM_EXPERTS; ++k) {
                    if (valid[k]) {
                        srcs[k] = gdr::ld_global_uint4(
                            reinterpret_cast<uint4*>(tokens_buf[k] + j));
                    } else {
                        srcs[k] = make_uint4(0, 0, 0, 0);
                    }
                }

                // Accumulate
                #pragma unroll(NUM_EXPERTS)
                for (unsigned k = 0; k < NUM_EXPERTS; ++k) {
                    if (!valid[k]) continue;
                    float src_fp32[VEC_SIZE];
                    bf16x8_to_fp32x8(srcs[k], src_fp32);
                    #pragma unroll
                    for (unsigned v = 0; v < VEC_SIZE; v++) {
                        acc[v] += src_fp32[v] * weights_buf[k];
                    }
                }

                uint4 result = fp32x8_to_bf16x8(acc);
                gdr::st_global_uint4(reinterpret_cast<uint4*>(dst_ptr + j), result);
            }
        }
    }

    // ========================================================================
    // Signal combine_recv_done + NVLink barrier for next dispatch
    // ========================================================================
    grid.sync();

    if (blockIdx.x == 0) {
        if (warp_id == 0) {
            if (lane_id == 0) {
                gdr::st_mmio_b8(combine_recv_done, 1);
                *combine_recv_flag = 0;
                *sync_counter = counter + 1;
            }
        } else if (warp_id == 1 && NODE_SIZE > 1) {
            auto local_rank = rank % NODE_SIZE;
            if (lane_id < NODE_SIZE) {
                gdr::st_volatile_u32(&sync_ptrs[local_rank][lane_id], counter + 1);
            }
        }
    }
}

// ============================================================================
// Host-side launch function declaration
// ============================================================================
int a2a_combine_recv(
    size_t num_blocks,
    size_t hidden_dim,
    size_t x_elemsize,
    size_t num_experts,
    size_t num_experts_per_token,
    size_t rank,
    size_t node_size,
    size_t world_size,
    size_t num_tokens,
    const int32_t* bound_m_ptr,
    const int32_t* indices_ptr,
    size_t indices_stride,
    const float* weights_ptr,
    size_t weights_stride,
    uint8_t* out_tokens_ptr,
    size_t out_tokens_stride,
    bool accumulate,
    uint8_t* recv_buffer,
    uint32_t* token_offset,
    uint32_t* expert_offsets,
    uint8_t* combine_recv_flag,
    uint8_t* combine_recv_done,
    uint32_t* sync_counter,
    uint32_t** sync_ptrs,
    uint64_t stream);

}  // namespace coop
}  // namespace deep_ep
