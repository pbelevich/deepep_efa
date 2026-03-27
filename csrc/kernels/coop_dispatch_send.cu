// ============================================================================
// Cooperative Dispatch Send — Host-side launch function
// ============================================================================

#include "coop_dispatch_send.cuh"

#include <cassert>
#include <algorithm>

namespace deep_ep {
namespace coop {

// Round up to multiple of alignment
template <typename T>
static constexpr T round_up(T value, T alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Ceiling division
template <typename T>
static constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

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
    uint64_t stream
) {
    constexpr size_t NUM_WARPS = 16;
    constexpr size_t NUM_THREADS = NUM_WARPS * 32;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    // There should be enough threads for reductions
    assert(world_size <= NUM_THREADS);
    assert(num_experts <= NUM_THREADS);

    // Compute token layout dimensions matching pplx-garden
    const size_t token_dim = round_up<size_t>(hidden_dim * x_elemsize, sizeof(int4));
    const size_t token_scale_dim = round_up<size_t>(hidden_dim_scale * x_scale_elemsize, sizeof(int4));
    const size_t token_stride = token_dim + token_scale_dim + 16;
    assert(token_stride % sizeof(int4) == 0);

    void* args[] = {
        const_cast<size_t*>(&token_dim),
        const_cast<size_t*>(&token_scale_dim),
        const_cast<size_t*>(&token_stride),
        const_cast<size_t*>(&num_experts),
        const_cast<size_t*>(&num_experts_per_token),
        const_cast<size_t*>(&max_private_tokens),
        const_cast<size_t*>(&rank),
        const_cast<size_t*>(&dp_size),
        const_cast<size_t*>(&node_size),
        const_cast<size_t*>(&world_size),
        const_cast<size_t*>(&num_tokens),
        &bound_m_ptr,
        &x_ptr,
        &x_stride,
        &x_scale_ptr,
        &x_scale_stride_elem,
        &x_scale_stride_token,
        &indices,
        &indices_stride,
        &weights,
        &weights_stride,
        &token_offset,
        &num_routed,
        &expert_offsets,
        &dispatch_route_done,
        &dispatch_send_done,
        &tx_ready,
        &send_buffer,
        &grid_counter,
        &sync_counter,
        &sync_ptrs,
        &recv_ptrs,
        const_cast<size_t*>(&hidden_dim_scale),
    };

    const size_t shared_memory = std::max(num_experts, (size_t)NUM_WARPS) * sizeof(uint32_t);

    cudaError_t status;

    // Template dispatch: TokenDim → NumExpertsPerToken → HiddenDimScale → NODE_SIZE → QUICK
    COOP_LAUNCH_TOKEN_DIM_DISPATCH(token_dim, {
        COOP_LAUNCH_NUM_EXPERTS_PER_TOKEN(num_experts_per_token, {
            COOP_LAUNCH_HIDDEN_DIM_SCALE(hidden_dim_scale, {
                COOP_LAUNCH_WORLD_SIZE(node_size, {
                    if (num_blocks >= num_tokens) {
                        status = cudaLaunchCooperativeKernel(
                            (void*)&a2a_dispatch_send_kernel<
                                true,
                                NUM_WARPS,
                                NODE_SIZE,
                                TokenDimTy,
                                HiddenDimScaleTy,
                                NumExpertsPerTokenTy
                            >,
                            dimGrid, dimBlock, args, shared_memory,
                            (cudaStream_t)stream
                        );
                    } else {
                        status = cudaLaunchCooperativeKernel(
                            (void*)&a2a_dispatch_send_kernel<
                                false,
                                NUM_WARPS,
                                NODE_SIZE,
                                TokenDimTy,
                                HiddenDimScaleTy,
                                NumExpertsPerTokenTy
                            >,
                            dimGrid, dimBlock, args, shared_memory,
                            (cudaStream_t)stream
                        );
                    }
                });
            });
        });
    });

    return status;
}

}  // namespace coop
}  // namespace deep_ep
