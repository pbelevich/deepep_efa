// ============================================================================
// Cooperative Dispatch Recv — Host-side launch function
// ============================================================================

#include "coop_dispatch_recv.cuh"

#include <cassert>
#include <algorithm>

namespace deep_ep {
namespace coop {

template <typename T>
static constexpr T round_up_local(T value, T alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

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
    uint64_t stream
) {
    constexpr size_t NUM_WARPS = 16;
    constexpr size_t NUM_THREADS = NUM_WARPS * 32;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    const size_t token_dim = round_up_local<size_t>(hidden_dim * x_elemsize, sizeof(float4));
    const size_t token_scale_dim = round_up_local<size_t>(hidden_dim_scale * x_scale_elemsize, sizeof(float4));
    const size_t token_stride = token_dim + token_scale_dim + 16;
    assert(token_stride % sizeof(float4) == 0);

    void* args[] = {
        const_cast<size_t*>(&token_dim),
        const_cast<size_t*>(&token_scale_dim),
        const_cast<size_t*>(&token_stride),
        &hidden_dim,
        &hidden_dim_scale,
        &x_elemsize,
        &x_scale_elemsize,
        &num_experts,
        &rank,
        &world_size,
        &out_num_tokens_ptr,
        &out_x_ptr,
        &out_x_stride,
        &out_x_scale_ptr,
        &out_x_scale_stride_elem,
        &out_x_scale_stride_token,
        &tokens_per_expert,
        &send_buffer,
        &recv_buffer,
        &source_rank,
        &source_offset,
        &padded_index,
        &num_routed,
        &num_recv_tokens_ptr,
        &num_recv_tokens_flag,
        &dispatch_recv_flag,
        &dispatch_recv_done,
        &grid_counter,
        &sync_counter,
        &sync_ptrs,
        &send_ptrs,
    };

    cudaError_t status;
    COOP_LAUNCH_WORLD_SIZE(node_size, {
        COOP_LAUNCH_TOKEN_DIM_DISPATCH(token_dim, {
            COOP_LAUNCH_HIDDEN_DIM_SCALE(hidden_dim_scale, {
                status = cudaLaunchCooperativeKernel(
                    (void*)&a2a_dispatch_recv_kernel<NUM_WARPS, NODE_SIZE, TokenDimTy, HiddenDimScaleTy>,
                    dimGrid, dimBlock, args, 0, (cudaStream_t)stream
                );
            });
        });
    });

    return status;
}

}  // namespace coop
}  // namespace deep_ep
