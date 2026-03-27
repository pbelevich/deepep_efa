// ============================================================================
// Cooperative Combine Send — Host-side launch function
// ============================================================================

#include "coop_combine_send.cuh"

#include <cassert>
#include <algorithm>

namespace deep_ep {
namespace coop {

template <typename T>
static constexpr T round_up_local(T value, T alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

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
    uint64_t stream
) {
    const size_t token_dim = round_up_local<size_t>(hidden_dim * x_elemsize, sizeof(int4));

    void* args[] = {
        const_cast<size_t*>(&token_dim),
        const_cast<size_t*>(&rank),
        &expert_x_ptr,
        &expert_x_stride,
        &tx_ready,
        &send_buffer,
        &recv_buffer,
        &source_rank,
        &combine_send_offset,
        &padded_index,
        &num_recv_tokens_ptr,
        &combine_send_done,
        &token_counter,
        &sync_counter,
        &sync_ptrs,
        &recv_ptrs,
    };

    dim3 dimGrid(num_blocks, 1, 1);

    cudaError_t status;
    COOP_LAUNCH_DP_SIZE(dp_size, {
        COOP_LAUNCH_WORLD_SIZE(node_size, {
            COOP_LAUNCH_TOKEN_DIM_COMBINE(token_dim, {
                constexpr size_t NUM_WARPS = 16;
                dim3 dimBlock(NUM_WARPS * 32, 1, 1);
                status = cudaLaunchCooperativeKernel(
                    (void*)&a2a_combine_send_kernel<NUM_WARPS, NODE_SIZE, DP_SIZE, TokenDimTy>,
                    dimGrid, dimBlock, args, 0, (cudaStream_t)stream
                );
            });
        });
    });

    return status;
}

}  // namespace coop
}  // namespace deep_ep
