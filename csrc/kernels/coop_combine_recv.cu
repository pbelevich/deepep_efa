// ============================================================================
// Cooperative Combine Recv — Host-side launch function
// ============================================================================

#include "coop_combine_recv.cuh"

#include <cassert>
#include <algorithm>

namespace deep_ep {
namespace coop {

template <typename T>
static constexpr T round_up_local(T value, T alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T>
static constexpr T ceil_div_local(T a, T b) {
    return (a + b - 1) / b;
}

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
    uint64_t stream
) {
    const size_t token_dim = round_up_local<size_t>(hidden_dim * x_elemsize, sizeof(int4));
    const size_t tokens_per_block = ceil_div_local<size_t>(num_tokens, num_blocks);

    // Convert accumulate bool to uint8_t for kernel parameter
    uint8_t accum_u8 = accumulate ? 1 : 0;

    void* args[] = {
        const_cast<size_t*>(&token_dim),
        &hidden_dim,
        &num_experts,
        &num_experts_per_token,
        &rank,
        &world_size,
        &num_tokens,
        &bound_m_ptr,
        &indices_ptr,
        &indices_stride,
        &weights_ptr,
        &weights_stride,
        &out_tokens_ptr,
        &out_tokens_stride,
        &accum_u8,
        &recv_buffer,
        &token_offset,
        &expert_offsets,
        &combine_recv_flag,
        &combine_recv_done,
        &sync_counter,
        &sync_ptrs,
    };

    constexpr size_t NUM_WARPS = 16;
    constexpr size_t NUM_THREADS = NUM_WARPS * 32;

    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(NUM_THREADS, 1, 1);

    const size_t shared_memory = tokens_per_block * num_experts_per_token * sizeof(uint32_t);

    cudaError_t status;

    // NOTE: pplx-garden dispatches on both in_dtype and out_dtype, but deepep's
    // combine always uses BF16 for both expert output and combined output.
    // We hardcode BF16→BF16 for now (matching deepep's existing combine path).
    COOP_LAUNCH_NUM_EXPERTS_PER_TOKEN(num_experts_per_token, {
        COOP_LAUNCH_WORLD_SIZE(node_size, {
            status = cudaLaunchCooperativeKernel(
                (void*)&a2a_combine_recv_kernel<NUM_WARPS, NODE_SIZE, NumExpertsPerTokenTy>,
                dimGrid, dimBlock, args, shared_memory, (cudaStream_t)stream
            );
        });
    });

    return status;
}

}  // namespace coop
}  // namespace deep_ep
