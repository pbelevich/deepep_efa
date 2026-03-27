#include <cstring>
#include <vector>

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace deep_ep {

namespace intranode {

template <int kNumRanks>
__global__ void barrier(int** barrier_signal_ptrs, int rank) {
    barrier_block<kNumRanks>(barrier_signal_ptrs, rank);
}

void barrier(int** barrier_signal_ptrs, int rank, int num_ranks, cudaStream_t stream) {
#define BARRIER_LAUNCH_CASE(ranks)                                  \
    LAUNCH_KERNEL(&cfg, barrier<ranks>, barrier_signal_ptrs, rank); \
    break

    SETUP_LAUNCH_CONFIG(1, 32, stream);
    SWITCH_RANKS(BARRIER_LAUNCH_CASE);
#undef BARRIER_LAUNCH_CASE
}

}  // namespace intranode

// In EFA mode, internode_efa.cu provides: get_source_meta_bytes, notify_dispatch,
// dispatch, cached_notify, combine. But we still need the runtime management stubs
// (get_unique_id, init, alloc, free, barrier, finalize) which are declared in api.cuh.
// In non-EFA mode (DISABLE_NVSHMEM only), provide crash stubs for everything.
namespace internode {

#ifdef ENABLE_EFA

// EFA mode: provide runtime management stubs only.
// The internode kernel functions (notify_dispatch, dispatch, etc.) are in internode_efa.cu.

std::vector<uint8_t> get_unique_id() {
    // In EFA mode, we don't use NVSHMEM unique IDs.
    // Return a dummy - the EFA address exchange happens separately.
    return std::vector<uint8_t>(128, 0);
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    // EFA initialization is handled by EfaWorkerManager
    return rank;
}

void* alloc(size_t size, size_t alignment) {
    // In EFA mode, RDMA buffer is allocated via cudaMalloc
    void* ptr = nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void free(void* ptr) {
    cudaFree(ptr);
}

void barrier() {
    // In EFA mode, barrier is handled by EfaWorkerManager::barrier()
}

void finalize() {
    // EFA cleanup is handled by EfaWorkerManager destructor
}

#else  // !ENABLE_EFA (DISABLE_NVSHMEM only, intranode-only mode)

// Stubs when both NVSHMEM and EFA are disabled (intranode-only mode)
std::vector<uint8_t> get_unique_id() {
    EP_HOST_ASSERT(false && "Internode not available without NVSHMEM or EFA");
    return {};
}

int init(const std::vector<uint8_t>& root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
    EP_HOST_ASSERT(false && "Internode not available without NVSHMEM or EFA");
    return -1;
}

void* alloc(size_t size, size_t alignment) {
    EP_HOST_ASSERT(false && "Internode not available without NVSHMEM or EFA");
    return nullptr;
}

void free(void* ptr) {
    EP_HOST_ASSERT(false && "Internode not available without NVSHMEM or EFA");
}

void barrier() {
    EP_HOST_ASSERT(false && "Internode not available without NVSHMEM or EFA");
}

void finalize() {
    EP_HOST_ASSERT(false && "Internode not available without NVSHMEM or EFA");
}

int get_source_meta_bytes() {
    return 2 * sizeof(int);
}

#endif  // !ENABLE_EFA

}  // namespace internode

// Internode low-latency stubs
// The original internode_ll.cu was removed (NVSHMEM-dependent).
// For EFA mode, these functions are not yet implemented.
// They will crash at runtime if called, which is acceptable for Iter 1.
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0,
                              int num_clean_int_0,
                              int* clean_1,
                              int num_clean_int_1,
                              int rank,
                              int num_ranks,
                              int* mask_buffer,
                              int* sync_buffer,
                              cudaStream_t stream) {
    EP_HOST_ASSERT(false && "Low-latency mode not available in EFA mode (Iter 1)");
}

void dispatch(void* packed_recv_x,
              void* packed_recv_x_scales,
              int* packed_recv_src_info,
              int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              int* mask_buffer,
              int* cumulative_local_expert_recv_stats,
              int64_t* dispatch_wait_recv_cost_stats,
              void* rdma_recv_x,
              int* rdma_recv_count,
              void* rdma_x,
              const void* x,
              const topk_idx_t* topk_idx,
              int* next_clean,
              int num_next_clean_int,
              int num_tokens,
              int hidden,
              int num_max_dispatch_tokens_per_rank,
              int num_topk,
              int num_experts,
              int rank,
              int num_ranks,
              bool use_fp8,
              bool round_scale,
              bool use_ue8m0,
              void* workspace,
              int num_device_sms,
              cudaStream_t stream,
              int phases) {
    EP_HOST_ASSERT(false && "Low-latency mode not available in EFA mode (Iter 1)");
}

void combine(void* combined_x,
             void* rdma_recv_x,
             int* rdma_recv_flag,
             void* rdma_send_x,
             const void* x,
             const topk_idx_t* topk_idx,
             const float* topk_weights,
             const int* src_info,
             const int64_t* layout_range,
             int* mask_buffer,
             int64_t* combine_wait_recv_cost_stats,
             int* next_clean,
             int num_next_clean_int,
             int num_combined_tokens,
             int hidden,
             int num_max_dispatch_tokens_per_rank,
             int num_topk,
             int num_experts,
             int rank,
             int num_ranks,
             bool use_logfmt,
             void* workspace,
             int num_device_sms,
             cudaStream_t stream,
             int phases,
             bool zero_copy) {
    EP_HOST_ASSERT(false && "Low-latency mode not available in EFA mode (Iter 1)");
}

void query_mask_buffer(int* mask_buffer_ptr, int num_ranks, int* output_mask_tensor, cudaStream_t stream) {
    EP_HOST_ASSERT(false && "Low-latency mode not available in EFA mode (Iter 1)");
}

void update_mask_buffer(int* mask_buffer_ptr, int rank_to_mask, bool mask, cudaStream_t stream) {
    EP_HOST_ASSERT(false && "Low-latency mode not available in EFA mode (Iter 1)");
}

void clean_mask_buffer(int* mask_buffer_ptr, int num_ranks, cudaStream_t stream) {
    EP_HOST_ASSERT(false && "Low-latency mode not available in EFA mode (Iter 1)");
}

}  // namespace internode_ll

}  // namespace deep_ep
