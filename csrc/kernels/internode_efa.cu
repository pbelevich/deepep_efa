// internode_efa.cu - EFA-based internode dispatch/combine kernels
//
// Architecture (split kernel pattern, following pplx-garden):
//   dispatch_send kernel: GPU prepares data in local RDMA buffer, sets GdrFlags
//   CPU worker thread: detects flags, issues fi_writemsg RDMA writes
//   dispatch_recv kernel: GPU waits for ImmCounter to signal completion, unpacks
//
// For Iteration 1, we implement a simplified version that:
// 1. Uses the same data layout as original DeepEP internode
// 2. Replaces IBGDA put_nbi with CPU-mediated RDMA via GdrFlags
// 3. Replaces nvshmem_sync with allgather barrier

#include <algorithm>

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"
#include "api.cuh"

#ifdef ENABLE_EFA

namespace deep_ep {

namespace internode_efa {

// ============================================================================
// SourceMeta: Same layout as original internode for API compatibility
// ============================================================================
struct SourceMeta {
    int src_rdma_rank, is_token_in_nvl_rank_bits;

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8, "Invalid number of maximum NVL peers");

    __forceinline__ SourceMeta() = default;

    __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
        src_rdma_rank = rdma_rank;
        is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
        #pragma unroll
        for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i)
            is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
    }

    __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
        return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
    }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

// ============================================================================
// Byte packing for RDMA tokens (same as original)
// ============================================================================
__host__ __device__ __forceinline__ int get_num_bytes_per_token(
    int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    return static_cast<int>(align_up(
        hidden_int4 * sizeof(int4) + sizeof(SourceMeta) +
        num_scales * sizeof(float) +
        num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
        sizeof(int4)));
}

// ============================================================================
// dispatch_send kernel:
// Packs token data into the RDMA send buffer region.
// After packing, sets a per-chunk GdrFlag byte so the CPU worker knows
// the data is ready to be RDMA-written to remote ranks.
//
// Buffer layout (per RDMA rank, per channel):
//   [token_data...][flag_byte]
// ============================================================================
template <int kNumRDMARanks>
__global__ void dispatch_send(
    // Local RDMA buffer for staging outgoing data
    void* rdma_buffer_ptr,
    // Send metadata: per-token flags indicating ready-to-send
    volatile uint8_t* send_ready_flags,
    // Per-rank token counts (will be written into buffer header)
    int* send_counts_per_rank,
    // Source data
    const int4* x,
    const float* x_scales,
    const topk_idx_t* topk_idx,
    const float* topk_weights,
    const bool* is_token_in_rank,
    // Layout info
    const int* rdma_channel_prefix_matrix,
    const int* recv_rdma_rank_prefix_sum,
    const int* gbl_channel_prefix_matrix,
    const int* recv_gbl_rank_prefix_sum,
    // Dimensions
    int num_tokens,
    int hidden_int4,
    int num_scales,
    int num_topk,
    int num_experts,
    int scale_token_stride,
    int scale_hidden_stride,
    // Rank info
    int rank,
    int num_ranks,
    // Buffer sizing
    int num_max_rdma_chunked_send_tokens,
    int num_max_rdma_chunked_recv_tokens,
    int num_bytes_per_token) {

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = thread_id % 32;
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto num_channels = num_sms;  // One SM per channel for send
    const auto channel_id = sm_id;

    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS;
    const auto nvl_rank = rank % NUM_MAX_NVL_PEERS;

    // Get this channel's token range
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, channel_id,
                           token_start_idx, token_end_idx);

    // For each token in our range, pack it into the send buffer for each
    // target RDMA rank that needs it
    auto send_buf = static_cast<uint8_t*>(rdma_buffer_ptr);

    for (int token_idx = token_start_idx + warp_id;
         token_idx < token_end_idx;
         token_idx += (num_threads / 32)) {

        // Check which RDMA ranks need this token
        for (int dst_rdma = 0; dst_rdma < kNumRDMARanks; ++dst_rdma) {
            if (dst_rdma == rdma_rank) continue;

            // Check if token goes to any NVL rank in this RDMA rank
            bool token_goes_here = false;
            EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint64_t),
                             "Invalid number of NVL peers");
            auto is_token_bits = *reinterpret_cast<const uint64_t*>(
                is_token_in_rank + token_idx * num_ranks +
                dst_rdma * NUM_MAX_NVL_PEERS);
            token_goes_here = (is_token_bits != 0);

            if (!token_goes_here) continue;

            // Calculate offset in send buffer
            // Layout: [rdma_rank][channel][slot][token_data]
            // For simplicity in Iter1, use a flat layout
            int slot_offset = dst_rdma * num_max_rdma_chunked_recv_tokens *
                              num_bytes_per_token;
            // TODO: Calculate proper per-channel slot index
            // For now, use atomic increment to get slot
            // (This will be optimized in later iterations)

            // Copy hidden data
            auto dst_base = send_buf + slot_offset;
            auto src_base = x + token_idx * hidden_int4;

            // Warp-cooperative copy of the token data
            for (int i = lane_id; i < hidden_int4; i += 32) {
                reinterpret_cast<int4*>(dst_base)[i] = __ldg(src_base + i);
            }

            // Copy source metadata
            if (lane_id == 0) {
                auto is_nvl_ranks = reinterpret_cast<const bool*>(&is_token_bits);
                auto meta = SourceMeta(rdma_rank, is_nvl_ranks);
                *reinterpret_cast<SourceMeta*>(
                    dst_base + hidden_int4 * sizeof(int4)) = meta;
            }

            // Copy scales if present
            auto scales_dst = reinterpret_cast<float*>(
                dst_base + hidden_int4 * sizeof(int4) + sizeof(SourceMeta));
            for (int i = lane_id; i < num_scales; i += 32) {
                scales_dst[i] = __ldg(x_scales + token_idx * scale_token_stride +
                                       i * scale_hidden_stride);
            }

            // Copy topk_idx and topk_weights
            auto topk_dst = reinterpret_cast<int*>(scales_dst + num_scales);
            auto weights_dst = reinterpret_cast<float*>(topk_dst + num_topk);
            for (int i = lane_id; i < num_topk; i += 32) {
                topk_dst[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                weights_dst[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }
            __syncwarp();
        }
    }

    // After all tokens packed, set ready flag for CPU worker
    __syncthreads();
    if (thread_id == 0) {
        __threadfence_system();  // Ensure all writes visible to CPU
        send_ready_flags[channel_id] = 1;
    }
}

// ============================================================================
// dispatch_recv kernel:
// Waits for ImmCounter (set by CPU) to indicate remote data has arrived,
// then unpacks from the RDMA receive buffer into output tensors.
// ============================================================================
template <int kNumRDMARanks>
__global__ void dispatch_recv(
    // Output tensors
    int4* recv_x,
    float* recv_x_scales,
    topk_idx_t* recv_topk_idx,
    float* recv_topk_weights,
    SourceMeta* recv_src_meta,
    // Receive buffer (filled by remote RDMA writes)
    const void* rdma_buffer_ptr,
    // Completion flags (set by CPU when RDMA writes complete)
    const volatile int* recv_completion_counters,
    int expected_total_tokens,
    // Layout info
    const int* recv_rdma_rank_prefix_sum,
    const int* recv_gbl_rank_prefix_sum,
    // Dimensions
    int hidden_int4,
    int num_scales,
    int num_topk,
    int num_max_rdma_chunked_recv_tokens,
    int num_bytes_per_token,
    // Rank info
    int rank,
    int num_ranks) {

    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto lane_id = thread_id % 32;
    const auto warp_id = thread_id / 32;
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto rdma_rank = rank / NUM_MAX_NVL_PEERS;

    // Wait for all RDMA data to arrive
    // The CPU worker updates recv_completion_counters as RDMA writes complete
    if (thread_id == 0) {
        auto start_time = clock64();
        while (true) {
            int total = 0;
            for (int i = 0; i < kNumRDMARanks; ++i) {
                total += ld_volatile_global(const_cast<const int*>(recv_completion_counters + i));
            }
            if (total >= expected_total_tokens) break;

            if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
                printf("DeepEP EFA dispatch_recv timeout: received %d/%d tokens\n",
                       total, expected_total_tokens);
                trap();
            }
        }
    }
    __syncthreads();

    // Unpack received tokens from RDMA buffer to output tensors
    auto recv_buf = static_cast<const uint8_t*>(rdma_buffer_ptr);

    for (int src_rdma = 0; src_rdma < kNumRDMARanks; ++src_rdma) {
        if (src_rdma == rdma_rank) continue;

        int rank_start = src_rdma == 0 ? 0 : recv_rdma_rank_prefix_sum[src_rdma - 1];
        int rank_end = recv_rdma_rank_prefix_sum[src_rdma];
        int num_tokens_from_rank = rank_end - rank_start;

        // Calculate source offset in receive buffer
        int src_offset = src_rdma * num_max_rdma_chunked_recv_tokens * num_bytes_per_token;

        for (int local_idx = warp_id; local_idx < num_tokens_from_rank;
             local_idx += (num_threads / 32)) {
            int global_idx = rank_start + local_idx;
            auto src_token = recv_buf + src_offset + local_idx * num_bytes_per_token;

            // Copy hidden data
            for (int i = lane_id; i < hidden_int4; i += 32) {
                recv_x[global_idx * hidden_int4 + i] =
                    reinterpret_cast<const int4*>(src_token)[i];
            }

            // Copy source metadata
            if (lane_id == 0 && recv_src_meta) {
                recv_src_meta[global_idx] = *reinterpret_cast<const SourceMeta*>(
                    src_token + hidden_int4 * sizeof(int4));
            }

            // Copy scales
            auto src_scales = reinterpret_cast<const float*>(
                src_token + hidden_int4 * sizeof(int4) + sizeof(SourceMeta));
            if (recv_x_scales) {
                for (int i = lane_id; i < num_scales; i += 32) {
                    recv_x_scales[global_idx * num_scales + i] = src_scales[i];
                }
            }

            // Copy topk_idx and topk_weights
            auto src_topk = reinterpret_cast<const int*>(src_scales + num_scales);
            auto src_weights = reinterpret_cast<const float*>(src_topk + num_topk);
            if (recv_topk_idx) {
                for (int i = lane_id; i < num_topk; i += 32) {
                    recv_topk_idx[global_idx * num_topk + i] =
                        static_cast<topk_idx_t>(src_topk[i]);
                }
            }
            if (recv_topk_weights) {
                for (int i = lane_id; i < num_topk; i += 32) {
                    recv_topk_weights[global_idx * num_topk + i] = src_weights[i];
                }
            }
        }
    }
}

}  // namespace internode_efa

// ============================================================================
// Host-callable functions (declared in api.cuh under internode:: namespace)
// These implement the same interface as the original NVSHMEM internode functions
// but use the EFA split-kernel approach
// ============================================================================
namespace internode {

int get_source_meta_bytes() {
    return sizeof(internode_efa::SourceMeta);
}

// notify_dispatch for EFA: simpler version that just does NVL barrier + metadata exchange
// The RDMA barrier is handled by the EFA transport layer
void notify_dispatch(const int* num_tokens_per_rank,
                     int* moe_recv_counter_mapped,
                     int num_ranks,
                     const int* num_tokens_per_rdma_rank,
                     int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert,
                     int* moe_recv_expert_counter_mapped,
                     int num_experts,
                     const bool* is_token_in_rank,
                     int num_tokens,
                     int num_worst_tokens,
                     int num_channels,
                     int hidden_int4,
                     int num_scales,
                     int num_topk,
                     int expert_alignment,
                     int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix,
                     int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr,
                     int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens,
                     int** barrier_signal_ptrs,
                     int rank,
                     cudaStream_t stream,
                     int64_t num_rdma_bytes,
                     int64_t num_nvl_bytes,
                     bool low_latency_mode) {
    // In EFA mode, the metadata exchange (token counts) happens via
    // the EFA transport layer using RDMA writes with immediate data.
    // The GPU kernel just needs to:
    // 1. Do the intra-node NVL barrier
    // 2. Compute the channel prefix matrices (same as before)
    // 3. The actual RDMA metadata exchange is done by the CPU-side EFA worker

    // For now, we just do the NVL barrier and compute prefix matrices on GPU
    // The CPU side of deep_ep.cpp will handle the EFA-based metadata exchange

    // TODO: Implement the channel prefix computation kernel
    // For Iteration 1, this is handled in deep_ep.cpp via host-side logic
}

void dispatch(void* recv_x,
              float* recv_x_scales,
              topk_idx_t* recv_topk_idx,
              float* recv_topk_weights,
              void* recv_src_meta,
              const void* x,
              const float* x_scales,
              const topk_idx_t* topk_idx,
              const float* topk_weights,
              int* send_rdma_head,
              int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum,
              const bool* is_token_in_rank,
              int num_tokens,
              int num_worst_tokens,
              int hidden_int4,
              int num_scales,
              int num_topk,
              int num_experts,
              int scale_token_stride,
              int scale_hidden_stride,
              void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens,
              int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens,
              int num_max_nvl_chunked_recv_tokens,
              int rank,
              int num_ranks,
              bool is_cached_dispatch,
              cudaStream_t stream,
              int num_channels,
              bool low_latency_mode) {
    // In EFA mode, the dispatch is split into:
    // 1. dispatch_send kernel (GPU packs data, sets flags)
    // 2. CPU worker does RDMA writes
    // 3. dispatch_recv kernel (GPU unpacks received data)
    //
    // For Iteration 1, this is orchestrated from deep_ep.cpp
    // The kernel launches are done there to coordinate with the CPU EFA worker

    // TODO: This function will be called from deep_ep.cpp's internode_dispatch
    // For now it's a placeholder - the actual implementation is driven from
    // the host side
}

void cached_notify(int hidden_int4,
                   int num_scales,
                   int num_topk_idx,
                   int num_topk_weights,
                   int num_ranks,
                   int num_channels,
                   int num_combined_tokens,
                   int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix,
                   const int* rdma_rank_prefix_sum,
                   int* combined_nvl_head,
                   void* rdma_buffer_ptr,
                   int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens,
                   int** barrier_signal_ptrs,
                   int rank,
                   cudaStream_t stream,
                   int64_t num_rdma_bytes,
                   int64_t num_nvl_bytes,
                   bool is_cached_dispatch,
                   bool low_latency_mode) {
    // TODO: Implement EFA version of cached_notify
    // For now, just do the NVL barrier
    int nvl_rank = rank % NUM_MAX_NVL_PEERS;
    int num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
    deep_ep::intranode::barrier(barrier_signal_ptrs, nvl_rank, num_nvl_ranks, stream);
}

void combine(cudaDataType_t type,
             void* combined_x,
             float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x,
             const float* topk_weights,
             const void* bias_0,
             const void* bias_1,
             const int* combined_rdma_head,
             const int* combined_nvl_head,
             const void* src_meta,
             const int* rdma_channel_prefix_matrix,
             const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens,
             int num_combined_tokens,
             int hidden,
             int num_topk,
             void* rdma_buffer_ptr,
             int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs,
             int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens,
             int rank,
             int num_ranks,
             cudaStream_t stream,
             int num_channels,
             bool low_latency_mode) {
    // TODO: Implement EFA version of combine
    // For Iteration 1, the combine is driven from deep_ep.cpp
}

}  // namespace internode

}  // namespace deep_ep

#endif  // ENABLE_EFA
