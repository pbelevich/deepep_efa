#pragma once

// ============================================================================
// Device Utilities for Cooperative Kernels
//
// Warp and block-level primitives matching pplx-garden's device_utils.cuh.
// ============================================================================

#include <cstdint>
#include "../transport/gdr_flags.cuh"

namespace deep_ep {
namespace coop {

static constexpr int WARP_SIZE = 32;

// ============================================================================
// Warp-level reductions
// ============================================================================

// Full warp sum using butterfly reduction
template <typename T>
__forceinline__ __device__ T warp_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Full warp AND using butterfly reduction
__forceinline__ __device__ uint32_t warp_and(uint32_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val &= __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp reduction max using shuffle-down
template <typename T>
__forceinline__ __device__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

// Warp reduction sum using shuffle-down
template <typename T>
__forceinline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Block-level reductions (via shared memory)
// ============================================================================

// Block reduce max — requires shared memory of size [num_warps]
template <typename T, int NUM_WARPS>
__forceinline__ __device__ T block_reduce_max(T val, T* shared_buf) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) shared_buf[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (lane < NUM_WARPS) ? shared_buf[lane] : T(0);
        val = warp_reduce_max(val);
    }
    __syncthreads();
    return val;
}

// Block reduce sum — requires shared memory of size [num_warps]
template <typename T, int NUM_WARPS>
__forceinline__ __device__ T block_reduce_sum(T val, T* shared_buf) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared_buf[warp] = val;
    __syncthreads();

    if (warp == 0) {
        val = (lane < NUM_WARPS) ? shared_buf[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}

// ============================================================================
// CDF (prefix sum) construction — tiled across warps
//
// Each thread computes a value, and the result is a CDF in shared memory.
// Used in dispatch_send for computing expert_offsets from token counts.
// ============================================================================
template <int NUM_WARPS>
__forceinline__ __device__ void build_cdf_tiled(
    uint32_t* shared_counts,  // [num_elements] — input counts
    uint32_t* shared_offsets, // [num_elements] — output prefix sums
    int num_elements)
{
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp = tid / WARP_SIZE;

    // Phase 1: Intra-warp prefix sum
    // Each warp handles a tile of elements
    int elements_per_warp = (num_elements + NUM_WARPS - 1) / NUM_WARPS;
    int start = warp * elements_per_warp;
    int end = min(start + elements_per_warp, num_elements);

    uint32_t val = 0;
    if (start + lane < end) {
        val = shared_counts[start + lane];
    }

    // Inclusive prefix sum within warp
    uint32_t prefix = val;
    #pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        uint32_t n = __shfl_up_sync(0xffffffff, prefix, d);
        if (lane >= d) prefix += n;
    }

    // Write partial results
    if (start + lane < end) {
        shared_offsets[start + lane] = prefix;
    }
    __syncthreads();

    // Phase 2: Cross-warp fixup
    // Warp 0 computes the total for each warp's tile and adds it to subsequent tiles
    if (warp == 0 && lane < NUM_WARPS) {
        uint32_t warp_total = 0;
        int warp_end = min((int)(lane + 1) * elements_per_warp, num_elements) - 1;
        if (warp_end >= 0 && warp_end < num_elements) {
            warp_total = shared_offsets[warp_end];
        }

        // Exclusive prefix sum of warp totals
        uint32_t warp_prefix = warp_total;
        #pragma unroll
        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            uint32_t n = __shfl_up_sync(0xffffffff, warp_prefix, d);
            if (lane >= d) warp_prefix += n;
        }
        // Make exclusive by shifting
        warp_prefix = __shfl_up_sync(0xffffffff, warp_prefix, 1);
        if (lane == 0) warp_prefix = 0;

        // Store fixup values
        shared_counts[lane] = warp_prefix;  // Reuse shared_counts as temp
    }
    __syncthreads();

    // Phase 3: Apply fixup — add warp's prefix total to all elements in its tile
    if (warp > 0 && start + lane < end) {
        shared_offsets[start + lane] += shared_counts[warp];
    }
    __syncthreads();

    // Convert inclusive prefix sum to exclusive by shifting
    // shared_offsets[i] = sum of shared_counts[0..i] (inclusive)
    // We want shared_offsets[i] = sum of shared_counts[0..i-1] (exclusive)
    if (tid == 0) {
        uint32_t prev = 0;
        for (int i = 0; i < num_elements; i++) {
            uint32_t cur = shared_offsets[i];
            shared_offsets[i] = prev;
            prev = cur;
        }
    }
    __syncthreads();
}

// ============================================================================
// Binary search on CDF array
// Returns the largest index i such that cdf[i] <= target
// ============================================================================
__forceinline__ __device__ int binary_search_cdf(
    const uint32_t* cdf, int n, uint32_t target)
{
    int lo = 0, hi = n - 1;
    int result = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cdf[mid] <= target) {
            result = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return result;
}

}  // namespace coop
}  // namespace deep_ep
