#pragma once

// ============================================================================
// Cooperative Kernel Launch Utilities
//
// Template dispatch infrastructure for cooperative kernels, matching
// pplx-garden's launch_utils.cuh. Provides:
//
// 1. Fixed<V> / NotFixed — compile-time vs runtime dimension wrappers
// 2. Dispatch macros — switch-case template specialization selectors
// 3. Cooperative launch helper
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace deep_ep {
namespace coop {

// ============================================================================
// Fixed<V>: Compile-time constant dimension.
// Enables the compiler to generate optimal code with known loop bounds,
// unrolled copies, and constant-folded address arithmetic.
// ============================================================================
template <size_t V>
struct Fixed {
    static constexpr size_t Value = V;
    __host__ __device__ constexpr operator size_t() const { return V; }
    // Allow construction from runtime values (ignored — the value is always V)
    __host__ __device__ constexpr Fixed() = default;
    __host__ __device__ constexpr explicit Fixed(size_t) {}
};

// ============================================================================
// NotFixed: Runtime dimension value.
// Fallback when the dimension isn't one of the compile-time specializations.
// ============================================================================
struct NotFixed {
    size_t value;
    __host__ __device__ constexpr operator size_t() const { return value; }
    __host__ __device__ explicit constexpr NotFixed(size_t v) : value(v) {}
};

// ============================================================================
// Helper to get value from either Fixed<V> or NotFixed
// ============================================================================
template <size_t V>
__host__ __device__ constexpr size_t get_dim(Fixed<V>) { return V; }
__host__ __device__ inline size_t get_dim(NotFixed nf) { return nf.value; }

// ============================================================================
// Cooperative kernel launch helper
// Launches a kernel with cudaLaunchCooperativeKernel using all SMs.
// ============================================================================
template <typename KernelFunc>
inline void launch_cooperative(KernelFunc kernel, int num_blocks, int threads_per_block,
                               size_t shared_mem, cudaStream_t stream, void** args) {
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)kernel, dim3(num_blocks), dim3(threads_per_block),
        args, shared_mem, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaLaunchCooperativeKernel failed: %s\n",
                cudaGetErrorString(err));
        throw std::runtime_error("Cooperative kernel launch failed");
    }
}

// Get number of SMs on current device
inline int get_num_sms() {
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    return num_sms;
}

// Get max blocks per SM for a cooperative kernel
template <typename KernelFunc>
inline int get_max_blocks_per_sm(KernelFunc kernel, int threads_per_block,
                                  size_t shared_mem = 0) {
    int max_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, kernel, threads_per_block, shared_mem);
    return max_blocks;
}

}  // namespace coop
}  // namespace deep_ep

// ============================================================================
// Template dispatch macros
// These generate switch-case statements that select Fixed<V> specializations
// at compile time based on runtime values.
// ============================================================================

// Helper: creates a case label that instantiates with Fixed<VAL>
#define _COOP_LAUNCH_TYPE(VAL) deep_ep::coop::Fixed<VAL>
#define _COOP_LAUNCH_VAL(VAL) deep_ep::coop::Fixed<VAL>()

// Token dim dispatch (dispatch direction: includes hidden + scale + metadata)
#define COOP_LAUNCH_TOKEN_DIM_DISPATCH(token_dim_bytes, BODY)        \
    do {                                                              \
        switch (token_dim_bytes) {                                    \
            case 2048: { using TokenDimTy = _COOP_LAUNCH_TYPE(2048); BODY; break; } \
            case 4096: { using TokenDimTy = _COOP_LAUNCH_TYPE(4096); BODY; break; } \
            case 7168: { using TokenDimTy = _COOP_LAUNCH_TYPE(7168); BODY; break; } \
            default: {                                                \
                using TokenDimTy = deep_ep::coop::NotFixed;           \
                BODY;                                                 \
                break;                                                \
            }                                                         \
        }                                                             \
    } while (0)

// Token dim dispatch (combine direction: just hidden * out_elemsize)
#define COOP_LAUNCH_TOKEN_DIM_COMBINE(token_dim_bytes, BODY)         \
    do {                                                              \
        switch (token_dim_bytes) {                                    \
            case 14336: { using TokenDimTy = _COOP_LAUNCH_TYPE(14336); BODY; break; } \
            default: {                                                \
                using TokenDimTy = deep_ep::coop::NotFixed;           \
                BODY;                                                 \
                break;                                                \
            }                                                         \
        }                                                             \
    } while (0)

// Hidden dim scale (element count, NOT bytes) dispatch
// Values: 8 = FP8 UE8M0 (hidden/128/4), 32 = FP8 block128 (hidden/128/2), 56 = FP8 (hidden/128)
#define COOP_LAUNCH_HIDDEN_DIM_SCALE(scale_elems, BODY)              \
    do {                                                              \
        switch (scale_elems) {                                        \
            case 8:  { using HiddenDimScaleTy = _COOP_LAUNCH_TYPE(8);  BODY; break; } \
            case 32: { using HiddenDimScaleTy = _COOP_LAUNCH_TYPE(32); BODY; break; } \
            case 56: { using HiddenDimScaleTy = _COOP_LAUNCH_TYPE(56); BODY; break; } \
            default: {                                                \
                using HiddenDimScaleTy = deep_ep::coop::NotFixed;     \
                BODY;                                                 \
                break;                                                \
            }                                                         \
        }                                                             \
    } while (0)

// Num experts per token dispatch
#define COOP_LAUNCH_NUM_EXPERTS_PER_TOKEN(n, BODY)                   \
    do {                                                              \
        switch (n) {                                                  \
            case 8: { using NumExpertsPerTokenTy = _COOP_LAUNCH_TYPE(8); BODY; break; } \
            default: {                                                \
                using NumExpertsPerTokenTy = deep_ep::coop::NotFixed;  \
                BODY;                                                 \
                break;                                                \
            }                                                         \
        }                                                             \
    } while (0)

// World size (node count) dispatch
#define COOP_LAUNCH_WORLD_SIZE(ws, BODY)                             \
    do {                                                              \
        switch (ws) {                                                 \
            case 1: { constexpr int NODE_SIZE = 1; BODY; break; }    \
            case 2: { constexpr int NODE_SIZE = 2; BODY; break; }    \
            case 4: { constexpr int NODE_SIZE = 4; BODY; break; }    \
            case 8: { constexpr int NODE_SIZE = 8; BODY; break; }    \
            default: {                                                \
                fprintf(stderr, "Unsupported node_size: %d\n", ws);  \
                throw std::runtime_error("Unsupported node_size");    \
            }                                                         \
        }                                                             \
    } while (0)

// DP size dispatch
#define COOP_LAUNCH_DP_SIZE(dp, BODY)                                \
    do {                                                              \
        switch (dp) {                                                 \
            case 1: { constexpr int DP_SIZE = 1; BODY; break; }      \
            case 2: { constexpr int DP_SIZE = 2; BODY; break; }      \
            case 4: { constexpr int DP_SIZE = 4; BODY; break; }      \
            case 8: { constexpr int DP_SIZE = 8; BODY; break; }      \
            default: {                                                \
                fprintf(stderr, "Unsupported dp_size: %d\n", dp);    \
                throw std::runtime_error("Unsupported dp_size");      \
            }                                                         \
        }                                                             \
    } while (0)

// QUICK mode dispatch (1 block per token vs multi-token per block)
#define COOP_LAUNCH_QUICK(num_blocks, num_tokens, BODY)              \
    do {                                                              \
        if ((num_blocks) >= (num_tokens)) {                          \
            constexpr bool QUICK = true;                              \
            BODY;                                                     \
        } else {                                                      \
            constexpr bool QUICK = false;                             \
            BODY;                                                     \
        }                                                             \
    } while (0)
