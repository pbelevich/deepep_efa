#pragma once

// ============================================================================
// Memory Intrinsics for Cooperative Kernels
//
// Complete set of PTX memory operations matching pplx-garden's memory.cuh.
// Organized by use case:
//
// 1. GDR Flag signaling (GPU <-> CPU via MMIO):
//    st_mmio_b8, ld_mmio_b8
//
// 2. NVLink synchronization (GPU <-> GPU volatile/release/acquire):
//    st_volatile_u32, ld_volatile_u32, ld_acquire_u32,
//    st_release_u32, st_relaxed_u32
//
// 3. Atomic operations:
//    add_release_gpu_u32, add_release_sys_u32
//
// 4. Fence instructions:
//    fence_acq_rel_gpu, fence_acquire_gpu, fence_release_gpu,
//    fence_acquire_system, fence_release_system
//
// 5. Bulk data movement (global memory):
//    ld_global_uint4, st_global_uint4 (regular)
//    ld_global_nc_uint4, st_global_nc_uint4 (non-coherent, bypass L1)
//
// 6. Shared memory:
//    ld_shared_uint4, st_shared_uint4
//
// 7. Warp-level:
//    elect_one_sync
// ============================================================================

#include <cstdint>

namespace deep_ep {
namespace gdr {

// ============================================================================
// 1. GDR Flag Signaling (GPU <-> CPU via PCIe BAR MMIO)
// ============================================================================

// GPU -> CPU signal: write a byte to GDRCopy-mapped memory
__forceinline__ __device__ void st_mmio_b8(uint8_t* flag_addr, uint8_t value) {
    uint32_t tmp = static_cast<uint32_t>(value);
    asm volatile(
        "{ st.mmio.relaxed.sys.global.b8 [%1], %0; }"
        : : "r"(tmp), "l"(flag_addr) :
    );
}

// CPU -> GPU signal: read a byte from GDRCopy-mapped memory
__forceinline__ __device__ uint8_t ld_mmio_b8(uint8_t* flag_addr) {
    uint32_t tmp;
    asm volatile(
        "{ ld.mmio.relaxed.sys.global.b8 %0, [%1]; }"
        : "=r"(tmp) : "l"(flag_addr) :
    );
    return static_cast<uint8_t>(tmp);
}

// ============================================================================
// 2. NVLink Synchronization (GPU <-> GPU)
//    Used for sync_ptrs barrier between kernels on same node.
// ============================================================================

// Volatile store — immediately visible to all SMs on this GPU
__forceinline__ __device__ void st_volatile_u32(uint32_t* flag_addr, uint32_t flag) {
    asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
}

// Volatile load — bypasses L1/L2 caches
__forceinline__ __device__ uint32_t ld_volatile_u32(uint32_t* flag_addr) {
    uint32_t flag;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
    return flag;
}

// Acquire load — ensures subsequent reads see data written before the release store
__forceinline__ __device__ uint32_t ld_acquire_u32(uint32_t* flag_addr) {
    uint32_t flag;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
    return flag;
}

// Release store — ensures all prior stores are visible before this write
__forceinline__ __device__ void st_release_u32(uint32_t* flag_addr, uint32_t flag) {
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
}

// Relaxed store — no ordering guarantees (use with explicit fences)
__forceinline__ __device__ void st_relaxed_u32(uint32_t* flag_addr, uint32_t flag) {
    asm volatile("st.relaxed.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
}

// ============================================================================
// 3. Atomic Operations
// ============================================================================

// GPU-scoped atomic add with release ordering
// Used for cross-block coordination within cooperative kernels.
__forceinline__ __device__ uint32_t add_release_gpu_u32(uint32_t* addr, uint32_t val) {
    uint32_t old;
    asm volatile(
        "atom.release.gpu.global.add.u32 %0, [%1], %2;"
        : "=r"(old) : "l"(addr), "r"(val) : "memory"
    );
    return old;
}

// System-scoped atomic add with release ordering
// Used for cross-GPU atomic coordination (NVLink).
__forceinline__ __device__ uint32_t add_release_sys_u32(uint32_t* addr, uint32_t val) {
    uint32_t old;
    asm volatile(
        "atom.release.sys.global.add.u32 %0, [%1], %2;"
        : "=r"(old) : "l"(addr), "r"(val)
    );
    return old;
}

// ============================================================================
// 4. Fence Instructions
// ============================================================================

// GPU-scope acquire-release fence
__forceinline__ __device__ void fence_acq_rel_gpu() {
    asm volatile("{ fence.acq_rel.gpu; }" ::: "memory");
}

// GPU-scope acquire fence
__forceinline__ __device__ void fence_acquire_gpu() {
    asm volatile("{ fence.acquire.gpu; }" ::: "memory");
}

// GPU-scope release fence
__forceinline__ __device__ void fence_release_gpu() {
    asm volatile("{ fence.release.gpu; }" ::: "memory");
}

// System-scope acquire fence
__forceinline__ __device__ void fence_acquire_system() {
    asm volatile("{ fence.acquire.sys; }" ::: "memory");
}

// System-scope release fence
__forceinline__ __device__ void fence_release_system() {
    asm volatile("{ fence.release.sys; }" ::: "memory");
}

// ============================================================================
// 5. Bulk Data Movement (Global Memory)
// ============================================================================

// Regular global uint4 load
__forceinline__ __device__ uint4 ld_global_uint4(const void* ptr) {
    uint4 v;
    asm volatile(
        "{ ld.global.v4.u32 {%0, %1, %2, %3}, [%4]; }"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr)
    );
    return v;
}

// Regular global uint4 store
__forceinline__ __device__ void st_global_uint4(void* ptr, uint4 v) {
    asm volatile(
        "{ st.global.v4.u32 [%0], {%1, %2, %3, %4}; }"
        : : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
    );
}

// Non-coherent uint4 load (bypass L1, L2 streaming)
// CRITICAL for DMA-BUF RDMA: reads from RDMA-written memory bypass GPU cache.
__forceinline__ __device__ uint4 ld_global_nc_uint4(const void* addr) {
    uint4 val;
    asm volatile(
        "ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
        : "l"(addr)
    );
    return val;
}

// Non-coherent uint4 store (bypass L1)
__forceinline__ __device__ void st_global_nc_uint4(void* addr, uint4 val) {
    asm volatile(
        "st.global.L1::no_allocate.v4.u32 [%0], {%1, %2, %3, %4};"
        : : "l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
    );
}

// ============================================================================
// 6. Shared Memory
// ============================================================================

__forceinline__ __device__ uint4 ld_shared_uint4(const void* ptr) {
    uint4 v;
    asm volatile(
        "{ ld.shared.v4.u32 {%0, %1, %2, %3}, [%4]; }"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
        : "l"(ptr)
    );
    return v;
}

__forceinline__ __device__ void st_shared_uint4(void* ptr, uint4 v) {
    asm volatile(
        "{ st.shared.v4.u32 [%0], {%1, %2, %3, %4}; }"
        : : "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
    );
}

// ============================================================================
// 7. Warp-Level Primitives
// ============================================================================

// Elect one thread per warp (returns non-zero for exactly one thread).
// Uses SM90 elect.sync instruction.
__forceinline__ __device__ uint32_t elect_one_sync() {
#if __CUDA_ARCH__ >= 900
    uint32_t pred = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "      elect.sync %%rx|%%px, %1;\n"
        "@%%px mov.s32 %0, 1;\n"
        "}\n"
        : "+r"(pred)
        : "r"(0xffffffff));
    return pred;
#else
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id == 0 ? 1u : 0u;
#endif
}

// Get lane ID within warp
__forceinline__ __device__ int get_lane_id() {
    int lane_id;
    asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

}  // namespace gdr
}  // namespace deep_ep
