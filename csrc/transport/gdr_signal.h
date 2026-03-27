#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <gdrapi.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <memory>
#include <stdexcept>
#include <vector>

namespace deep_ep {
namespace efa {

// Check GDRCopy errors
#define GDR_CHECK(call)                                                        \
    do {                                                                        \
        int ret_ = (call);                                                      \
        if (ret_ != 0) {                                                        \
            fprintf(stderr, "GDRCopy error at %s:%d: ret=%d\n",                 \
                    __FILE__, __LINE__, ret_);                                  \
            fflush(stderr);                                                     \
            throw std::runtime_error("GDRCopy error");                          \
        }                                                                       \
    } while (0)

// GDR handle singleton - opens /dev/gdrdrv once
gdr_t get_gdr_handle();

// ============================================================================
// GdrFlag: 1-byte GPU<->CPU signal via MMIO (GDRCopy)
// GPU writes 1 byte via MMIO store, CPU reads via GDRCopy mapping.
// Provides ~1us GPU->CPU notification latency.
// ============================================================================
class GdrFlag {
public:
    GdrFlag();
    ~GdrFlag();

    // Allocate and pin the GPU memory, map via GDRCopy
    void init();

    // GPU sets the flag (call from CUDA kernel via device pointer)
    // The device pointer can be written to by the GPU
    volatile uint8_t* device_ptr() const { return d_flag_; }

    // CPU reads the flag via GDRCopy-mapped pointer (~1us)
    uint8_t read() const { return *cpu_flag_; }

    // CPU clears the flag via GDRCopy-mapped pointer
    void clear() { *cpu_flag_ = 0; }

    // CPU sets the flag via GDRCopy-mapped pointer
    void set(uint8_t val) { *cpu_flag_ = val; }

    // CPU spin-waits until flag becomes non-zero, then clears it.
    // This is the standard GPU→CPU synchronization pattern.
    void wait() {
        while (*cpu_flag_ == 0) {
            _mm_pause();
        }
        *cpu_flag_ = 0;
    }

    bool is_initialized() const { return initialized_; }

private:
    // GPU-side memory (cuMemAlloc'd, page-aligned for GDRCopy)
    void* d_mem_ = nullptr;
    CUdeviceptr raw_d_mem_ = 0;  // Raw (unaligned) allocation for freeing
    volatile uint8_t* d_flag_ = nullptr;

    // GDRCopy mapping
    gdr_t gdr_ = nullptr;
    gdr_mh_t mh_ = {};
    void* cpu_map_ = nullptr;
    volatile uint8_t* cpu_flag_ = nullptr;

    bool initialized_ = false;
};

// ============================================================================
// GdrVec: Bulk CPU->GPU metadata transfer via GDRCopy
// Used to write structured metadata (e.g., token counts) from CPU to GPU
// with low latency via MMIO.
// ============================================================================
template <typename T>
class GdrVec {
public:
    GdrVec() = default;
    ~GdrVec();

    // Allocate N elements on GPU and map via GDRCopy
    void init(size_t count);

    // CPU writes to element i
    void write(size_t i, const T& val) {
        reinterpret_cast<volatile T*>(cpu_map_ptr_)[i] = val;
    }

    // CPU reads element i
    T read(size_t i) const {
        return reinterpret_cast<volatile T*>(cpu_map_ptr_)[i];
    }

    // Bulk copy from host array to GPU via GDRCopy mapping.
    // Uses gdr_copy_to_mapping() which internally performs WC (write-combining)
    // stores + _mm_sfence(), matching pplx-garden's GdrVec::copy() semantics.
    // This ensures all writes are flushed from the WC buffer and visible to
    // the GPU before this function returns.
    void copy(const T* src, size_t n) {
        gdr_copy_to_mapping(mh_, const_cast<void*>(cpu_map_ptr_), src, n * sizeof(T));
    }

    // Bulk copy from std::vector
    void copy(const std::vector<T>& src) {
        copy(src.data(), src.size());
    }

    // GPU device pointer for kernel access
    T* device_ptr() const { return reinterpret_cast<T*>(d_mem_); }

    size_t count() const { return count_; }

private:
    void* d_mem_ = nullptr;
    CUdeviceptr raw_d_mem_ = 0;  // Raw (unaligned) allocation for freeing
    gdr_t gdr_ = nullptr;
    gdr_mh_t mh_ = {};
    void* cpu_map_ = nullptr;
    volatile void* cpu_map_ptr_ = nullptr;
    size_t count_ = 0;
    bool initialized_ = false;
};

// ============================================================================
// GdrCounter: Atomic counter that auto-sets a GdrFlag when target is reached.
//
// This is the key synchronization primitive for the pplx-garden architecture:
// - The CPU worker calls wait(N) to set a target of N completions
// - Each RDMA completion calls inc() which atomically increments the counter
// - When the counter reaches 0 (target met), the GdrFlag is auto-set
// - The GPU kernel polls the GdrFlag via ld.mmio.b8 for ~1us notification
//
// Protocol (matching pplx-garden's GdrCounter):
//   1. Worker calls wait(N) → does fetch_sub(N) on counter
//      - If old_val >= N, target already met → set flag immediately
//      - Otherwise, counter goes negative → RDMA completions will fill it
//   2. Each CQ completion calls inc() → does fetch_add(1)
//      - If counter reaches 0, set the GdrFlag
//
// This design handles out-of-order completions: inc() can be called before
// or after wait(), and the flag will be set exactly when both conditions are met.
// ============================================================================
class GdrCounter {
public:
    GdrCounter() = default;

    // Initialize with a GdrFlag to signal.
    // The flag must be initialized before calling this.
    void init(GdrFlag* flag) {
        flag_ = flag;
        counter_.store(0, std::memory_order_relaxed);
    }

    // Called by the CPU worker before waiting for completions.
    // Sets the target: subtracts N from the counter.
    // If completions already arrived (counter >= N), flag is set immediately.
    void wait(int32_t target) {
        int32_t old = counter_.fetch_sub(target, std::memory_order_acq_rel);
        if (old >= target) {
            // All completions already arrived
            flag_->set(1);
        }
        // Otherwise counter is negative; inc() will bring it to 0
    }

    // Called by the CQ completion handler (one call per RDMA completion).
    // Increments the counter by 1. If counter reaches 0, sets the flag.
    void inc() {
        int32_t old = counter_.fetch_add(1, std::memory_order_acq_rel);
        if (old == -1) {
            // Counter just went from -1 to 0 → target met
            flag_->set(1);
        }
    }

    // Reset counter and flag for the next cycle
    void reset() {
        counter_.store(0, std::memory_order_relaxed);
        flag_->clear();
    }

    // Get the associated flag (for passing device_ptr to kernels)
    GdrFlag* flag() const { return flag_; }

    // Get current counter value (for debugging)
    int32_t value() const { return counter_.load(std::memory_order_relaxed); }

private:
    std::atomic<int32_t> counter_{0};
    GdrFlag* flag_ = nullptr;
};

// ============================================================================
// ImmCounter (CPU-side): Simple atomic counter for immediate-data-based
// completion tracking. Worker calls wait(N) which spin-loops until counter >= 0.
//
// Unlike GdrCounter, this does NOT auto-set a GdrFlag — it's used for
// synchronization points where the CPU worker itself needs to wait
// (e.g., route exchange, barriers) rather than signaling the GPU.
// ============================================================================
class CpuImmCounter {
public:
    CpuImmCounter() = default;

    void init() {
        counter_.store(0, std::memory_order_relaxed);
    }

    // Worker calls this to wait for N completions.
    // Subtracts N from counter, then spin-waits until counter >= 0.
    void wait(int32_t target) {
        int32_t old = counter_.fetch_sub(target, std::memory_order_acq_rel);
        if (old >= target) {
            return;  // Already have enough completions
        }
        // Spin until completions arrive
        while (counter_.load(std::memory_order_acquire) < 0) {
            _mm_pause();
        }
    }

    // Called per RDMA completion
    void inc() {
        counter_.fetch_add(1, std::memory_order_release);
    }

    // Reset for next cycle
    void reset() {
        counter_.store(0, std::memory_order_relaxed);
    }

    int32_t value() const { return counter_.load(std::memory_order_relaxed); }

private:
    std::atomic<int32_t> counter_{0};
};

}  // namespace efa
}  // namespace deep_ep
