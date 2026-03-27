#include "gdr_signal.h"
#include <cuda.h>
#include <cstring>
#include <stdexcept>

namespace deep_ep {
namespace efa {

static gdr_t g_gdr_handle = nullptr;

gdr_t get_gdr_handle() {
    if (!g_gdr_handle) {
        g_gdr_handle = gdr_open();
        if (!g_gdr_handle) {
            throw std::runtime_error("Failed to open GDRCopy handle. "
                                     "Is gdrdrv module loaded?");
        }
    }
    return g_gdr_handle;
}

// GDRCopy requires 64KB-aligned GPU memory on newer systems.
// We allocate extra to ensure alignment, then pin/map the aligned portion.
static constexpr size_t kGdrPageSize = 65536;  // 64KB

static CUdeviceptr gdr_alloc_aligned(size_t size, CUdeviceptr* raw_ptr_out) {
    // Round up size to page boundary
    size_t alloc_size = ((size + kGdrPageSize - 1) / kGdrPageSize) * kGdrPageSize;
    // Allocate with extra page for alignment
    CUdeviceptr raw_ptr = 0;
    CUresult res = cuMemAlloc(&raw_ptr, alloc_size + kGdrPageSize);
    if (res != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc failed for GDRCopy buffer");
    }
    // Align to page boundary
    CUdeviceptr aligned = ((raw_ptr + kGdrPageSize - 1) / kGdrPageSize) * kGdrPageSize;
    // Zero the memory
    cuMemsetD8(aligned, 0, alloc_size);
    *raw_ptr_out = raw_ptr;
    return aligned;
}

GdrFlag::GdrFlag() = default;

GdrFlag::~GdrFlag() {
    if (initialized_) {
        if (cpu_map_) {
            gdr_unmap(gdr_, mh_, cpu_map_, kGdrPageSize);
        }
        if (d_mem_) {
            gdr_unpin_buffer(gdr_, mh_);
            // Free the raw (unaligned) allocation
            cuMemFree(raw_d_mem_);
        }
    }
}

void GdrFlag::init() {
    gdr_ = get_gdr_handle();

    // Allocate 64KB-aligned GPU memory (required by GDRCopy)
    CUdeviceptr raw_ptr = 0;
    CUdeviceptr aligned_ptr = gdr_alloc_aligned(kGdrPageSize, &raw_ptr);
    d_mem_ = reinterpret_cast<void*>(aligned_ptr);
    raw_d_mem_ = raw_ptr;
    d_flag_ = reinterpret_cast<volatile uint8_t*>(aligned_ptr);

    // Pin and map via GDRCopy
    GDR_CHECK(gdr_pin_buffer(gdr_, aligned_ptr, kGdrPageSize, 0, 0, &mh_));
    GDR_CHECK(gdr_map(gdr_, mh_, &cpu_map_, kGdrPageSize));

    // Get the info to compute the correct offset
    gdr_info_t info;
    GDR_CHECK(gdr_get_info(gdr_, mh_, &info));

    // The mapped address may have an offset from the page boundary
    size_t off = aligned_ptr - info.va;
    cpu_flag_ = reinterpret_cast<volatile uint8_t*>(
        static_cast<uint8_t*>(cpu_map_) + off);

    // Clear the flag
    *cpu_flag_ = 0;

    initialized_ = true;
}

// ============================================================================
// GdrVec implementation
// ============================================================================
template <typename T>
GdrVec<T>::~GdrVec() {
    if (initialized_) {
        if (cpu_map_) {
            size_t alloc_size = ((count_ * sizeof(T) + kGdrPageSize - 1) / kGdrPageSize) * kGdrPageSize;
            if (alloc_size < kGdrPageSize) alloc_size = kGdrPageSize;
            gdr_unmap(gdr_, mh_, cpu_map_, alloc_size);
        }
        if (d_mem_) {
            gdr_unpin_buffer(gdr_, mh_);
            cuMemFree(raw_d_mem_);
        }
    }
}

template <typename T>
void GdrVec<T>::init(size_t count) {
    gdr_ = get_gdr_handle();
    count_ = count;

    // Round up to 64KB page size
    size_t data_size = count * sizeof(T);
    size_t alloc_size = ((data_size + kGdrPageSize - 1) / kGdrPageSize) * kGdrPageSize;
    if (alloc_size < kGdrPageSize) alloc_size = kGdrPageSize;

    CUdeviceptr raw_ptr = 0;
    CUdeviceptr aligned_ptr = gdr_alloc_aligned(alloc_size, &raw_ptr);
    d_mem_ = reinterpret_cast<void*>(aligned_ptr);
    raw_d_mem_ = raw_ptr;

    GDR_CHECK(gdr_pin_buffer(gdr_, aligned_ptr, alloc_size, 0, 0, &mh_));
    GDR_CHECK(gdr_map(gdr_, mh_, &cpu_map_, alloc_size));

    gdr_info_t info;
    GDR_CHECK(gdr_get_info(gdr_, mh_, &info));

    size_t off = aligned_ptr - info.va;
    cpu_map_ptr_ = static_cast<uint8_t*>(cpu_map_) + off;

    initialized_ = true;
}

// Explicit template instantiations for common types
template class GdrVec<int32_t>;
template class GdrVec<uint32_t>;
template class GdrVec<int64_t>;
template class GdrVec<uint64_t>;

}  // namespace efa
}  // namespace deep_ep
