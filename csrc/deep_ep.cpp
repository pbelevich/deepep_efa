#include "deep_ep.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <chrono>
#include <memory>
#include <optional>
#include <unistd.h>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

// Cooperative kernel headers (host-only API, no CUDA device code)
#ifdef ENABLE_EFA
#include "kernels/coop_api.h"
#endif

// =============================================================================
// EFA VMM allocation: cuMemCreate/cuMemMap with gpuDirectRDMACapable=1
// This produces memory that can export DMA-BUF file descriptors, enabling
// true zero-copy GPU-Direct RDMA via the EFA provider (no bounce buffers).
// =============================================================================
#ifdef ENABLE_EFA
namespace efa_vmm {

struct VmmAllocation {
    CUmemGenericAllocationHandle handle;
    CUdeviceptr ptr;
    size_t size;  // aligned size
};

static VmmAllocation alloc(size_t size_raw) {
    VmmAllocation result = {};

    CUdevice device;
    CU_CHECK(cuCtxGetDevice(&device));

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    // Request POSIX file descriptor handle type — required for cuMemGetHandleForAddressRange
    // to be able to export a DMA-BUF fd for this allocation.
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    // Mark as GPU-Direct RDMA capable — tells the driver to allocate memory
    // in a region accessible to PCIe peers (NIC), essential for DMA-BUF/RDMA.
    prop.allocFlags.gpuDirectRDMACapable = 1;

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    // Align size up to granularity
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0) size = granularity;

    CU_CHECK(cuMemCreate(&result.handle, size, &prop, 0));
    CU_CHECK(cuMemAddressReserve(&result.ptr, size, granularity, 0, 0));
    CU_CHECK(cuMemMap(result.ptr, size, 0, result.handle, 0));

    // Set read/write access for all GPUs on this node
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::vector<CUmemAccessDesc> access_desc(device_count);
    for (int i = 0; i < device_count; i++) {
        access_desc[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[i].location.id = i;
        access_desc[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    CU_CHECK(cuMemSetAccess(result.ptr, size, access_desc.data(), device_count));

    result.size = size;
    return result;
}

static void free(VmmAllocation& alloc) {
    if (alloc.ptr != 0) {
        CU_CHECK(cuMemUnmap(alloc.ptr, alloc.size));
        CU_CHECK(cuMemAddressFree(alloc.ptr, alloc.size));
        CU_CHECK(cuMemRelease(alloc.handle));
        alloc.ptr = 0;
        alloc.size = 0;
    }
}

}  // namespace efa_vmm
#endif  // ENABLE_EFA

namespace shared_memory {
void cu_mem_set_access_all(void* ptr, size_t size) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    CUmemAccessDesc access_desc[device_count];
    for (int idx = 0; idx < device_count; ++idx) {
        access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[idx].location.id = idx;
        access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc, device_count));
}

void cu_mem_free(void* ptr) {
    CUmemGenericAllocationHandle handle;
    CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    CU_CHECK(cuMemRelease(handle));
}

size_t get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

SharedMemoryAllocator::SharedMemoryAllocator(bool use_fabric) : use_fabric(use_fabric) {}

void SharedMemoryAllocator::malloc(void** ptr, size_t size_raw) {
    if (use_fabric) {
        CUdevice device;
        CU_CHECK(cuCtxGetDevice(&device));

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        prop.location.id = device;

        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

        size_t size = get_size_align_to_granularity(size_raw, granularity);

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaMalloc(ptr, size_raw));
    }
}

void SharedMemoryAllocator::free(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void SharedMemoryAllocator::get_mem_handle(MemHandle* mem_handle, void* ptr) {
    size_t size = 0;
    CU_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;

    if (use_fabric) {
        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

        CU_CHECK(cuMemExportToShareableHandle(&mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        CUDA_CHECK(cudaIpcGetMemHandle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }
}

void SharedMemoryAllocator::open_mem_handle(void** ptr, MemHandle* mem_handle) {
    if (use_fabric) {
        size_t size = mem_handle->size;

        CUmemGenericAllocationHandle handle;
        CU_CHECK(cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));

        CU_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CU_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        cu_mem_set_access_all(*ptr, size);
    } else {
        CUDA_CHECK(cudaIpcOpenMemHandle(ptr, mem_handle->inner.cuda_ipc_mem_handle, cudaIpcMemLazyEnablePeerAccess));
    }
}

void SharedMemoryAllocator::close_mem_handle(void* ptr) {
    if (use_fabric) {
        cu_mem_free(ptr);
    } else {
        CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
    }
}
}  // namespace shared_memory

namespace deep_ep {

Buffer::Buffer(int rank,
               int num_ranks,
               int64_t num_nvl_bytes,
               int64_t num_rdma_bytes,
               bool low_latency_mode,
               bool explicitly_destroy,
               bool enable_shrink,
               bool use_fabric)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      enable_shrink(enable_shrink),
      low_latency_mode(low_latency_mode),
      explicitly_destroy(explicitly_destroy),
      comm_stream(at::cuda::getStreamFromPool(true)),
      shared_memory_allocator(use_fabric) {
    // Metadata memory
    int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
    int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
    int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

    // Common checks
    EP_STATIC_ASSERT(NUM_BUFFER_ALIGNMENT_BYTES % sizeof(int4) == 0, "Invalid alignment");
    EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                   (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(num_nvl_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(num_rdma_bytes / sizeof(int4) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

    // Get ranks
    CUDA_CHECK(cudaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
#ifdef DISABLE_NVSHMEM
#ifndef ENABLE_EFA
    EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
#endif
#endif

    // Get device info
    cudaDeviceProp device_prop = {};
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Number of per-channel bytes cannot be large
    EP_HOST_ASSERT(ceil_div<int64_t>(num_nvl_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());
    EP_HOST_ASSERT(ceil_div<int64_t>(num_rdma_bytes, num_device_sms / 2) < std::numeric_limits<int>::max());

    if (num_nvl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handles
        shared_memory_allocator.malloc(&buffer_ptrs[nvl_rank],
                                       num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes);
        shared_memory_allocator.get_mem_handle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]);
        buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

        // Set barrier signals
        barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
        barrier_signal_ptrs_gpu =
            reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    if (not explicitly_destroy) {
        destroy();
    } else if (not destroyed) {
        printf("WARNING: destroy() was not called before DeepEP buffer destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_NVL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? nvl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    const shared_memory::MemHandle& handle = ipc_handles[nvl_rank];
    return {reinterpret_cast<const char*>(&handle), sizeof(handle)};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get NVSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = static_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
    auto num_bytes = (use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes) - offset;
    EP_HOST_ASSERT(num_bytes > 0 && "Offset exceeds buffer size");
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

torch::Stream Buffer::get_comm_stream() const {
    return comm_stream;
}

void Buffer::destroy() {
    EP_HOST_ASSERT(not destroyed);

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    if (num_nvl_bytes > 0) {
        // Barrier
        intranode::barrier(barrier_signal_ptrs_gpu, nvl_rank, num_nvl_ranks, comm_stream);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_nvl_ranks; ++i)
                if (i != nvl_rank)
                    shared_memory_allocator.close_mem_handle(buffer_ptrs[i]);
        }

        // Free local buffer and error flag
        shared_memory_allocator.free(buffer_ptrs[nvl_rank]);
    }

    // Free NVSHMEM / EFA
#ifdef ENABLE_EFA
    if (is_available() and num_rdma_bytes > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        // EFA cleanup: free the RDMA buffer (allocated via CUDA VMM in EFA mode)
        efa_vmm::VmmAllocation vmm_alloc;
        vmm_alloc.handle = rdma_vmm_handle_;
        vmm_alloc.ptr = rdma_vmm_ptr_;
        vmm_alloc.size = rdma_vmm_size_;
        efa_vmm::free(vmm_alloc);
        rdma_buffer_ptr = nullptr;
        // EFA worker manager cleanup is handled by its destructor
    }
#elif !defined(DISABLE_NVSHMEM)
    if (is_available() and num_rdma_bytes > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        if (enable_shrink) {
            internode::free(mask_buffer_ptr);
            internode::free(sync_buffer_ptr);
        }
        internode::finalize();
    }
#endif

    // Free workspace and MoE counter
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

    destroyed = true;
    available = false;
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    if (num_nvl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == shared_memory::HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE);
                shared_memory_allocator.open_mem_handle(&buffer_ptrs[i], &ipc_handles[i]);
                barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(&ipc_handles[i], handle_str.c_str(), shared_memory::HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and barrier signal pointers to GPU
        CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(barrier_signal_ptrs_gpu, barrier_signal_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Sync NVSHMEM handles and allocate memory / EFA setup
#ifdef ENABLE_EFA
    if (num_rdma_bytes > 0) {
        // In EFA mode, allocate RDMA buffer via CUDA VMM (cuMemCreate/cuMemMap)
        // with gpuDirectRDMACapable=1. This enables cuMemGetHandleForAddressRange
        // to export a DMA-BUF fd, which in turn enables true zero-copy GPU-Direct
        // RDMA via the EFA provider (no bounce buffers).
        auto vmm = efa_vmm::alloc(num_rdma_bytes);
        rdma_vmm_handle_ = vmm.handle;
        rdma_vmm_ptr_ = vmm.ptr;
        rdma_vmm_size_ = vmm.size;
        rdma_buffer_ptr = reinterpret_cast<void*>(vmm.ptr);
        CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

        // Allocate shrink buffers if needed (small, cudaMalloc is fine)
        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            CUDA_CHECK(cudaMalloc(&mask_buffer_ptr, num_mask_buffer_bytes));
            CUDA_CHECK(cudaMalloc(&sync_buffer_ptr, num_sync_buffer_bytes));
            CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
            CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // NOTE: The actual EFA transport initialization (libfabric endpoint setup,
        // memory registration, address exchange) is done from Python via the
        // buffer.py sync method, which calls a separate init_efa() method.
        // This keeps the C++ code clean and allows the Python side to manage
        // the allgather communication.
    }
#elif !defined(DISABLE_NVSHMEM)
    if (num_rdma_bytes > 0) {
        // Initialize NVSHMEM
        EP_HOST_ASSERT(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto nvshmem_rank = low_latency_mode ? rank : rdma_rank;
        auto num_nvshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        EP_HOST_ASSERT(nvshmem_rank == internode::init(root_unique_id, nvshmem_rank, num_nvshmem_ranks, low_latency_mode));
        internode::barrier();

        // Allocate
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        // Clean buffer (mainly for low-latency mode)
        CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

        // Allocate and clean shrink buffer
        if (enable_shrink) {
            int num_mask_buffer_bytes = num_ranks * sizeof(int);
            int num_sync_buffer_bytes = num_ranks * sizeof(int);
            mask_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_mask_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            sync_buffer_ptr = reinterpret_cast<int*>(internode::alloc(num_sync_buffer_bytes, NUM_BUFFER_ALIGNMENT_BYTES));
            CUDA_CHECK(cudaMemset(mask_buffer_ptr, 0, num_mask_buffer_bytes));
            CUDA_CHECK(cudaMemset(sync_buffer_ptr, 0, num_sync_buffer_bytes));
        }

        // Barrier
        internode::barrier();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
#endif

    // Ready to use
    available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(
    const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    layout::get_dispatch_layout(topk_idx.data_ptr<topk_idx_t>(),
                                num_tokens_per_rank.data_ptr<int>(),
                                num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                num_tokens_per_expert.data_ptr<int>(),
                                is_token_in_rank.data_ptr<bool>(),
                                num_tokens,
                                num_topk,
                                num_ranks,
                                num_experts,
                                comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x,
                           const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx,
                           const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank,
                           const torch::Tensor& is_token_in_rank,
                           const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                           const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment,
                           int num_worst_tokens,
                           const Config& config,
                           std::optional<EventHandle>& previous_event,
                           bool async,
                           bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(
            rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu, barrier_signal_ptrs_gpu, rank, num_ranks, comm_stream);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   num_tokens,
                                   is_token_in_rank.data_ptr<bool>(),
                                   channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int,
                                   expert_alignment,
                                   buffer_ptrs_gpu,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   num_channels);

        if (num_worst_tokens > 0) {
            // No CPU sync, just allocate the worst case
            num_recv_tokens = num_worst_tokens;

            // Must be forward with top-k stuffs
            EP_HOST_ASSERT(topk_idx.has_value());
            EP_HOST_ASSERT(topk_weights.has_value());
        } else {
            // Synchronize total received tokens and tokens per expert
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS)
                    throw std::runtime_error("DeepEP error: CPU recv timeout");
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Dispatch
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +                                                                     // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +                                                              // Channel start offset
            num_channels * num_ranks * sizeof(int) +                                                              // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +                                                          // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * recv_x.element_size() +  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +                     // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(topk_idx_t) +   // Top-k index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float) +        // Top-k weight buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) * num_scales        // FP8 scale buffer
        <= num_nvl_bytes);
    intranode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_src_idx.data_ptr<int>(),
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(),
                        channel_prefix_matrix.data_ptr<int>(),
                        num_tokens,
                        num_worst_tokens,
                        static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)),
                        num_topk,
                        num_experts,
                        num_scales,
                        scale_token_stride,
                        scale_hidden_stride,
                        buffer_ptrs_gpu,
                        rank,
                        num_ranks,
                        comm_stream,
                        config.num_sms,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        rank_prefix_matrix,
                        channel_prefix_matrix,
                        recv_x,
                        recv_src_idx,
                        recv_channel_prefix_matrix,
                        send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_expert,
                         cached_channel_prefix_matrix,
                         cached_rank_prefix_matrix,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rank_prefix_matrix,
            channel_prefix_matrix,
            recv_channel_prefix_matrix,
            recv_src_idx,
            send_head,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::intranode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_idx,
    const torch::Tensor& rank_prefix_matrix,
    const torch::Tensor& channel_prefix_matrix,
    const torch::Tensor& send_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                   rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                   channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu,
                                     send_head.data_ptr<int>(),
                                     num_channels,
                                     num_recv_tokens,
                                     num_channels * num_ranks * 2,
                                     barrier_signal_ptrs_gpu,
                                     rank,
                                     num_ranks,
                                     comm_stream);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_recv_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * x.element_size() +  // Data buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +             // Source index buffer
                       num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk * sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       recv_x.data_ptr(),
                       recv_topk_weights_ptr,
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       src_idx.data_ptr<int>(),
                       rank_prefix_matrix.data_ptr<int>(),
                       channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(),
                       num_tokens,
                       num_recv_tokens,
                       hidden,
                       num_topk,
                       buffer_ptrs_gpu,
                       rank,
                       num_ranks,
                       comm_stream,
                       config.num_sms,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, recv_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::vector<int>,
           torch::Tensor,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<torch::Tensor>,
           std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x,
                            const std::optional<torch::Tensor>& x_scales,
                            const std::optional<torch::Tensor>& topk_idx,
                            const std::optional<torch::Tensor>& topk_weights,
                            const std::optional<torch::Tensor>& num_tokens_per_rank,
                            const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                            const torch::Tensor& is_token_in_rank,
                            const std::optional<torch::Tensor>& num_tokens_per_expert,
                            int cached_num_recv_tokens,
                            int cached_num_rdma_recv_tokens,
                            const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                            const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                            const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                            const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                            int expert_alignment,
                            int num_worst_tokens,
                            const Config& config,
                            std::optional<EventHandle>& previous_event,
                            bool async,
                            bool allocate_on_comm_stream) {
#if !defined(DISABLE_NVSHMEM) || defined(ENABLE_EFA)
    // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
    // If users of DeepEP need to execute other Python code on other threads, such as KV transfer, their code will get stuck due to GIL
    // unless we release GIL here.
    pybind11::gil_scoped_release release;

    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                       cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                       cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    topk_idx_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<topk_idx_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32 or x_scales->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(x_scales->dim() == 2);
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float*>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        internode::cached_notify(hidden_int4,
                                 num_scales,
                                 num_topk,
                                 num_topk,
                                 num_ranks,
                                 num_channels,
                                 0,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 rdma_buffer_ptr,
                                 config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu,
                                 config.num_max_nvl_chunked_recv_tokens,
                                 barrier_signal_ptrs_gpu,
                                 rank,
                                 comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_nvl_bytes,
                                 true,
                                 low_latency_mode);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(),
                                   moe_recv_counter_mapped,
                                   num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(),
                                   moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(),
                                   moe_recv_expert_counter_mapped,
                                   num_experts,
                                   is_token_in_rank.data_ptr<bool>(),
                                   num_tokens,
                                   num_worst_tokens,
                                   num_channels,
                                   hidden_int4,
                                   num_scales,
                                   num_topk,
                                   expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(),
                                   recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(),
                                   recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr,
                                   config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu,
                                   config.num_max_nvl_chunked_recv_tokens,
                                   barrier_signal_ptrs_gpu,
                                   rank,
                                   comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_nvl_bytes,
                                   low_latency_mode);

        // Synchronize total received tokens and tokens per expert
        if (num_worst_tokens > 0) {
            num_recv_tokens = num_worst_tokens;
            num_rdma_recv_tokens = num_worst_tokens;
        } else {
            auto start_time = std::chrono::high_resolution_clock::now();
            while (true) {
                // Read total count
                num_recv_tokens = static_cast<int>(*moe_recv_counter);
                num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

                // Read per-expert count
                bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
                for (int i = 0; i < num_local_experts and ready; ++i)
                    ready &= moe_recv_expert_counter[i] >= 0;

                if (ready)
                    break;

                // Timeout check
                if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() >
                    NUM_CPU_TIMEOUT_SECS) {
                    printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                    for (int i = 0; i < num_local_experts; ++i)
                        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                    throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
                }
            }
            num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
        }
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
         recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_nvl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
        send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
    }

    // Assign pointers
    topk_idx_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<topk_idx_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                             : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = static_cast<float*>(recv_x_scales->data_ptr());
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.cu` file
    internode::dispatch(recv_x.data_ptr(),
                        recv_x_scales_ptr,
                        recv_topk_idx_ptr,
                        recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(),
                        x_scales_ptr,
                        topk_idx_ptr,
                        topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                        cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(),
                        recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(),
                        recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        is_token_in_rank.data_ptr<bool>(),
                        num_tokens,
                        num_worst_tokens,
                        hidden_int4,
                        num_scales,
                        num_topk,
                        num_experts,
                        scale_token_stride,
                        scale_hidden_stride,
                        rdma_buffer_ptr,
                        config.num_max_rdma_chunked_send_tokens,
                        config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu,
                        config.num_max_nvl_chunked_send_tokens,
                        config.num_max_nvl_chunked_recv_tokens,
                        rank,
                        num_ranks,
                        cached_mode,
                        comm_stream,
                        num_channels,
                        low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        is_token_in_rank,
                        recv_x,
                        rdma_channel_prefix_matrix,
                        recv_rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {x_scales,
                         topk_idx,
                         topk_weights,
                         num_tokens_per_rank,
                         num_tokens_per_rdma_rank,
                         num_tokens_per_expert,
                         cached_rdma_channel_prefix_matrix,
                         cached_recv_rdma_rank_prefix_sum,
                         cached_gbl_channel_prefix_matrix,
                         cached_recv_gbl_rank_prefix_sum,
                         recv_topk_idx,
                         recv_topk_weights,
                         recv_x_scales,
                         recv_rdma_channel_prefix_matrix,
                         recv_gbl_channel_prefix_matrix,
                         send_rdma_head,
                         send_nvl_head,
                         recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {recv_x,
            recv_x_scales,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix,
            gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix,
            recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix,
            recv_gbl_rank_prefix_sum,
            recv_src_meta,
            send_rdma_head,
            send_nvl_head,
            event};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& topk_weights,
    const std::optional<torch::Tensor>& bias_0,
    const std::optional<torch::Tensor>& bias_1,
    const torch::Tensor& src_meta,
    const torch::Tensor& is_combined_token_in_rank,
    const torch::Tensor& rdma_channel_prefix_matrix,
    const torch::Tensor& rdma_rank_prefix_sum,
    const torch::Tensor& gbl_channel_prefix_matrix,
    const torch::Tensor& combined_rdma_head,
    const torch::Tensor& combined_nvl_head,
    const Config& config,
    std::optional<EventHandle>& previous_event,
    bool async,
    bool allocate_on_comm_stream) {
#if !defined(DISABLE_NVSHMEM) || defined(ENABLE_EFA)
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                   is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                   rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                   rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                   gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                   combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and combined_nvl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
         hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                   combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::cuda::setCurrentCUDAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4,
                             0,
                             0,
                             num_topk,
                             num_ranks,
                             num_channels,
                             num_combined_tokens,
                             combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(),
                             rdma_rank_prefix_sum.data_ptr<int>(),
                             combined_nvl_head.data_ptr<int>(),
                             rdma_buffer_ptr,
                             config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu,
                             config.num_max_nvl_chunked_recv_tokens,
                             barrier_signal_ptrs_gpu,
                             rank,
                             comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes,
                             false,
                             low_latency_mode);

    // Assign bias pointers
    auto bias_opts = std::vector<std::optional<torch::Tensor>>({bias_0, bias_1});
    void* bias_ptrs[2] = {nullptr, nullptr};
    for (int i = 0; i < 2; ++i)
        if (bias_opts[i].has_value()) {
            auto bias = bias_opts[i].value();
            EP_HOST_ASSERT(bias.dim() == 2 and bias.is_contiguous());
            EP_HOST_ASSERT(bias.scalar_type() == x.scalar_type());
            EP_HOST_ASSERT(bias.size(0) == num_combined_tokens and bias.size(1) == hidden);
            bias_ptrs[i] = bias.data_ptr();
        }

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    internode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()),
                       combined_x.data_ptr(),
                       combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(),
                       topk_weights_ptr,
                       bias_ptrs[0],
                       bias_ptrs[1],
                       combined_rdma_head.data_ptr<int>(),
                       combined_nvl_head.data_ptr<int>(),
                       src_meta.data_ptr(),
                       rdma_channel_prefix_matrix.data_ptr<int>(),
                       rdma_rank_prefix_sum.data_ptr<int>(),
                       gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens,
                       num_combined_tokens,
                       hidden,
                       num_topk,
                       rdma_buffer_ptr,
                       config.num_max_rdma_chunked_send_tokens,
                       config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu,
                       config.num_max_nvl_chunked_send_tokens,
                       config.num_max_nvl_chunked_recv_tokens,
                       rank,
                       num_ranks,
                       comm_stream,
                       num_channels,
                       low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t : {x,
                        src_meta,
                        is_combined_token_in_rank,
                        rdma_channel_prefix_matrix,
                        rdma_rank_prefix_sum,
                        gbl_channel_prefix_matrix,
                        combined_x,
                        combined_rdma_head,
                        combined_nvl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to : {topk_weights, combined_topk_weights, bias_0, bias_1}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::cuda::setCurrentCUDAStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
#else  // !defined(DISABLE_NVSHMEM) || defined(ENABLE_EFA)
    EP_HOST_ASSERT(false and "NVSHMEM/EFA is disabled during compilation");
    return {};
#endif
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first,
                                           clean_meta_0.second,
                                           clean_meta_1.first,
                                           clean_meta_1.second,
                                           rank,
                                           num_ranks,
                                           mask_buffer_ptr,
                                           sync_buffer_ptr,
                                           at::cuda::getCurrentCUDAStream());
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
#endif
}

std::tuple<torch::Tensor,
           std::optional<torch::Tensor>,
           torch::Tensor,
           torch::Tensor,
           torch::Tensor,
           std::optional<EventHandle>,
           std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x,
                             const torch::Tensor& topk_idx,
                             const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
                             const std::optional<torch::Tensor>& dispatch_wait_recv_cost_stats,
                             int num_max_dispatch_tokens_per_rank,
                             int num_experts,
                             bool use_fp8,
                             bool round_scale,
                             bool use_ue8m0,
                             bool async,
                             bool return_recv_hook) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    // Diagnosis tensors
    if (cumulative_local_expert_recv_stats.has_value()) {
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->dim() == 1 and cumulative_local_expert_recv_stats->is_contiguous());
        EP_HOST_ASSERT(cumulative_local_expert_recv_stats->size(0) == num_experts / num_ranks);
    }
    if (dispatch_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->dim() == 1 and dispatch_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(dispatch_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_topk = static_cast<int>(topk_idx.size(1));
    auto num_local_experts = num_experts / num_ranks;

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                      x.options().dtype(use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16));
    auto packed_recv_src_info =
        torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));

    // Allocate column-majored scales
    auto packed_recv_x_scales = std::optional<torch::Tensor>();
    void* packed_recv_x_scales_ptr = nullptr;
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");

    if (use_fp8) {
        // TODO: support unaligned cases
        EP_HOST_ASSERT(hidden % 512 == 0);
        if (not use_ue8m0) {
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 128, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kFloat32).device(torch::kCUDA));
        } else {
            EP_HOST_ASSERT(round_scale);
            packed_recv_x_scales = torch::empty({num_local_experts, hidden / 512, num_ranks * num_max_dispatch_tokens_per_rank},
                                                torch::dtype(torch::kInt).device(torch::kCUDA));
        }
        packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
        packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr();
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(
            packed_recv_x.data_ptr(),
            packed_recv_x_scales_ptr,
            packed_recv_src_info.data_ptr<int>(),
            packed_recv_layout_range.data_ptr<int64_t>(),
            packed_recv_count.data_ptr<int>(),
            mask_buffer_ptr,
            cumulative_local_expert_recv_stats.has_value() ? cumulative_local_expert_recv_stats->data_ptr<int>() : nullptr,
            dispatch_wait_recv_cost_stats.has_value() ? dispatch_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
            buffer.dispatch_rdma_recv_data_buffer,
            buffer.dispatch_rdma_recv_count_buffer,
            buffer.dispatch_rdma_send_buffer,
            x.data_ptr(),
            topk_idx.data_ptr<topk_idx_t>(),
            next_clean_meta.first,
            next_clean_meta.second,
            num_tokens,
            hidden,
            num_max_dispatch_tokens_per_rank,
            num_topk,
            num_experts,
            rank,
            num_ranks,
            use_fp8,
            round_scale,
            use_ue8m0,
            workspace,
            num_device_sms,
            launch_stream,
            phases);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const torch::Tensor& x,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    const torch::Tensor& src_info,
    const torch::Tensor& layout_range,
    const std::optional<torch::Tensor>& combine_wait_recv_cost_stats,
    int num_max_dispatch_tokens_per_rank,
    int num_experts,
    bool use_logfmt,
    bool zero_copy,
    bool async,
    bool return_recv_hook,
    const std::optional<torch::Tensor>& out) {
#ifndef DISABLE_NVSHMEM
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == c10::CppTypeToScalarType<topk_idx_t>::value);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);

    if (combine_wait_recv_cost_stats.has_value()) {
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->scalar_type() == torch::kInt64);
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->dim() == 1 and combine_wait_recv_cost_stats->is_contiguous());
        EP_HOST_ASSERT(combine_wait_recv_cost_stats->size(0) == num_ranks);
    }

    auto hidden = static_cast<int>(x.size(2));
    auto num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::cuda::getCurrentCUDAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not(async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    torch::Tensor combined_x;
    if (out.has_value()) {
        EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
        EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
        EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
        combined_x = out.value();
    } else {
        combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    }

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer,
                              buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(),
                              topk_idx.data_ptr<topk_idx_t>(),
                              topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(),
                              layout_range.data_ptr<int64_t>(),
                              mask_buffer_ptr,
                              combine_wait_recv_cost_stats.has_value() ? combine_wait_recv_cost_stats->data_ptr<int64_t>() : nullptr,
                              next_clean_meta.first,
                              next_clean_meta.second,
                              num_combined_tokens,
                              hidden,
                              num_max_dispatch_tokens_per_rank,
                              num_topk,
                              num_experts,
                              rank,
                              num_ranks,
                              use_logfmt,
                              workspace,
                              num_device_sms,
                              launch_stream,
                              phases,
                              zero_copy);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) const {
#ifndef DISABLE_NVSHMEM
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);

    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto dtype = torch::kBFloat16;
    auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

    EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
    return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                            {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                            {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                            torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
#else
    EP_HOST_ASSERT(false and "NVSHMEM is disabled during compilation");
    return {};
#endif
}

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

void Buffer::low_latency_update_mask_buffer(int rank_to_mask, bool mask) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(rank_to_mask >= 0 and rank_to_mask < num_ranks);
    internode_ll::update_mask_buffer(mask_buffer_ptr, rank_to_mask, mask, at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_query_mask_buffer(const torch::Tensor& mask_status) {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    EP_HOST_ASSERT(mask_status.numel() == num_ranks && mask_status.scalar_type() == torch::kInt32);

    internode_ll::query_mask_buffer(
        mask_buffer_ptr, num_ranks, reinterpret_cast<int*>(mask_status.data_ptr()), at::cuda::getCurrentCUDAStream());
}

void Buffer::low_latency_clean_mask_buffer() {
    EP_HOST_ASSERT(mask_buffer_ptr != nullptr and "Shrink mode must be enabled");
    internode_ll::clean_mask_buffer(mask_buffer_ptr, num_ranks, at::cuda::getCurrentCUDAStream());
}

}  // namespace deep_ep

#ifdef ENABLE_EFA
namespace deep_ep {

void Buffer::init_efa(const pybind11::function& allgather_fn_py) {
    EP_HOST_ASSERT(num_rdma_bytes > 0 && rdma_buffer_ptr != nullptr);
    EP_HOST_ASSERT(!efa_manager_ && "EFA already initialized");

    // Create EFA worker manager
    efa_manager_ = std::make_unique<efa::EfaWorkerManager>();

    // For internode mode, each NVL rank 0 in a node does the RDMA.
    // But we initialize on ALL ranks so everyone has transport ready.
    // The rdma_rank indexes nodes (0..num_rdma_ranks-1).
    // Within a node, only nvl_rank == 0 does the actual RDMA transfers
    // in the current simplified design. Other ranks participate in the
    // address exchange but don't issue writes.
    efa_rdma_rank_ = rdma_rank;

    efa_manager_->init(device_id, rank, num_ranks);

    // Wrap the Python allgather function to match C++ signature
    auto allgather_cpp = [&allgather_fn_py](const std::vector<uint8_t>& local_data) -> std::vector<std::vector<uint8_t>> {
        pybind11::gil_scoped_acquire acquire;

        // Convert local data to bytes
        pybind11::bytes local_bytes(reinterpret_cast<const char*>(local_data.data()), local_data.size());

        // Call Python allgather
        pybind11::object result = allgather_fn_py(local_bytes);

        // Convert result back to vector of vectors
        pybind11::list result_list = result.cast<pybind11::list>();
        std::vector<std::vector<uint8_t>> all_data;
        all_data.reserve(result_list.size());
        for (auto& item : result_list) {
            std::string s = item.cast<std::string>();
            all_data.emplace_back(s.begin(), s.end());
        }
        return all_data;
    };

    efa_manager_->setup_rdma(rdma_buffer_ptr, rdma_vmm_size_, allgather_cpp);
}

// Per-GPU tag counter for EFA RDMA transfers (efa_all_to_all + efa_count_exchange)
// Uses thread_local so each GPU thread has its own independent counter.
// This ensures that corresponding calls on different ranks/nodes get the same tag,
// even when multiple GPUs on the same node race for tag allocation.
static thread_local uint8_t g_efa_transfer_tag{0};

// Shared receive buffer tracking across efa_all_to_all and efa_count_exchange.
// Tracks how many receive buffers are currently posted on each NIC.
// Must be initialized to the setup_rdma initial pool size (num_ranks * 256)
// on first use.
static thread_local std::vector<int> g_posted_per_nic;
static thread_local bool g_posted_per_nic_initialized{false};

// Forward declaration: NC-load gather kernel for reading RDMA-written counts
// Lives in deep_ep::efa_kernels namespace in efa_kernels.cu
// Note: We are already inside namespace deep_ep at this point.
namespace efa_kernels {
void nc_gather_counts(const int32_t* src, int32_t* dst,
                      int num_ranks, int stride_int32, int self_rank,
                      cudaStream_t stream);
}  // namespace efa_kernels

void Buffer::efa_all_to_all(const torch::Tensor& send_buf,
                            const std::vector<int64_t>& send_sizes,
                            const std::vector<int64_t>& send_offsets,
                            const torch::Tensor& recv_buf,
                            const std::vector<int64_t>& recv_sizes,
                            const std::vector<int64_t>& recv_offsets,
                            int override_tag,
                            int override_baseline,
                            const std::vector<int32_t>* send_token_counts,
                            std::vector<int32_t>* recv_token_counts,
                            int packed_bytes_per_token) {
    EP_HOST_ASSERT(efa_manager_ && efa_manager_->is_initialized());

    // Call counter for debugging
    static thread_local int a2a_call_count = 0;
    int this_call = a2a_call_count++;

    // Profiling control
    static bool profile_enabled = (std::getenv("DEEPEP_PROFILE") != nullptr &&
                                   std::string(std::getenv("DEEPEP_PROFILE")) == "1");
    static thread_local int profile_call_count = 0;
    static int profile_warmup = 5;
    static int profile_interval = 20;
    static thread_local double accum_stream_sync = 0, accum_drain_old = 0, accum_post_recv = 0;
    static thread_local double accum_rdma_writes = 0, accum_cq_wait = 0, accum_drain_tx = 0;
    static thread_local double accum_dev_sync = 0, accum_total = 0;
    static thread_local int accum_count = 0;
    static thread_local int warmup_remaining = profile_warmup;

    // Read interval from env once
    static bool interval_init = false;
    if (!interval_init) {
        const char* env_interval = std::getenv("DEEPEP_PROFILE_INTERVAL");
        if (env_interval) profile_interval = std::atoi(env_interval);
        const char* env_warmup = std::getenv("DEEPEP_PROFILE_WARMUP");
        if (env_warmup) profile_warmup = std::atoi(env_warmup);
        warmup_remaining = profile_warmup;
        interval_init = true;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_mark = t_start;
    auto us_since = [&](auto& prev) {
        auto now = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(now - prev).count();
        prev = now;
        return us;
    };

    // Release GIL since we'll busy-wait on CQs
    pybind11::gil_scoped_release release;

    int num_peers = static_cast<int>(send_sizes.size());
    EP_HOST_ASSERT(num_peers == static_cast<int>(send_offsets.size()));
    EP_HOST_ASSERT(num_peers == static_cast<int>(recv_sizes.size()));
    EP_HOST_ASSERT(num_peers == static_cast<int>(recv_offsets.size()));

    // Ensure GPU data is ready
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));

    double t_stream_sync = us_since(t_mark);

    // Both send_buf and recv_buf are within the registered rdma_buffer_ptr region.
    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);
    uint8_t* send_base = static_cast<uint8_t*>(send_buf.data_ptr());
    uint8_t* recv_base = static_cast<uint8_t*>(recv_buf.data_ptr());

    // Compute the offset of recv_buf within the RDMA buffer for remote writes
    uint64_t recv_buf_rdma_offset = recv_base - rdma_base;

    auto& transport = efa_manager_->transport();
    auto& counters = efa_manager_->counters();
    int num_nics = transport.num_nics();

    // Tag for this transfer: either pre-reserved or auto-allocated.
    // Pre-reserved tags are used when the caller needs to capture the baseline
    // before intermediate operations (e.g., count exchange) that might pick up
    // early completions for this tag.
    uint8_t tag;
    if (override_tag >= 0) {
        tag = static_cast<uint8_t>(override_tag);
    } else {
        tag = g_efa_transfer_tag++;
    }

    // Debug: trace tag allocation and imm_count usage (disabled)

    // Iter 44 FIX: Capture per-rank CQ baselines for freshness detection.
    // When recv_token_counts is requested, we need to distinguish fresh
    // token counts (from early arrivals of THIS call) vs stale ones (from
    // previous epochs or non-dispatch intermediate calls). record() stores
    // the CQ count at which each token_count was set, and get_fresh_token_count()
    // compares against the per-rank baseline to determine freshness.
    // No reset needed — freshness is determined by CQ count comparison.
    std::vector<int> per_rank_cq_baseline;
    if (recv_token_counts) {
        per_rank_cq_baseline.resize(num_peers);
        for (int i = 0; i < num_peers; ++i) {
            per_rank_cq_baseline[i] = counters.get_count(tag, i);
        }
    }

    // Count expected incoming writes.
    // Each remote peer shards its data across NICs. For small data sizes,
    // not all NICs may be used (min shard is SHARD_ALIGN bytes).
    static constexpr size_t SHARD_ALIGN = 8192;  // Match pplx-garden's MIN_SIZE
    // Limit outstanding writes per NIC to avoid overwhelming EFA send queue.
    // EFA RDM endpoints typically support ~128 outstanding operations.
    // 64 is enough for EP64 (56 remote peers, ~56 writes/NIC) with minimal drain pauses.
    static constexpr int MAX_OUTSTANDING_PER_NIC = 64;

    int num_expected_recvs = 0;
    for (int i = 0; i < num_peers; ++i) {
        if (i != rank && recv_sizes[i] > 0) {
            // Compute actual number of shards the remote peer will send
            size_t total_bytes = static_cast<size_t>(recv_sizes[i]);
            size_t bytes_per_shard = ((total_bytes / num_nics) + SHARD_ALIGN - 1) & ~(SHARD_ALIGN - 1);
            if (bytes_per_shard == 0) bytes_per_shard = SHARD_ALIGN;
            size_t shard_offset = 0;
            int actual_shards = 0;
            for (int n = 0; n < num_nics; ++n) {
                size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
                if (shard_len == 0) break;
                actual_shards++;
                shard_offset += shard_len;
            }
            num_expected_recvs += actual_shards;
        }
    }

    // Capture baseline count for this tag (instead of resetting).
    // This allows back-to-back transfers to work without barriers:
    // early-arriving completions from a future transfer are simply counted
    // cumulatively, so we wait for (baseline + expected) instead of (0 + expected).
    // When override_baseline is provided, use that (caller captured it earlier
    // to avoid race with intermediate CQ polling that inflates the baseline).
    int rx_baseline = (override_baseline >= 0) ? override_baseline : counters.get_total(tag);

    // Drain leftover RX CQ entries from previous transfers on ALL NICs
    // Use file-scope g_posted_per_nic for cross-function tracking.
    // Initialize to setup_rdma's initial pool size (num_ranks * 256) on first use.
    if (!g_posted_per_nic_initialized) {
        int initial_pool = transport.num_ranks() * 256;
        g_posted_per_nic.resize(num_nics, initial_pool);
        g_posted_per_nic_initialized = true;
    }
    // Iter 43: When override_tag is used (after efa_imm_barrier), the barrier's
    // CQ polling consumed an unknown number of recv buffers without updating our
    // tracking. Reset to 0 to force full re-posting.
    if (override_tag >= 0) {
        for (int n = 0; n < num_nics; ++n) {
            g_posted_per_nic[n] = 0;
        }
    }
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport.endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = efa::poll_rx_cq(ep, drain_entries, 64);
            if (drained > 0) {
                g_posted_per_nic[n] -= drained;  // track consumed receives
            }
            for (int j = 0; j < drained; ++j) {
                if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                    counters.record(static_cast<uint32_t>(drain_entries[j].data));
                }
            }
        } while (drained > 0);
    }

    double t_drain_old = us_since(t_mark);

    // Persistent receive pool — only replenish what was consumed.
    // Maintains running count across calls, reducing post_recv from ~25us to ~5us.
    static thread_local char recv_dummy[64] = {};
    int target_per_nic = std::max(num_expected_recvs, 64) + 16;
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport.endpoint(n);
        int needed = target_per_nic - g_posted_per_nic[n];
        for (int i = 0; i < needed; ++i) {
            int ret = efa::post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
            if (ret == -FI_EAGAIN) {
                struct fi_cq_data_entry drain_entries[64];
                int drained = efa::poll_rx_cq(ep, drain_entries, 64);
                for (int j = 0; j < drained; ++j) {
                    if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        counters.record(static_cast<uint32_t>(drain_entries[j].data));
                    }
                }
                g_posted_per_nic[n] -= drained;  // consumed receives
                ret = efa::post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
            }
            if (ret == 0) {
                g_posted_per_nic[n]++;
            } else if (ret != -FI_EAGAIN) {
                fprintf(stderr, "EFA post_recv failed: NIC=%d, ret=%d (%s)\n",
                        n, ret, fi_strerror(-ret));
            }
        }
    }

    double t_post_recv = us_since(t_mark);

    // Issue RDMA writes to all remote peers, sharding across NICs
    // Each peer's data is split into num_nics shards, each sent via a different NIC
    int total_writes = 0;
    std::vector<int> writes_per_nic(num_nics, 0);
    std::vector<int> tx_completed(num_nics, 0);
    int total_tx_completed = 0;

    for (int i = 0; i < num_peers; ++i) {
        if (i == rank) {
            // Local copy (async on comm_stream, completed by final stream sync)
            if (send_sizes[i] > 0) {
                void* src = send_base + send_offsets[i];
                void* dst = recv_base + recv_offsets[i];
                CUDA_CHECK(cudaMemcpyAsync(dst, src, send_sizes[i], cudaMemcpyDeviceToDevice, comm_stream.stream()));
            }
            continue;
        }

        if (send_sizes[i] <= 0 && !send_token_counts) continue;

        // Iter 44: For 0-token peers when using imm counts, send a notification-only
        // RDMA write with SHARD_ALIGN bytes on NIC 0 carrying the token count.
        if (send_sizes[i] <= 0 && send_token_counts) {
            auto& ep = transport.endpoint(0);
            // Send SHARD_ALIGN bytes from beginning of send buffer (content doesn't matter)
            uint32_t imm = efa::ImmCounterMap::encode_with_tokens(tag, rank, 0);
            int ret = efa::rdma_write_with_imm(
                ep, ep.remote_addrs[i], send_base, SHARD_ALIGN,
                recv_buf_rdma_offset + recv_offsets[i],
                ep.remote_keys[i], ep.remote_base_addrs[i], imm, nullptr);
            struct fi_cq_data_entry tx_entries[64];
            while (ret == -FI_EAGAIN) {
                int polled = efa::poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) { tx_completed[0] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_entries[64];
                int rx_polled = efa::poll_rx_cq(ep, rx_entries, 64);
                if (rx_polled > 0) g_posted_per_nic[0] -= rx_polled;
                for (int j = 0; j < rx_polled; ++j) {
                    if (rx_entries[j].flags & FI_REMOTE_CQ_DATA)
                        counters.record(static_cast<uint32_t>(rx_entries[j].data));
                }
                ret = efa::rdma_write_with_imm(
                    ep, ep.remote_addrs[i], send_base, SHARD_ALIGN,
                    recv_buf_rdma_offset + recv_offsets[i],
                    ep.remote_keys[i], ep.remote_base_addrs[i], imm, nullptr);
            }
            if (ret != 0) throw std::runtime_error("EFA RDMA write failed: " + std::string(fi_strerror(-ret)));
            writes_per_nic[0]++;
            total_writes++;
            continue;
        }

        // Shard this peer's data across NICs
        size_t total_bytes = static_cast<size_t>(send_sizes[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + SHARD_ALIGN - 1) & ~(SHARD_ALIGN - 1);

        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;

            auto& ep = transport.endpoint(n);

            // Proactive flow control: drain TX CQ if too many outstanding writes on this NIC
            while (writes_per_nic[n] - tx_completed[n] >= MAX_OUTSTANDING_PER_NIC) {
                struct fi_cq_data_entry drain[64];
                int polled = efa::poll_tx_cq(ep, drain, 64);
                if (polled > 0) {
                    tx_completed[n] += polled;
                    total_tx_completed += polled;
                }
                // Also poll RX to avoid deadlock
                struct fi_cq_data_entry rx_drain[64];
                int rx_polled = efa::poll_rx_cq(ep, rx_drain, 64);
                if (rx_polled > 0) {
                    g_posted_per_nic[n] -= rx_polled;  // Iter 40: track consumed receives
                }
                for (int j = 0; j < rx_polled; ++j) {
                    if (rx_drain[j].flags & FI_REMOTE_CQ_DATA) {
                        counters.record(static_cast<uint32_t>(rx_drain[j].data));
                    }
                }
            }

            void* src = send_base + send_offsets[i] + shard_offset;
            uint64_t dst_offset = recv_buf_rdma_offset + recv_offsets[i] + shard_offset;
            // Iter 44: First shard (n==0) carries token count in imm data if send_token_counts provided.
            // Other shards use normal count=1 encoding.
            uint32_t imm;
            if (send_token_counts && n == 0) {
                imm = efa::ImmCounterMap::encode_with_tokens(tag, rank, static_cast<uint16_t>((*send_token_counts)[i]));
            } else {
                imm = efa::ImmCounterMap::encode(tag, rank, 1);
            }

            int ret = efa::rdma_write_with_imm(
                ep,
                ep.remote_addrs[i],
                src,
                shard_len,
                dst_offset,
                ep.remote_keys[i],
                ep.remote_base_addrs[i],
                imm,
                nullptr);

            // Handle EAGAIN
            struct fi_cq_data_entry tx_entries[64];
            while (ret == -FI_EAGAIN) {
                int polled = efa::poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) {
                    tx_completed[n] += polled;
                    total_tx_completed += polled;
                }
                // Also poll RX on this NIC
                struct fi_cq_data_entry rx_entries[64];
                int rx_polled = efa::poll_rx_cq(ep, rx_entries, 64);
                if (rx_polled > 0) {
                    g_posted_per_nic[n] -= rx_polled;  // Iter 40: track consumed receives
                }
                for (int j = 0; j < rx_polled; ++j) {
                    if (rx_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        counters.record(static_cast<uint32_t>(rx_entries[j].data));
                    }
                }
                ret = efa::rdma_write_with_imm(
                    ep, ep.remote_addrs[i], src, shard_len,
                    dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                    imm, nullptr);
            }

            if (ret != 0) {
                throw std::runtime_error("EFA RDMA write failed: " + std::string(fi_strerror(-ret)));
            }
            writes_per_nic[n]++;
            total_writes++;
            shard_offset += shard_len;
        }
    }

    double t_rdma_writes = us_since(t_mark);

    // Wait for all writes to complete (TX CQ on all NICs) AND all incoming writes (RX CQ)
    auto start_time = std::chrono::high_resolution_clock::now();
    struct fi_cq_data_entry cq_entries[64];

    // Iter 44: Two-phase CQ wait when recv_token_counts is requested.
    // Phase 1: Wait until all remote peers have sent their token_count (first shard imm).
    // Phase 2: Compute exact expected shards from token counts, wait for all.
    // When recv_token_counts is NOT requested, use the traditional single-phase wait.
    bool use_imm_counts = (recv_token_counts != nullptr);
    int num_remote_peers = 0;
    for (int i = 0; i < num_peers; ++i) {
        if (i != rank) num_remote_peers++;
    }
    bool phase2_started = !use_imm_counts;  // Skip phase 1 if not using imm counts
    // For non-imm-count path: use rx_baseline + num_expected_recvs
    // For imm-count path: use absolute_expected (includes per-rank baselines)
    int exact_expected_recvs = use_imm_counts ? 0 : num_expected_recvs;
    // absolute_expected_total: total CQ count we expect after all entries arrive.
    // For non-imm-count path: rx_baseline + exact_expected_recvs
    // For imm-count path: sum(per_rank_cq_baseline) + exact_new_entries
    int absolute_expected_total = use_imm_counts ? 0 : (rx_baseline + exact_expected_recvs);

    while (total_tx_completed < total_writes ||
           (!phase2_started) ||
           (phase2_started && counters.get_total(tag) < absolute_expected_total)) {
        for (int n = 0; n < num_nics; ++n) {
            auto& ep = transport.endpoint(n);

            // Poll TX CQ
            if (tx_completed[n] < writes_per_nic[n]) {
                int polled = efa::poll_tx_cq(ep, cq_entries, 64);
                if (polled > 0) {
                    tx_completed[n] += polled;
                    total_tx_completed += polled;
                }
            }

            // Poll RX CQ
            {
                int polled = efa::poll_rx_cq(ep, cq_entries, 64);
                if (polled > 0) {
                    g_posted_per_nic[n] -= polled;  // track consumed receives
                }
                for (int j = 0; j < polled; ++j) {
                    if (cq_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        counters.record(static_cast<uint32_t>(cq_entries[j].data));
                    }
                }
                // Iter 44: Replenish receive buffers if running low.
                // Without this, the imm-count two-phase wait can consume all
                // pre-posted buffers before phase 2 determines the expected count,
                // causing a deadlock where remote peers' RDMA writes have no
                // receive buffers to land on.
                if (g_posted_per_nic[n] < 32) {
                    int to_post = target_per_nic - g_posted_per_nic[n];
                    for (int rp = 0; rp < to_post; ++rp) {
                        int ret = efa::post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
                        if (ret == 0) {
                            g_posted_per_nic[n]++;
                        } else {
                            break;  // EFA queue full or error
                        }
                    }
                }
            }
        }

        // Iter 44 FIX: Phase transition — check if all remote peers have sent token counts.
        // Use get_fresh_token_count() which compares the CQ count at which the token_count
        // was set against the per-rank baseline, ensuring only CURRENT epoch values are used.
        if (use_imm_counts && !phase2_started) {
            int known_peers = 0;
            for (int i = 0; i < num_peers; ++i) {
                if (i == rank) continue;
                if (counters.get_fresh_token_count(tag, i, per_rank_cq_baseline[i]) >= 0) {
                    known_peers++;
                }
            }
            if (known_peers >= num_remote_peers) {
                // All token counts received — compute absolute expected total.
                // Start from per-rank baselines (entries from BEFORE this call),
                // then add new entries expected from each peer.
                absolute_expected_total = 0;
                exact_expected_recvs = 0;  // for debug only
                for (int i = 0; i < num_peers; ++i) {
                    if (i == rank) continue;
                    absolute_expected_total += per_rank_cq_baseline[i];  // pre-existing entries
                    int tc = counters.get_fresh_token_count(tag, i, per_rank_cq_baseline[i]);
                    if (tc > 0) {
                        size_t total_bytes = static_cast<size_t>(tc) * packed_bytes_per_token;
                        size_t bytes_per_shard = ((total_bytes / num_nics) + SHARD_ALIGN - 1) & ~(SHARD_ALIGN - 1);
                        if (bytes_per_shard == 0) bytes_per_shard = SHARD_ALIGN;
                        size_t shard_offset = 0;
                        for (int sn = 0; sn < num_nics; ++sn) {
                            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
                            if (shard_len == 0) break;
                            absolute_expected_total++;
                            exact_expected_recvs++;
                            shard_offset += shard_len;
                        }
                    }
                    // tc==0: sender sends 1 notification shard
                    if (tc == 0) {
                        absolute_expected_total++;
                        exact_expected_recvs++;
                    }
                }
                phase2_started = true;
            }
        }

        // Periodic progress log (every 5 seconds) and timeout check
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        
        if (elapsed_s >= 5 && (elapsed_s % 5 == 0)) {
            // Throttle: only print once per second by checking ms
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
            if (elapsed_ms % 5000 < 50) {  // within first 50ms of 5-second boundary
                int known = 0;
                if (use_imm_counts) {
                    for (int i = 0; i < num_peers; ++i) {
                        if (i == rank) continue;
                        if (counters.get_fresh_token_count(tag, i, per_rank_cq_baseline[i]) >= 0) known++;
                    }
                }
                fprintf(stderr, "EFA a2a progress: rank=%d call#=%d tag=%u elapsed=%lds "
                        "tx=%d/%d rx_total=%d expected=%d phase2=%d known=%d/%d posted=[%d,%d]\n",
                        rank, this_call, static_cast<unsigned>(tag), (long)elapsed_s,
                        total_tx_completed, total_writes, counters.get_total(tag),
                        absolute_expected_total, (int)phase2_started,
                        known, num_remote_peers,
                        (num_nics > 0 ? g_posted_per_nic[0] : -1),
                        (num_nics > 1 ? g_posted_per_nic[1] : -1));
            }
        }
        
        if (elapsed_s > 60) {
            int known = 0;
            std::string missing_peers;
            if (use_imm_counts) {
                for (int i = 0; i < num_peers; ++i) {
                    if (i == rank) continue;
                    int tc = counters.get_fresh_token_count(tag, i, per_rank_cq_baseline[i]);
                    if (tc >= 0) {
                        known++;
                    } else {
                        if (!missing_peers.empty()) missing_peers += ",";
                        missing_peers += std::to_string(i);
                    }
                }
            }
            // Print per-peer CQ counts for this tag
            std::string per_peer_cq;
            for (int i = 0; i < num_peers; ++i) {
                if (i > 0) per_peer_cq += ",";
                per_peer_cq += std::to_string(counters.get_count(tag, i));
            }
            // Build per-rank baseline string for debugging (only for imm-count path)
            std::string baseline_str;
            if (use_imm_counts) {
                for (int i = 0; i < num_peers; ++i) {
                    if (i > 0) baseline_str += ",";
                    baseline_str += std::to_string(per_rank_cq_baseline[i]);
                }
            }
            // Build g_posted_per_nic string for debugging
            std::string posted_str;
            for (int n = 0; n < num_nics; ++n) {
                if (n > 0) posted_str += ",";
                posted_str += std::to_string(g_posted_per_nic[n]);
            }
            fprintf(stderr, "EFA all-to-all timeout: rank=%d, call#=%d, total_tx=%d/%d, rx=%d/%d (rx_base=%d), "
                    "num_nics=%d, tag=%u, use_imm=%d, phase2=%d, known_peers=%d/%d, "
                    "missing=[%s], per_peer_cq=[%s], per_rank_base=[%s], posted=[%s]\n",
                    rank, this_call, total_tx_completed, total_writes, counters.get_total(tag),
                    absolute_expected_total, rx_baseline, num_nics,
                    static_cast<unsigned>(tag), (int)use_imm_counts, (int)phase2_started, known, num_remote_peers,
                    missing_peers.c_str(), per_peer_cq.c_str(), baseline_str.c_str(), posted_str.c_str());
            throw std::runtime_error("EFA all-to-all timeout");
        }
    }

    double t_cq_wait = us_since(t_mark);

    // Iter 44 FIX: Extract recv_token_counts using freshness-aware API
    if (recv_token_counts) {
        counters.get_all_fresh_token_counts(tag, per_rank_cq_baseline, *recv_token_counts);
    }

    // Drain remaining TX CQ entries on all NICs
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport.endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = efa::poll_tx_cq(ep, drain_entries, 64);
        } while (drained > 0);
    }

    double t_drain_tx = us_since(t_mark);

    // Full device sync required: RDMA writes via DMA-BUF bypass GPU caches.
    // cudaStreamSynchronize is insufficient — it doesn't provide the full GPU memory
    // fence needed to ensure RDMA-written data is visible to subsequent GPU operations.
    CUDA_CHECK(cudaDeviceSynchronize());

    double t_dev_sync = us_since(t_mark);
    double t_total = std::chrono::duration<double, std::micro>(
        std::chrono::high_resolution_clock::now() - t_start).count();

    // Accumulate and print profiling
    if (profile_enabled) {
        profile_call_count++;
        if (warmup_remaining > 0) {
            warmup_remaining--;
        } else {
            accum_stream_sync += t_stream_sync;
            accum_drain_old += t_drain_old;
            accum_post_recv += t_post_recv;
            accum_rdma_writes += t_rdma_writes;
            accum_cq_wait += t_cq_wait;
            accum_drain_tx += t_drain_tx;
            accum_dev_sync += t_dev_sync;
            accum_total += t_total;
            accum_count++;

            if (accum_count >= profile_interval) {
                int n = accum_count;
                fprintf(stderr, "[EFA_PROFILE rank=%d call=%d] avg of %d: "
                        "stream_sync=%.1fus drain_old=%.1fus post_recv=%.1fus "
                        "rdma_writes=%.1fus cq_wait=%.1fus drain_tx=%.1fus "
                        "dev_sync=%.1fus TOTAL=%.1fus | "
                        "tx=%d rx_exp=%d nics=%d\n",
                        rank, profile_call_count, n,
                        accum_stream_sync / n, accum_drain_old / n, accum_post_recv / n,
                        accum_rdma_writes / n, accum_cq_wait / n, accum_drain_tx / n,
                        accum_dev_sync / n, accum_total / n,
                        total_writes, num_expected_recvs, num_nics);
                accum_stream_sync = accum_drain_old = accum_post_recv = 0;
                accum_rdma_writes = accum_cq_wait = accum_drain_tx = 0;
                accum_dev_sync = accum_total = 0;
                accum_count = 0;
            }
        }
    }
}

// ============================================================================
// EFA RDMA Count Exchange: scatter per-rank counts via RDMA (replaces NCCL)
// ============================================================================

std::tuple<std::vector<int32_t>, int, std::vector<int32_t>, int>
Buffer::efa_count_exchange(const torch::Tensor& send_counts_gpu,
                            int64_t cuda_event_ptr) {
    EP_HOST_ASSERT(efa_manager_ && efa_manager_->is_initialized());

    pybind11::gil_scoped_release release;

    int num_peers = num_ranks;

    // 2. Copy send_counts from GPU to host via comm_stream (not default stream!)
    //    Using comm_stream avoids blocking on the pack_only kernel running on default stream.
    //    The count kernel already completed (we synced count_event above).
    //    send_counts_gpu was written by count_only on default stream.
    //    We need comm_stream to wait for count_only before reading send_counts:
    //    the count_event was recorded on default stream, and we already EventSync'd it,
    //    so the data is in GPU memory. But comm_stream hasn't seen it yet.
    //    We need a stream-level dependency: record the count event on default stream,
    //    make comm_stream wait for it.
    if (cuda_event_ptr != 0) {
        cudaEvent_t event = reinterpret_cast<cudaEvent_t>(cuda_event_ptr);
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream.stream(), event, 0));
    } else {
        // Fallback: make comm_stream wait for default stream
        cudaEvent_t fallback_event;
        CUDA_CHECK(cudaEventCreate(&fallback_event));
        CUDA_CHECK(cudaEventRecord(fallback_event, nullptr));
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream.stream(), fallback_event, 0));
        CUDA_CHECK(cudaEventDestroy(fallback_event));
    }

    std::vector<int32_t> send_counts_host(num_peers);
    CUDA_CHECK(cudaMemcpyAsync(send_counts_host.data(), send_counts_gpu.data_ptr(),
                                num_peers * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                comm_stream.stream()));
    // Sync comm_stream to get send_counts on host
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));

    int total_send = 0;
    for (int i = 0; i < num_peers; ++i) total_send += send_counts_host[i];

    // 3. Copy send_counts into RDMA send buffer's count region (offset 0 in send half)
    //    Use comm_stream to avoid blocking on default stream's pack kernel.
    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;

    CUDA_CHECK(cudaMemcpyAsync(rdma_base, send_counts_host.data(),
                                num_peers * sizeof(int32_t), cudaMemcpyHostToDevice,
                                comm_stream.stream()));
    // Sync comm_stream to ensure data is in GPU RDMA buffer before NIC reads it
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));

    auto& transport = efa_manager_->transport();
    auto& counters = efa_manager_->counters();
    int num_nics = transport.num_nics();

    // Tag for this transfer — shares global counter with efa_all_to_all
    uint8_t tag = g_efa_transfer_tag++;
    uint8_t barrier_tag = g_efa_transfer_tag++;
    uint8_t next_tag = g_efa_transfer_tag++;
    int next_baseline = counters.get_total(next_tag);

    // Capture barrier baseline
    int barrier_baseline = counters.get_total(barrier_tag);

    // Drain any leftover RX CQ entries from previous operations.
    // Update shared g_posted_per_nic tracking.
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport.endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = efa::poll_rx_cq(ep, drain_entries, 64);
            if (drained > 0 && g_posted_per_nic_initialized) {
                g_posted_per_nic[n] -= drained;
            }
            for (int j = 0; j < drained; ++j) {
                if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                    counters.record(static_cast<uint32_t>(drain_entries[j].data));
                }
            }
        } while (drained > 0);
    }

    // Post receive buffers for barrier arrivals — update shared tracking.
    static thread_local char recv_dummy[64] = {};
    int target_per_nic = std::max((num_peers - 1) * 2, 64) + 16;  // generous for barrier + early next-op
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport.endpoint(n);
        for (int i = 0; i < target_per_nic; ++i) {
            int ret = efa::post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
            if (ret == -FI_EAGAIN) break;  // CQ full, enough posted
            if (ret == 0 && g_posted_per_nic_initialized) {
                g_posted_per_nic[n]++;
            }
        }
    }

    // Send zero-byte RDMA writes with immediate data to all remote peers via NIC 0
    auto& ep0 = transport.endpoint(0);
    int total_writes = 0;
    uint32_t imm = efa::ImmCounterMap::encode(barrier_tag, static_cast<uint8_t>(rank), 1);

    for (int i = 0; i < num_peers; ++i) {
        if (i == rank) continue;

        int ret = efa::rdma_write_with_imm(
            ep0, ep0.remote_addrs[i],
            rdma_buffer_ptr, 0,  // 0-byte write (just imm notification)
            0, ep0.remote_keys[i], ep0.remote_base_addrs[i],
            imm, nullptr);

        // Handle EAGAIN by draining TX CQ
        while (ret == -FI_EAGAIN) {
            struct fi_cq_data_entry tx_entries[64];
            efa::poll_tx_cq(ep0, tx_entries, 64);
            ret = efa::rdma_write_with_imm(
                ep0, ep0.remote_addrs[i],
                rdma_buffer_ptr, 0,
                0, ep0.remote_keys[i], ep0.remote_base_addrs[i],
                imm, nullptr);
        }
        total_writes++;
    }

    // Wait for all TX completions and all RX arrivals for barrier tag
    int tx_remaining = total_writes;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Expected barrier entries: one from each remote peer
    int num_barrier_expected = num_peers - 1;

    while (tx_remaining > 0 || counters.get_total(barrier_tag) < barrier_baseline + num_barrier_expected) {
        for (int n = 0; n < num_nics; ++n) {
            auto& ep = transport.endpoint(n);
            struct fi_cq_data_entry cq_entries[64];

            // Poll TX CQ (only on NIC 0 where we sent)
            if (n == 0 && tx_remaining > 0) {
                int polled = efa::poll_tx_cq(ep, cq_entries, 64);
                if (polled > 0) tx_remaining -= polled;
            }

            // Poll RX CQ on all NICs
            int rx_polled = efa::poll_rx_cq(ep, cq_entries, 64);
            for (int j = 0; j < rx_polled; ++j) {
                if (cq_entries[j].flags & FI_REMOTE_CQ_DATA) {
                    counters.record(static_cast<uint32_t>(cq_entries[j].data));
                }
            }
            // Replenish receive buffers consumed by this poll — update shared tracking.
            if (rx_polled > 0 && g_posted_per_nic_initialized) {
                g_posted_per_nic[n] -= rx_polled;
            }
            for (int i = 0; i < rx_polled; ++i) {
                int ret = efa::post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
                if (ret == 0 && g_posted_per_nic_initialized) {
                    g_posted_per_nic[n]++;
                }
            }
        }

        // Timeout check (60 seconds)
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 60) {
            fprintf(stderr, "efa_imm_barrier timeout: rank=%d tx_rem=%d rx=%d/%d tag=%u\n",
                    rank, tx_remaining,
                    counters.get_total(barrier_tag) - barrier_baseline, num_barrier_expected,
                    static_cast<unsigned>(barrier_tag));
            break;
        }
    }

    // Reset efa_all_to_all's posted_per_nic tracking by using a sentinel.
    // We consumed unknown number of recv buffers from the shared pool.
    // Set a flag so the next efa_all_to_all re-posts from scratch.
    // Implementation: store -1 in the static posted_per_nic vector's first element
    // as a "dirty" signal. Actually, we can't access it here...
    // Instead, we return the pre-reserved next tag/baseline and the caller
    // passes them to efa_all_to_all via override_tag/override_baseline.
    // The efa_all_to_all will handle its own recv pool management.

    // DEAD CODE: This function was part of the Phase 2/3 attempt (RDMA count exchange + barrier).
    // It is not used in the v3 dispatch path (which uses in-band token counts instead).
    // Return dummy values to satisfy the signature: (send_counts, next_tag, recv_counts, next_baseline)
    std::vector<int32_t> dummy_send(num_peers, 0);
    std::vector<int32_t> dummy_recv(num_peers, 0);
    return std::make_tuple(std::move(dummy_send), 0, std::move(dummy_recv), 0);
}

// Stub for efa_imm_barrier (dead code — Phase 3 RDMA barrier was abandoned)
std::pair<int, int> Buffer::efa_imm_barrier() {
    return {0, 0};
}

// ============================================================================
// LL Pipeline Implementation: GDRCopy-signaled async RDMA worker
//
// Architecture:
//   1. Python calls init_ll_pipeline() once to create the worker thread
//   2. For dispatch: Python launches pack kernel → calls start_dispatch(counts)
//      → worker thread detects pack_done flag, reads counts, issues RDMA writes
//      → worker signals recv_done when CQ complete
//   3. Python calls wait_dispatch() which spin-waits on command_done
//   4. Same pattern for combine
//
// Benefits over efa_all_to_all():
//   - No cudaStreamSynchronize before RDMA (worker already has GDR counts)
//   - No cudaDeviceSynchronize after RDMA (worker sets recv_done flag)
//   - Worker runs on dedicated CPU thread — doesn't block Python
//   - Worker polls GDR flags at ~1us granularity
// ============================================================================

void Buffer::init_ll_pipeline(int packed_bytes_per_token, int num_max_tokens_per_rank) {
    EP_HOST_ASSERT(efa_manager_ && efa_manager_->is_initialized());

    if (!ll_pipeline_) {
        ll_pipeline_ = std::make_unique<efa::LLPipelineWorker>();
    }

    // Configure the pipeline
    efa::LLPipelineConfig config;
    config.num_ranks = num_ranks;
    config.local_rank = rank;
    config.device_id = device_id;
    config.packed_bytes_per_token = packed_bytes_per_token;
    config.num_max_dispatch_tokens_per_rank = num_max_tokens_per_rank;

    // RDMA buffer layout: send half [0, half_rdma), recv half [half_rdma, num_rdma_bytes)
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;
    // Data offset within each half (skip count exchange area)
    // In the current layout, efa_all_to_all uses the RDMA buffer directly.
    // For pipeline mode, we use the same layout as ll_efa_dispatch_data_v3:
    //   send half: [0, half_rdma) — packed send data at offset 0
    //   recv half: [half_rdma, end) — packed recv data at offset data_offset
    // The Python caller sets data_offset in the cache.
    // For simplicity, data_offset = 0 within each half (no count exchange area in pipeline mode).
    uint64_t data_offset = 0;  // Packed data starts at beginning of each half

    config.half_rdma = half_rdma;
    config.data_offset = data_offset;
    config.slot_size = static_cast<uint64_t>(num_max_tokens_per_rank) * packed_bytes_per_token;

    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);
    config.rdma_base = rdma_base;
    config.send_data_base = rdma_base + data_offset;  // Send half starts at offset 0
    config.recv_data_base = rdma_base + half_rdma + data_offset;  // Recv half
    config.recv_data_rdma_offset = half_rdma + data_offset;  // For remote RDMA writes

    config.initialized = true;

    if (!ll_pipeline_initialized_) {
        // First init: create worker thread
        ll_pipeline_->init(&efa_manager_->transport(), &efa_manager_->counters(), config);
        ll_pipeline_->post_initial_recvs();
        ll_pipeline_initialized_ = true;
        fprintf(stderr, "[Rank %d] LL Pipeline initialized: packed_bpt=%d, max_tokens/rank=%d, "
                "slot_size=%lu, half_rdma=%lu\n",
                rank, packed_bytes_per_token, num_max_tokens_per_rank,
                config.slot_size, half_rdma);
    } else {
        // Update config (e.g., packed_bytes_per_token changed)
        ll_pipeline_->update_config(config);
    }
}

void Buffer::ll_pipeline_start_dispatch(const std::vector<int32_t>& send_counts,
                                         const std::vector<int32_t>& recv_counts) {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);

    // Write recv counts to GDR-mapped memory (worker reads these)
    for (size_t i = 0; i < recv_counts.size() && i < static_cast<size_t>(num_ranks); ++i) {
        ll_pipeline_->dispatch_recv_counts().write(i, recv_counts[i]);
    }

    // Write send counts to GDR-mapped memory
    for (size_t i = 0; i < send_counts.size() && i < static_cast<size_t>(num_ranks); ++i) {
        ll_pipeline_->dispatch_send_counts().write(i, send_counts[i]);
    }

    // Signal worker to start dispatch RDMA
    ll_pipeline_->start_dispatch();
}

void Buffer::ll_pipeline_wait_dispatch() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    // Release GIL while spin-waiting
    pybind11::gil_scoped_release release;
    ll_pipeline_->wait_dispatch_done();
}

void Buffer::ll_pipeline_start_combine(const std::vector<int32_t>& send_counts,
                                        const std::vector<int32_t>& recv_counts) {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);

    // Write counts to GDR-mapped memory
    ll_pipeline_->set_combine_send_counts(send_counts);
    ll_pipeline_->set_combine_recv_counts(recv_counts);

    // Signal worker to start combine RDMA
    ll_pipeline_->start_combine();
}

void Buffer::ll_pipeline_wait_combine() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    pybind11::gil_scoped_release release;
    ll_pipeline_->wait_combine_done();
}

void Buffer::ll_pipeline_barrier() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    pybind11::gil_scoped_release release;
    ll_pipeline_->efa_barrier();
}

std::tuple<int64_t, int64_t> Buffer::ll_pipeline_get_dispatch_gdr_ptrs() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    // Return (pack_done_device_ptr, send_counts_device_ptr) as int64
    // GPU kernel can write to pack_done_ptr to signal data is packed
    // GPU kernel can write send_counts to the GDR-mapped vector
    int64_t pack_done_ptr = reinterpret_cast<int64_t>(
        const_cast<uint8_t*>(ll_pipeline_->dispatch_pack_done().device_ptr()));
    int64_t send_counts_ptr = reinterpret_cast<int64_t>(
        ll_pipeline_->dispatch_send_counts().device_ptr());
    return {pack_done_ptr, send_counts_ptr};
}

std::tuple<int64_t, int64_t> Buffer::ll_pipeline_get_combine_gdr_ptrs() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    int64_t pack_done_ptr = reinterpret_cast<int64_t>(
        const_cast<uint8_t*>(ll_pipeline_->combine_pack_done().device_ptr()));
    int64_t send_counts_ptr = reinterpret_cast<int64_t>(
        ll_pipeline_->combine_send_counts().device_ptr());
    return {pack_done_ptr, send_counts_ptr};
}

void Buffer::efa_barrier() {
    if (ll_pipeline_initialized_ && ll_pipeline_) {
        pybind11::gil_scoped_release release;
        ll_pipeline_->efa_barrier();
    }
}

// ============================================================================
// Iter 43: Fused RDMA count exchange + data transfer for LL dispatch.
// Replaces NCCL all_to_all_single for count exchange with EFA RDMA.
// Uses efa_all_to_all with uniform sizes for count exchange (small messages),
// then efa_all_to_all again for data transfer.
//
// Count exchange layout:
//   Send: our send_counts[num_ranks] at offset 0 in send half
//   Recv: recv_half has num_ranks rows of num_ranks ints each
//         recv_half[peer * num_ranks + our_rank] = how many tokens peer sends us
//
// Flow:
//   1. comm_stream waits for count_event → D2H send_counts → H2D into RDMA buf
//   2. efa_all_to_all for counts (uniform sizes)
//   3. NC-load recv_counts from RDMA buffer
//   4. Compute offsets, efa_all_to_all for data transfer
// ============================================================================
std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
Buffer::ll_efa_rdma_dispatch(const torch::Tensor& send_counts_gpu,
                              int64_t count_event_ptr,
                              int64_t pack_done_event_ptr,
                              const torch::Tensor& send_buf,
                              const torch::Tensor& recv_buf,
                              int packed_bytes_per_token,
                              int64_t slot_size) {
    EP_HOST_ASSERT(efa_manager_ && efa_manager_->is_initialized());

    // NOTE: Do NOT release GIL here — efa_all_to_all (called below) has its
    // own gil_scoped_release and nesting them triggers an assertion failure.

    int num_peers = num_ranks;
    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;

    // --- Step 1: Make comm_stream wait for count_event, D2H copy send_counts ---
    if (count_event_ptr != 0) {
        cudaEvent_t event = reinterpret_cast<cudaEvent_t>(count_event_ptr);
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream.stream(), event, 0));
    }

    std::vector<int32_t> send_counts_host(num_peers);
    CUDA_CHECK(cudaMemcpyAsync(send_counts_host.data(), send_counts_gpu.data_ptr(),
                                num_peers * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                comm_stream.stream()));

    // --- Step 2: H2D copy send_counts into RDMA send buffer's count region ---
    // Count region is at offset 0 in the send half of RDMA buffer.
    CUDA_CHECK(cudaMemcpyAsync(rdma_base, send_counts_host.data(),
                                num_peers * sizeof(int32_t), cudaMemcpyHostToDevice,
                                comm_stream.stream()));

    // Sync comm_stream: we need send_counts on host (for offset computation later)
    // and in RDMA buffer (for NIC to read).
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));

    // --- Step 3: Count exchange via efa_all_to_all ---
    // Each rank sends its full send_counts array (num_ranks * 4 bytes) to every peer.
    // Each rank receives every peer's send_counts array.
    // This reuses the battle-tested efa_all_to_all CQ/receive handling.
    //
    // IMPORTANT: Pre-reserve the data transfer's tag BEFORE the count exchange runs.
    // Without this, CQ polling during the count exchange picks up early arrivals
    // from other ranks' data transfers (same tag), which inflates the baseline
    // and causes a permanent deficit in the expected completion count.
    uint8_t data_tag = g_efa_transfer_tag++;
    // Capture data tag's baseline NOW, before any CQ polling happens.
    auto& counters = efa_manager_->counters();
    int data_baseline = counters.get_total(data_tag);
    // Count exchange will auto-allocate its own tag inside efa_all_to_all.

    size_t array_size = num_peers * sizeof(int32_t);
    size_t count_recv_total = num_peers * array_size;

    // Create tensor views into RDMA buffer for count exchange
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_id);
    torch::Tensor count_send_t = torch::from_blob(rdma_base, {(int64_t)array_size}, opts);
    torch::Tensor count_recv_t = torch::from_blob(rdma_base + half_rdma, {(int64_t)count_recv_total}, opts);

    // Uniform sizes: every peer gets array_size bytes
    // recv_offsets[i] = rank * array_size: tells peer i to place our data at row [rank]
    std::vector<int64_t> count_send_sizes(num_peers);
    std::vector<int64_t> count_send_offsets(num_peers);
    std::vector<int64_t> count_recv_sizes(num_peers);
    std::vector<int64_t> count_recv_offsets(num_peers);
    for (int i = 0; i < num_peers; ++i) {
        if (i == rank) {
            count_send_sizes[i] = 0;
            count_recv_sizes[i] = 0;
        } else {
            count_send_sizes[i] = static_cast<int64_t>(array_size);
            count_recv_sizes[i] = static_cast<int64_t>(array_size);
        }
        count_send_offsets[i] = 0;  // All peers read from same source
        // On the remote, we want our data at row [rank], i.e. offset rank * array_size
        count_recv_offsets[i] = static_cast<int64_t>(rank) * static_cast<int64_t>(array_size);
    }

    // efa_all_to_all handles GIL release, CQ polling, receives, etc.
    efa_all_to_all(count_send_t, count_send_sizes, count_send_offsets,
                   count_recv_t, count_recv_sizes, count_recv_offsets);

    // --- Step 4: Make comm_stream wait for pack_done (overlap with count exchange) ---
    if (pack_done_event_ptr != 0) {
        cudaEvent_t pack_event = reinterpret_cast<cudaEvent_t>(pack_done_event_ptr);
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream.stream(), pack_event, 0));
    }

    // --- Step 5: Read recv_counts from RDMA buffer ---
    // RDMA writes bypass GPU L2 cache. Use cudaDeviceSynchronize + NC loads.
    // Layout: recv_half[peer * array_size + our_rank * 4] has how many tokens peer sends us.
    CUDA_CHECK(cudaDeviceSynchronize());

    // NC gather kernel reads scattered counts
    torch::Tensor recv_counts_gpu = torch::empty({num_peers}, torch::TensorOptions()
        .dtype(torch::kInt32).device(torch::kCUDA, device_id));

    {
        const int32_t* src = reinterpret_cast<const int32_t*>(rdma_base + half_rdma);
        int32_t* dst = recv_counts_gpu.data_ptr<int32_t>();
        efa_kernels::nc_gather_counts(src, dst, num_peers, num_peers, rank, comm_stream.stream());
    }

    // Self-count: we didn't RDMA-write to ourselves
    CUDA_CHECK(cudaMemcpyAsync(
        recv_counts_gpu.data_ptr<int32_t>() + rank,
        send_counts_host.data() + rank,
        sizeof(int32_t), cudaMemcpyHostToDevice, comm_stream.stream()));

    // D2H the recv_counts
    std::vector<int32_t> recv_counts_host(num_peers);
    CUDA_CHECK(cudaMemcpyAsync(recv_counts_host.data(), recv_counts_gpu.data_ptr(),
                                num_peers * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                comm_stream.stream()));
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));

    // --- Step 7: Compute offsets and call efa_all_to_all for data transfer ---
    int64_t total_send = 0, total_recv = 0;
    std::vector<int64_t> send_sizes(num_peers);
    std::vector<int64_t> send_offsets(num_peers);
    std::vector<int64_t> recv_sizes(num_peers);
    std::vector<int64_t> recv_offsets(num_peers);

    int64_t offset = 0;
    for (int i = 0; i < num_peers; ++i) {
        total_send += send_counts_host[i];
        total_recv += recv_counts_host[i];
        send_sizes[i] = static_cast<int64_t>(send_counts_host[i]) * packed_bytes_per_token;
        send_offsets[i] = offset;
        offset += send_sizes[i];
        recv_sizes[i] = static_cast<int64_t>(recv_counts_host[i]) * packed_bytes_per_token;
        recv_offsets[i] = static_cast<int64_t>(rank) * slot_size;
    }

    if (total_send > 0 || total_recv > 0) {
        // Use pre-reserved data_tag and data_baseline to avoid race with count exchange CQ polling
        efa_all_to_all(send_buf, send_sizes, send_offsets, recv_buf, recv_sizes, recv_offsets,
                       static_cast<int>(data_tag), data_baseline);
    }

    return std::make_tuple(std::move(send_counts_host), std::move(recv_counts_host),
                           total_send, total_recv);
}

// ============================================================================
// Iter 39: Fused LL dispatch data transfer
// Computes per-rank sizes/offsets from counts and calls efa_all_to_all internally.
// Eliminates ~35us of Python-side offset computation overhead.
// ============================================================================
void Buffer::ll_efa_dispatch_data(const std::vector<int32_t>& send_counts,
                                   const std::vector<int32_t>& recv_counts,
                                   const torch::Tensor& send_buf,
                                   const torch::Tensor& recv_buf,
                                   int packed_bytes_per_token,
                                   int64_t slot_size) {
    int num_ranks = static_cast<int>(send_counts.size());

    // Compute send sizes and offsets (contiguous packing)
    std::vector<int64_t> send_sizes(num_ranks);
    std::vector<int64_t> send_offsets(num_ranks);
    int64_t offset = 0;
    for (int i = 0; i < num_ranks; ++i) {
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * packed_bytes_per_token;
        send_offsets[i] = offset;
        offset += send_sizes[i];
    }

    // Compute recv sizes and offsets (slot-based: every peer writes to our rank's slot in THEIR buffer)
    // recv_offsets[i] = rank * slot_size means: when we send to peer i, the data lands at
    // offset (our_rank * slot_size) in peer i's recv buffer
    std::vector<int64_t> recv_sizes(num_ranks);
    std::vector<int64_t> recv_offsets(num_ranks);
    for (int i = 0; i < num_ranks; ++i) {
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * packed_bytes_per_token;
        recv_offsets[i] = static_cast<int64_t>(rank) * slot_size;
    }

    // Call the existing efa_all_to_all (handles stream sync, RDMA writes, CQ polling)
    efa_all_to_all(send_buf, send_sizes, send_offsets, recv_buf, recv_sizes, recv_offsets);
}

// ============================================================================
// Iter 42: Fused LL dispatch data transfer v2 — reads counts from pinned CPU tensors
// Eliminates Python-side nccl_stream.synchronize() + .tolist() + sum() overhead.
// The comm_stream sync inside efa_all_to_all waits for NCCL + D2H + pack_done.
// Returns (send_counts, recv_counts, total_send, total_recv).
// ============================================================================
std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
Buffer::ll_efa_dispatch_data_v2(const torch::Tensor& send_counts_pin,
                                 const torch::Tensor& recv_counts_pin,
                                 const torch::Tensor& send_buf,
                                 const torch::Tensor& recv_buf,
                                 int packed_bytes_per_token,
                                 int64_t slot_size) {
    // efa_all_to_all's internal cudaStreamSynchronize(comm_stream) will wait for:
    // - NCCL all_to_all_single (count exchange)
    // - D2H copies of counts to pinned tensors
    // - pack_done event (pack kernel on default stream)
    // After that sync, we can safely read from pinned CPU tensors.

    // Read counts from pinned CPU memory (after comm_stream sync inside efa_all_to_all)
    // BUT: we need counts BEFORE calling efa_all_to_all to compute sizes/offsets.
    // So we must sync comm_stream first, then read counts, then call efa_all_to_all.
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));

    const int32_t* send_ptr = send_counts_pin.data_ptr<int32_t>();
    const int32_t* recv_ptr = recv_counts_pin.data_ptr<int32_t>();
    int nr = num_ranks;

    std::vector<int32_t> send_counts(send_ptr, send_ptr + nr);
    std::vector<int32_t> recv_counts(recv_ptr, recv_ptr + nr);

    int64_t total_send = 0, total_recv = 0;
    std::vector<int64_t> send_sizes(nr);
    std::vector<int64_t> send_offsets(nr);
    std::vector<int64_t> recv_sizes(nr);
    std::vector<int64_t> recv_offsets(nr);

    int64_t offset = 0;
    for (int i = 0; i < nr; ++i) {
        total_send += send_counts[i];
        total_recv += recv_counts[i];
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * packed_bytes_per_token;
        send_offsets[i] = offset;
        offset += send_sizes[i];
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * packed_bytes_per_token;
        recv_offsets[i] = static_cast<int64_t>(rank) * slot_size;
    }

    if (total_send > 0 || total_recv > 0) {
        efa_all_to_all(send_buf, send_sizes, send_offsets, recv_buf, recv_sizes, recv_offsets);
    }

    return std::make_tuple(std::move(send_counts), std::move(recv_counts), total_send, total_recv);
}

// ============================================================================
// Iter 44: Fused LL dispatch data transfer v3 — NO NCCL count exchange
// Token counts sent in-band via EFA imm data (encode_with_tokens).
// Saves ~240us by eliminating NCCL all_to_all_single.
// ============================================================================
std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
Buffer::ll_efa_dispatch_data_v3(const torch::Tensor& send_counts_pin,
                                 const torch::Tensor& send_buf,
                                 const torch::Tensor& recv_buf,
                                 const torch::Tensor& recv_counts_gpu,
                                 int packed_bytes_per_token,
                                 int64_t slot_size) {
    // comm_stream has:
    // - wait(count_event) → D2H copy of send_counts → wait(pack_done_event)
    // The cudaStreamSynchronize inside efa_all_to_all waits for all of these.
    // After sync, we can safely read from pinned CPU memory and GPU data is ready.
    // NCCL all_to_all_single (on default stream) also syncs before we reach here.
    CUDA_CHECK(cudaStreamSynchronize(comm_stream.stream()));
    // Also sync default stream: back-to-back dispatches may have unpack kernels
    // from the previous dispatch still reading the RDMA recv buffer on the default
    // stream. We must wait for those to finish before posting new EFA RDMA writes
    // that overwrite the recv buffer.
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    const int32_t* send_ptr = send_counts_pin.data_ptr<int32_t>();
    const int32_t* recv_ptr = recv_counts_gpu.data_ptr<int32_t>();
    int nr = num_ranks;

    std::vector<int32_t> send_counts(send_ptr, send_ptr + nr);
    // recv_counts already populated by NCCL all-to-all in Python.
    // Read from GPU tensor (which was filled by dist.all_to_all_single).
    std::vector<int32_t> recv_counts(nr);
    CUDA_CHECK(cudaMemcpy(recv_counts.data(), recv_ptr, nr * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int64_t total_send = 0;
    int64_t total_recv = 0;
    std::vector<int64_t> send_sizes(nr);
    std::vector<int64_t> send_offsets(nr);
    std::vector<int64_t> recv_sizes(nr);
    std::vector<int64_t> recv_offsets(nr);

    int64_t offset = 0;
    for (int i = 0; i < nr; ++i) {
        total_send += send_counts[i];
        total_recv += recv_counts[i];
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * packed_bytes_per_token;
        send_offsets[i] = offset;
        offset += send_sizes[i];
        // Recv sizes now known from NCCL count exchange.
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * packed_bytes_per_token;
        recv_offsets[i] = static_cast<int64_t>(rank) * slot_size;
    }

    // Call efa_all_to_all WITHOUT imm counts — recv_sizes are known from NCCL.
    // This avoids the two-phase wait and freshness check entirely.
    if (nr > 1) {
        efa_all_to_all(send_buf, send_sizes, send_offsets, recv_buf, recv_sizes, recv_offsets);
    }

    // recv_counts_gpu already has the correct values from NCCL — no need to copy again.

    return std::make_tuple(std::move(send_counts), std::move(recv_counts), total_send, total_recv);
}

// ============================================================================
// Iter 42: Fused LL combine data transfer v2 — reads counts from pinned CPU tensors
// ============================================================================
std::tuple<std::vector<int32_t>, std::vector<int32_t>, int64_t, int64_t>
Buffer::ll_efa_combine_data_v2(const torch::Tensor& send_counts_pin,
                                const torch::Tensor& recv_counts_pin,
                                const torch::Tensor& send_buf,
                                const torch::Tensor& recv_buf,
                                int hidden,
                                int bytes_per_elem,
                                int64_t combine_slot_size) {
    // For combine: send_counts_pin has dispatch's recv_counts (what we received, now sending back)
    //              recv_counts_pin has dispatch's send_counts (what we sent, now receiving back)
    // No comm_stream sync needed here — dist.barrier() already synchronized all streams.
    // But we still need to read counts from pinned CPU memory.
    const int32_t* send_ptr = send_counts_pin.data_ptr<int32_t>();
    const int32_t* recv_ptr = recv_counts_pin.data_ptr<int32_t>();
    int nr = num_ranks;

    std::vector<int32_t> send_counts(send_ptr, send_ptr + nr);
    std::vector<int32_t> recv_counts(recv_ptr, recv_ptr + nr);

    int64_t total_send = 0, total_recv = 0;
    int64_t elems_per_token = static_cast<int64_t>(hidden) * bytes_per_elem;
    std::vector<int64_t> send_sizes(nr);
    std::vector<int64_t> send_offsets(nr);
    std::vector<int64_t> recv_sizes(nr);
    std::vector<int64_t> recv_offsets(nr);

    int64_t offset = 0;
    for (int i = 0; i < nr; ++i) {
        total_send += send_counts[i];
        total_recv += recv_counts[i];
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * elems_per_token;
        send_offsets[i] = offset;
        offset += send_sizes[i];
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * elems_per_token;
        recv_offsets[i] = static_cast<int64_t>(rank) * combine_slot_size;
    }

    if (total_send > 0 || total_recv > 0) {
        efa_all_to_all(send_buf, send_sizes, send_offsets, recv_buf, recv_sizes, recv_offsets);
    }

    return std::make_tuple(std::move(send_counts), std::move(recv_counts), total_send, total_recv);
}

// ============================================================================
// Iter 39: Fused LL combine data transfer
// ============================================================================
void Buffer::ll_efa_combine_data(const std::vector<int32_t>& send_counts,
                                   const std::vector<int32_t>& recv_counts,
                                   const torch::Tensor& send_buf,
                                   const torch::Tensor& recv_buf,
                                   int hidden,
                                   int bytes_per_elem,
                                   int64_t combine_slot_size,
                                   int override_tag,
                                   int override_baseline) {
    // Iter 45: Make comm_stream wait for the default stream so that
    // gather+unsort kernels (launched async on default stream) complete
    // before efa_all_to_all sends RDMA writes. Previously, the synchronous
    // dist.barrier() implicitly synced the default stream. Now that the
    // barrier is async and overlapped with gather+unsort, we need an
    // explicit event-based dependency.
    {
        cudaEvent_t ev;
        CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev, nullptr));  // record on default stream
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream.stream(), ev, 0));
        CUDA_CHECK(cudaEventDestroy(ev));
    }

    int num_ranks = static_cast<int>(send_counts.size());
    int64_t elems_per_token = static_cast<int64_t>(hidden) * bytes_per_elem;

    // Combine: send_counts[i] = recv_counts_from_dispatch[i] (we're sending back
    // what we received in dispatch). recv_counts[i] = send_counts_from_dispatch[i]
    // (we're receiving back what we sent in dispatch).
    std::vector<int64_t> send_sizes(num_ranks);
    std::vector<int64_t> send_offsets(num_ranks);
    int64_t offset = 0;
    for (int i = 0; i < num_ranks; ++i) {
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * elems_per_token;
        send_offsets[i] = offset;
        offset += send_sizes[i];
    }

    std::vector<int64_t> recv_sizes(num_ranks);
    std::vector<int64_t> recv_offsets(num_ranks);
    for (int i = 0; i < num_ranks; ++i) {
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * elems_per_token;
        recv_offsets[i] = static_cast<int64_t>(rank) * combine_slot_size;
    }

    efa_all_to_all(send_buf, send_sizes, send_offsets, recv_buf, recv_sizes, recv_offsets,
                   override_tag, override_baseline);
}

// ============================================================================
// Iter 49: Cooperative Kernel RDMA Transfers
//
// These use efa_all_to_all with coop-specific token stride and buffer layout.
// ============================================================================

void Buffer::coop_efa_dispatch_rdma(
    const std::vector<int32_t>& send_counts,
    const std::vector<int32_t>& recv_counts,
    int coop_token_stride,
    int max_tokens_per_rank
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    int nr = num_ranks;
    int64_t slot_size = static_cast<int64_t>(max_tokens_per_rank) * coop_token_stride;

    // No explicit sync needed: caller ensures cooperative kernel completed via stream.synchronize()
    // and NCCL all_to_all_single acts as cross-rank barrier.

    // Build send/recv tensors from RDMA buffer halves
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto send_buf = torch::from_blob(rdma_buffer_ptr, {static_cast<int64_t>(half_rdma)}, options);
    auto recv_buf = torch::from_blob(
        static_cast<uint8_t*>(rdma_buffer_ptr) + half_rdma,
        {static_cast<int64_t>(half_rdma)}, options);

    // Compute per-rank send sizes and offsets (contiguous per-rank in send_buffer)
    std::vector<int64_t> send_sizes(nr), send_offsets(nr);
    std::vector<int64_t> recv_sizes(nr), recv_offsets(nr);
    int64_t s_off = 0;
    for (int i = 0; i < nr; ++i) {
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * coop_token_stride;
        send_offsets[i] = s_off;
        s_off += send_sizes[i];
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * coop_token_stride;
        // Slot-based recv: data from sender rank i goes into slot i
        recv_offsets[i] = static_cast<int64_t>(i) * slot_size;
    }

    // Issue RDMA
    if (nr > 1) {
        efa_all_to_all(send_buf, send_sizes, send_offsets,
                       recv_buf, recv_sizes, recv_offsets);
    }
}

void Buffer::coop_efa_combine_rdma(
    const std::vector<int32_t>& send_counts,
    const std::vector<int32_t>& recv_counts,
    int combine_token_dim,
    const std::vector<int64_t>& dst_group_offsets_tokens
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    int nr = num_ranks;

    // Sync all GPU streams before RDMA (for L2 coherence safety)
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    // Build send/recv tensors from RDMA buffer halves
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    auto send_buf = torch::from_blob(rdma_buffer_ptr, {static_cast<int64_t>(half_rdma)}, options);
    auto recv_buf = torch::from_blob(
        static_cast<uint8_t*>(rdma_buffer_ptr) + half_rdma,
        {static_cast<int64_t>(half_rdma)}, options);

    // Combine RDMA layout:
    // Send: combine_send kernel packs tokens contiguously per destination rank.
    //       src_group_offset[r] = cumulative send tokens for ranks < r.
    // Recv: tokens must land at expert_offsets-based positions.
    //       dst_group_offset[r] = cumulative recv tokens from ranks < r = prefix_sum(recv_counts).
    //       (Same as expert_offsets for DP=1, NODE=1: each rank owns consecutive experts.)

    std::vector<int64_t> send_sizes(nr), send_offsets(nr);
    std::vector<int64_t> recv_sizes(nr), recv_offsets(nr);
    int64_t s_off = 0;
    for (int i = 0; i < nr; ++i) {
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * combine_token_dim;
        send_offsets[i] = s_off;
        s_off += send_sizes[i];
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * combine_token_dim;
        recv_offsets[i] = dst_group_offsets_tokens[i] * combine_token_dim;
    }

    if (nr > 1) {
        efa_all_to_all(send_buf, send_sizes, send_offsets,
                       recv_buf, recv_sizes, recv_offsets);
    }
}

// ============================================================================
// Iter 48: Cooperative Kernel Wrappers
//
// These wrap the cooperative kernel launch functions with Buffer-level state
// management: scratch buffer allocation, GDR flag management, and pointer
// computation from the RDMA buffer layout.
// ============================================================================

static inline size_t round_up_sz(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void Buffer::coop_init(int max_tokens, int num_topk, int num_experts,
                        int max_recv_tokens) {
    // Note: Does not require ll_pipeline_ — just allocates CUDA memory + GDR flags.
    // Pipeline dependency will be needed when wiring up the actual coop data flow.

    auto& s = coop_scratch_;
    if (s.initialized &&
        max_tokens <= s.max_tokens_allocated &&
        num_topk <= s.max_topk_allocated &&
        num_experts <= s.num_experts_allocated &&
        max_recv_tokens <= s.max_recv_tokens_allocated) {
        // Already big enough
        return;
    }

    // Free old allocations if resizing (only cudaMalloc'd pointers)
    if (s.token_offset) cudaFree(s.token_offset);
    if (s.expert_offsets) cudaFree(s.expert_offsets);
    if (s.grid_counter) cudaFree(s.grid_counter);
    if (s.grid_counter_2) cudaFree(s.grid_counter_2);
    if (s.sync_counter) cudaFree(s.sync_counter);
    if (s.combine_token_counter) cudaFree(s.combine_token_counter);
    // GdrVec members auto-cleanup via destructor (no need to free)

    // Allocate GPU-only buffers (not CPU-accessed on critical path)
    CUDA_CHECK(cudaMalloc(&s.token_offset, max_tokens * num_topk * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s.expert_offsets, num_experts * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s.grid_counter, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s.grid_counter_2, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s.sync_counter, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&s.combine_token_counter, sizeof(uint32_t)));

    // Allocate GDR-mapped metadata arrays (Iter 53)
    // These use GdrVec so the CPU worker can read/write via MMIO (~1us)
    // instead of synchronous cudaMemcpy (~5-15us per call).
    int experts_per_rank = (num_experts + num_ranks - 1) / num_ranks;
    s.gdr_num_routed.init(num_experts);
    s.gdr_source_rank.init(max_recv_tokens);
    s.gdr_source_offset.init(max_recv_tokens);
    s.gdr_padded_index.init(max_recv_tokens);
    s.gdr_tokens_per_expert.init(experts_per_rank);
    s.gdr_num_recv_tokens.init(2);
    s.gdr_combine_send_offset.init(max_recv_tokens);

    // Set legacy raw pointers as aliases to GdrVec device_ptr() for kernel launches
    s.num_routed = s.gdr_num_routed.device_ptr();
    s.source_rank = s.gdr_source_rank.device_ptr();
    s.source_offset = s.gdr_source_offset.device_ptr();
    s.padded_index = s.gdr_padded_index.device_ptr();
    s.tokens_per_expert = s.gdr_tokens_per_expert.device_ptr();
    s.num_recv_tokens = s.gdr_num_recv_tokens.device_ptr();
    s.combine_send_offset = s.gdr_combine_send_offset.device_ptr();

    // Zero-initialize counters
    CUDA_CHECK(cudaMemset(s.grid_counter, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(s.grid_counter_2, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(s.sync_counter, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(s.combine_token_counter, 0, sizeof(uint32_t)));

    // NVLink ptrs: for NODE_SIZE=1, allocate dummy GPU arrays with nullptr entries.
    // The cooperative kernels unconditionally read recv_ptrs[0], so we can't pass a null pointer.
    if (!s.sync_ptrs) {
        uint32_t* dummy_sync = nullptr;
        CUDA_CHECK(cudaMalloc(&s.sync_ptrs, sizeof(uint32_t*)));
        CUDA_CHECK(cudaMemcpy(s.sync_ptrs, &dummy_sync, sizeof(uint32_t*), cudaMemcpyHostToDevice));
    }
    if (!s.recv_ptrs) {
        uint8_t* dummy_recv = nullptr;
        CUDA_CHECK(cudaMalloc(&s.recv_ptrs, sizeof(uint8_t*)));
        CUDA_CHECK(cudaMemcpy(s.recv_ptrs, &dummy_recv, sizeof(uint8_t*), cudaMemcpyHostToDevice));
    }
    if (!s.send_ptrs) {
        uint8_t* dummy_send = nullptr;
        CUDA_CHECK(cudaMalloc(&s.send_ptrs, sizeof(uint8_t*)));
        CUDA_CHECK(cudaMemcpy(s.send_ptrs, &dummy_send, sizeof(uint8_t*), cudaMemcpyHostToDevice));
    }

    // Initialize GDR flags
    if (!s.initialized) {
        s.dispatch_route_done.init();
        s.dispatch_pack_done.init();
        s.tx_ready.init();
        s.num_recv_tokens_flag.init();
        s.dispatch_recv_flag.init();
        s.dispatch_recv_done.init();
        s.combine_send_done.init();
        s.combine_recv_flag.init();
        s.combine_recv_done.init();

        // Initialize GDR counters (Iter 53)
        // dispatch_rdma_counter auto-sets dispatch_recv_flag when RDMA completions reach target
        // combine_rdma_counter auto-sets combine_recv_flag when RDMA completions reach target
        s.dispatch_rdma_counter.init(&s.dispatch_recv_flag);
        s.combine_rdma_counter.init(&s.combine_recv_flag);

        // Clear all flags
        s.dispatch_route_done.clear();
        s.dispatch_pack_done.clear();
        s.tx_ready.clear();
        s.num_recv_tokens_flag.clear();
        s.dispatch_recv_flag.clear();
        s.dispatch_recv_done.clear();
        s.combine_send_done.clear();
        s.combine_recv_flag.clear();
        s.combine_recv_done.clear();

        // Pre-set tx_ready = 1 (send buffer available for first iteration)
        s.tx_ready.set(1);
    }

    s.max_tokens_allocated = max_tokens;
    s.max_topk_allocated = num_topk;
    s.num_experts_allocated = num_experts;
    s.max_recv_tokens_allocated = max_recv_tokens;
    s.initialized = true;

    fprintf(stderr, "[Rank %d] Coop scratch initialized (GdrVec): max_tokens=%d, topk=%d, "
            "num_experts=%d, max_recv=%d, experts_per_rank=%d\n",
            rank, max_tokens, num_topk, num_experts, max_recv_tokens, experts_per_rank);
}

std::vector<int64_t> Buffer::coop_get_ptrs() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;
    // Return all device pointers as int64 for Python
    return {
        reinterpret_cast<int64_t>(s.token_offset),
        reinterpret_cast<int64_t>(s.expert_offsets),
        reinterpret_cast<int64_t>(s.num_routed),
        reinterpret_cast<int64_t>(s.grid_counter),
        reinterpret_cast<int64_t>(s.grid_counter_2),
        reinterpret_cast<int64_t>(s.sync_counter),
        // GDR flag device pointers
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.dispatch_route_done.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.tx_ready.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.num_recv_tokens_flag.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.dispatch_recv_flag.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.dispatch_recv_done.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.combine_send_done.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.combine_recv_flag.device_ptr())),
        reinterpret_cast<int64_t>(const_cast<uint8_t*>(s.combine_recv_done.device_ptr())),
    };
}

int Buffer::coop_dispatch_send(
    const torch::Tensor& x,
    const std::optional<torch::Tensor>& x_scales,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    int num_experts,
    int num_tokens,
    int hidden_dim,
    int hidden_dim_scale,
    int dp_size,
    int64_t stream_ptr
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;

    // Reset grid counter
    CUDA_CHECK(cudaMemsetAsync(s.grid_counter, 0, sizeof(uint32_t),
                               (cudaStream_t)stream_ptr));

    // Determine element sizes from tensor dtypes
    size_t x_elemsize = x.element_size();
    size_t x_scale_elemsize = x_scales.has_value() ? x_scales->element_size() : 0;

    // Compute number of SMs for cooperative launch
    int num_sms = num_device_sms;  // Use all SMs
    size_t num_blocks = std::min(num_sms, std::max(1, num_tokens));

    return coop::a2a_dispatch_send(
        num_blocks,
        static_cast<size_t>(hidden_dim),
        static_cast<size_t>(hidden_dim_scale),
        static_cast<size_t>(num_experts),
        static_cast<size_t>(topk_idx.size(1)),
        0,  // max_private_tokens = 0 for NODE_SIZE=1 (no NVLink private writes)
        static_cast<size_t>(rank),
        static_cast<size_t>(num_ranks),  // dp_size = num_ranks → dp_group = rank/num_ranks = 0; num_routed is per-GPU local buffer
        1,  // node_size = 1 (all traffic through EFA for first pass)
        static_cast<size_t>(num_ranks),
        static_cast<size_t>(num_tokens),
        nullptr,  // bound_m_ptr (no dynamic token count)
        reinterpret_cast<const uint8_t*>(x.data_ptr()),
        x_elemsize,
        x.stride(0) * x_elemsize,
        x_scales.has_value() ? reinterpret_cast<const uint8_t*>(x_scales->data_ptr()) : nullptr,
        x_scale_elemsize,
        x_scales.has_value() ? x_scales->stride(1) * x_scale_elemsize : 0,
        x_scales.has_value() ? x_scales->stride(0) * x_scale_elemsize : 0,
        reinterpret_cast<const int32_t*>(topk_idx.data_ptr()),
        topk_idx.stride(0),
        reinterpret_cast<const float*>(topk_weights.data_ptr()),
        topk_weights.stride(0),
        s.token_offset,
        s.num_routed,
        s.expert_offsets,
        const_cast<uint8_t*>(s.dispatch_route_done.device_ptr()),
        // dispatch_send_done: use coop_scratch's own dispatch_pack_done flag
        const_cast<uint8_t*>(s.dispatch_pack_done.device_ptr()),
        const_cast<uint8_t*>(s.tx_ready.device_ptr()),
        // send_buffer: first half of RDMA buffer
        static_cast<uint8_t*>(rdma_buffer_ptr),
        s.grid_counter,
        s.sync_counter,
        s.sync_ptrs,  // null for NODE_SIZE=1
        s.recv_ptrs,  // null for NODE_SIZE=1
        static_cast<uint64_t>(stream_ptr)
    );
}

int Buffer::coop_dispatch_recv(
    torch::Tensor& out_x,
    std::optional<torch::Tensor>& out_x_scales,
    torch::Tensor& out_num_tokens,
    int num_experts,
    int hidden_dim,
    int src_elemsize,
    int src_scale_elemsize,
    int hidden_dim_scale,
    int64_t stream_ptr
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;

    // Reset grid counter
    CUDA_CHECK(cudaMemsetAsync(s.grid_counter_2, 0, sizeof(uint32_t),
                               (cudaStream_t)stream_ptr));

    // Source format (what's in the send/recv buffer) — determines token_stride
    size_t x_elemsize = static_cast<size_t>(src_elemsize);
    size_t x_scale_elemsize = static_cast<size_t>(src_scale_elemsize);
    
    // Output format
    size_t out_elemsize = out_x.element_size();
    
    int num_sms = num_device_sms;

    // Send and recv buffers from RDMA layout
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;
    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);

    return coop::a2a_dispatch_recv(
        num_sms,  // Use all SMs
        static_cast<size_t>(hidden_dim),
        static_cast<size_t>(hidden_dim_scale),
        x_elemsize,
        x_scale_elemsize,
        static_cast<size_t>(num_experts),
        static_cast<size_t>(rank),
        1,  // node_size = 1
        static_cast<size_t>(num_ranks),
        reinterpret_cast<int32_t*>(out_num_tokens.data_ptr()),
        reinterpret_cast<uint8_t*>(out_x.data_ptr()),
        out_x.stride(0) * out_elemsize,
        out_x_scales.has_value() ? reinterpret_cast<uint8_t*>(out_x_scales->data_ptr()) : nullptr,
        out_x_scales.has_value() ? out_x_scales->stride(1) * x_scale_elemsize : 0,
        out_x_scales.has_value() ? out_x_scales->stride(0) * x_scale_elemsize : 0,
        s.tokens_per_expert,
        rdma_base,            // send_buffer (for self-tokens)
        rdma_base + half_rdma + efa::kRouteRegionSize, // recv_buffer (EFA-received tokens), offset past route region
        s.source_rank,
        s.source_offset,
        s.padded_index,
        s.num_routed,
        s.num_recv_tokens,
        const_cast<uint8_t*>(s.num_recv_tokens_flag.device_ptr()),
        // dispatch_recv_flag: use coop_scratch's own flag (CPU sets when RDMA done)
        const_cast<uint8_t*>(s.dispatch_recv_flag.device_ptr()),
        const_cast<uint8_t*>(s.dispatch_recv_done.device_ptr()),
        s.grid_counter_2,
        s.sync_counter,
        s.sync_ptrs,
        s.send_ptrs,
        static_cast<uint64_t>(stream_ptr)
    );
}

void Buffer::coop_set_recv_metadata(
    const std::vector<uint32_t>& source_rank,
    const std::vector<uint32_t>& source_offset,
    const std::vector<uint32_t>& padded_index,
    const std::vector<uint32_t>& combine_send_offset,
    const std::vector<uint32_t>& tokens_per_expert,
    int total_recv_tokens,
    int efa_recv_tokens
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;

    // Iter 53: Use GdrVec MMIO writes instead of cudaMemcpy
    size_t n = source_rank.size();
    if (n > 0) {
        s.gdr_source_rank.copy(source_rank.data(), n);
        s.gdr_source_offset.copy(source_offset.data(), n);
        s.gdr_padded_index.copy(padded_index.data(), n);
        s.gdr_combine_send_offset.copy(combine_send_offset.data(), n);
    }
    if (!tokens_per_expert.empty()) {
        s.gdr_tokens_per_expert.copy(tokens_per_expert.data(), tokens_per_expert.size());
    }
    // Write num_recv_tokens: [total, efa_only]
    s.gdr_num_recv_tokens.write(0, static_cast<uint32_t>(total_recv_tokens));
    s.gdr_num_recv_tokens.write(1, static_cast<uint32_t>(efa_recv_tokens));
}

int Buffer::coop_combine_send(
    const torch::Tensor& expert_x,
    int hidden_dim,
    int dp_size,
    int64_t stream_ptr
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;

    // Reset token counter
    CUDA_CHECK(cudaMemsetAsync(s.combine_token_counter, 0, sizeof(uint32_t),
                               (cudaStream_t)stream_ptr));

    size_t x_elemsize = expert_x.element_size();
    int num_sms = num_device_sms;

    // RDMA layout
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;
    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);

    return coop::a2a_combine_send(
        num_sms,
        static_cast<size_t>(hidden_dim),
        x_elemsize,
        static_cast<size_t>(rank),
        1,  // node_size = 1
        static_cast<size_t>(dp_size),
        reinterpret_cast<const uint8_t*>(expert_x.data_ptr()),
        expert_x.stride(0) * x_elemsize,
        const_cast<uint8_t*>(s.tx_ready.device_ptr()),
        rdma_base,              // send_buffer
        rdma_base + half_rdma + efa::kRouteRegionSize,  // recv_buffer, offset past route region
        s.source_rank,
        s.combine_send_offset,
        s.padded_index,
        s.num_recv_tokens,
        // combine_send_done: use coop_scratch's own flag
        const_cast<uint8_t*>(s.combine_send_done.device_ptr()),
        s.combine_token_counter,
        s.sync_counter,
        s.sync_ptrs,
        s.recv_ptrs,
        static_cast<uint64_t>(stream_ptr)
    );
}

int Buffer::coop_combine_recv(
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    torch::Tensor& combined_x,
    int num_experts,
    int num_tokens,
    int hidden_dim,
    bool accumulate,
    int64_t stream_ptr
) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;

    size_t x_elemsize = combined_x.element_size();
    int num_sms = num_device_sms;
    size_t num_blocks = std::min(num_sms, std::max(1, num_tokens));

    // recv_buffer: combine recv is in the RDMA recv half
    uint64_t half_rdma = static_cast<uint64_t>(num_rdma_bytes) / 2;
    uint8_t* rdma_base = static_cast<uint8_t*>(rdma_buffer_ptr);

    return coop::a2a_combine_recv(
        num_blocks,
        static_cast<size_t>(hidden_dim),
        x_elemsize,
        static_cast<size_t>(num_experts),
        static_cast<size_t>(topk_idx.size(1)),
        static_cast<size_t>(rank),
        1,  // node_size = 1
        static_cast<size_t>(num_ranks),
        static_cast<size_t>(num_tokens),
        nullptr,  // bound_m_ptr
        reinterpret_cast<const int32_t*>(topk_idx.data_ptr()),
        topk_idx.stride(0),
        reinterpret_cast<const float*>(topk_weights.data_ptr()),
        topk_weights.stride(0),
        reinterpret_cast<uint8_t*>(combined_x.data_ptr()),
        combined_x.stride(0),  // element stride, NOT byte stride (kernel does bf16* + stride)
        accumulate,
        rdma_base + half_rdma + efa::kRouteRegionSize,  // recv_buffer for combine, offset past route region
        s.token_offset,
        s.expert_offsets,
        // combine_recv_flag: use coop_scratch's own flag (CPU sets when RDMA done)
        const_cast<uint8_t*>(s.combine_recv_flag.device_ptr()),
        const_cast<uint8_t*>(s.combine_recv_done.device_ptr()),
        s.sync_counter,
        s.sync_ptrs,
        static_cast<uint64_t>(stream_ptr)
    );
}

std::vector<int32_t> Buffer::coop_read_send_counts(int num_experts, int experts_per_rank) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;

    // Read per-expert counts from GPU
    std::vector<uint32_t> per_expert(num_experts);
    CUDA_CHECK(cudaMemcpy(per_expert.data(), s.num_routed,
                          num_experts * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Aggregate to per-rank counts
    std::vector<int32_t> per_rank(num_ranks, 0);
    for (int e = 0; e < num_experts; ++e) {
        int r = e / experts_per_rank;
        if (r < num_ranks) {
            per_rank[r] += static_cast<int32_t>(per_expert[e]);
        }
    }
    return per_rank;
}

torch::Tensor Buffer::coop_get_num_routed_tensor(int num_experts) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    // Return a GPU tensor view of num_routed (zero-copy, uint32 → int32 reinterpret)
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    return torch::from_blob(coop_scratch_.num_routed, {num_experts}, options);
}

void Buffer::coop_signal_tx_ready() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    coop_scratch_.tx_ready.set(1);
}

void Buffer::coop_wait_route_done() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    // Spin-wait for dispatch_route_done flag (GPU→CPU)
    while (coop_scratch_.dispatch_route_done.read() == 0) {
        _mm_pause();
    }
    coop_scratch_.dispatch_route_done.clear();
}

void Buffer::coop_signal_recv_metadata_ready() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    coop_scratch_.num_recv_tokens_flag.set(1);
}

void Buffer::coop_signal_dispatch_recv_ready() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    coop_scratch_.dispatch_recv_flag.set(1);
}

void Buffer::coop_signal_combine_recv_ready() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    coop_scratch_.combine_recv_flag.set(1);
}

void Buffer::coop_wait_dispatch_recv_done() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    while (coop_scratch_.dispatch_recv_done.read() == 0) {
        _mm_pause();
    }
    coop_scratch_.dispatch_recv_done.clear();
}

void Buffer::coop_wait_combine_send_done() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    while (coop_scratch_.combine_send_done.read() == 0) {
        _mm_pause();
    }
    coop_scratch_.combine_send_done.clear();
}

void Buffer::coop_wait_combine_recv_done() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    while (coop_scratch_.combine_recv_done.read() == 0) {
        _mm_pause();
    }
    coop_scratch_.combine_recv_done.clear();
}

void Buffer::coop_reset_flags() {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    auto& s = coop_scratch_;
    s.dispatch_route_done.clear();
    s.dispatch_pack_done.clear();
    // tx_ready is set by coop_signal_tx_ready or pre-set
    s.num_recv_tokens_flag.clear();
    s.dispatch_recv_flag.clear();
    s.dispatch_recv_done.clear();
    s.combine_send_done.clear();
    s.combine_recv_flag.clear();
    s.combine_recv_done.clear();
}

// ============================================================================
// Iter 61: Reset tag counters between benchmark phases to prevent tag wrapping
// ============================================================================
void Buffer::coop_reset_tags() {
    if (ll_pipeline_) {
        ll_pipeline_->reset_tags();
    }
}

// ============================================================================
// Iter 50: Cooperative Worker Thread API
// ============================================================================

void Buffer::coop_worker_init(int num_experts, int num_topk,
                               int max_tokens_per_rank,
                               int coop_token_stride,
                               int combine_token_dim) {
    EP_HOST_ASSERT(coop_scratch_.initialized);
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    auto& s = coop_scratch_;

    efa::CoopConfig cc;
    cc.num_experts = num_experts;
    cc.experts_per_rank = (num_experts + num_ranks - 1) / num_ranks;
    cc.num_topk = num_topk;
    cc.max_tokens_per_rank = max_tokens_per_rank;
    cc.coop_token_stride = coop_token_stride;
    cc.combine_token_dim = combine_token_dim;

    // GPU pointers (aliases to GdrVec device_ptr())
    cc.gpu_num_routed = s.num_routed;
    cc.gpu_source_rank = s.source_rank;
    cc.gpu_source_offset = s.source_offset;
    cc.gpu_padded_index = s.padded_index;
    cc.gpu_combine_send_offset = s.combine_send_offset;
    cc.gpu_tokens_per_expert = s.tokens_per_expert;
    cc.gpu_num_recv_tokens = s.num_recv_tokens;

    // GDR-mapped metadata vectors (Iter 53)
    cc.gdr_num_routed = &s.gdr_num_routed;
    cc.gdr_source_rank = &s.gdr_source_rank;
    cc.gdr_source_offset = &s.gdr_source_offset;
    cc.gdr_padded_index = &s.gdr_padded_index;
    cc.gdr_combine_send_offset = &s.gdr_combine_send_offset;
    cc.gdr_tokens_per_expert = &s.gdr_tokens_per_expert;
    cc.gdr_num_recv_tokens = &s.gdr_num_recv_tokens;

    // GDR flag pointers
    cc.dispatch_route_done = &s.dispatch_route_done;
    cc.dispatch_pack_done = &s.dispatch_pack_done;
    cc.tx_ready = &s.tx_ready;
    cc.num_recv_tokens_flag = &s.num_recv_tokens_flag;
    cc.dispatch_recv_flag = &s.dispatch_recv_flag;
    cc.dispatch_recv_done = &s.dispatch_recv_done;
    cc.combine_send_done = &s.combine_send_done;
    cc.combine_recv_flag = &s.combine_recv_flag;
    cc.combine_recv_done = &s.combine_recv_done;

    // GDR counters for auto-signaling (Iter 53)
    cc.dispatch_rdma_counter = &s.dispatch_rdma_counter;
    cc.combine_rdma_counter = &s.combine_rdma_counter;

    cc.initialized = true;

    ll_pipeline_->set_coop_config(cc);
    ll_pipeline_->set_coop_mode(true);

    fprintf(stderr, "[Rank %d] Coop worker initialized: experts=%d, topk=%d, "
            "max_tok/rank=%d, dispatch_stride=%d, combine_dim=%d\n",
            rank, num_experts, num_topk, max_tokens_per_rank,
            coop_token_stride, combine_token_dim);
}

void Buffer::start_coop_dispatch() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    ll_pipeline_->start_coop_dispatch();
}

void Buffer::wait_coop_dispatch_done() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    pybind11::gil_scoped_release release;
    ll_pipeline_->wait_coop_dispatch_done();
}

void Buffer::start_coop_combine() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    ll_pipeline_->start_coop_combine();
}

void Buffer::wait_coop_combine_done() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    pybind11::gil_scoped_release release;
    ll_pipeline_->wait_coop_combine_done();
}

std::tuple<std::vector<int32_t>, std::vector<int32_t>, int, int>
Buffer::coop_worker_get_counts() {
    EP_HOST_ASSERT(ll_pipeline_initialized_ && ll_pipeline_);
    // Read stashed counts from the worker
    std::vector<int32_t> send_counts(num_ranks);
    std::vector<int32_t> recv_counts(num_ranks);
    // Access the worker's stashed counts (they're public-ish via the coop_config_)
    // The worker stores them in coop_send_counts_ / coop_recv_counts_
    // We need a getter. For now, expose via a simple accessor.
    // Actually, the pipeline worker is a unique_ptr. Let's just get the data.
    for (int i = 0; i < num_ranks; ++i) {
        send_counts[i] = ll_pipeline_->coop_send_counts()[i];
        recv_counts[i] = ll_pipeline_->coop_recv_counts()[i];
    }
    int total_send = 0, total_recv = 0;
    for (int i = 0; i < num_ranks; ++i) {
        total_send += send_counts[i];
        total_recv += recv_counts[i];
    }
    return {send_counts, recv_counts, total_send, total_recv};
}

}  // namespace deep_ep
#endif  // ENABLE_EFA

// Forward declarations for EFA kernel functions (defined in efa_kernels.cu)
#ifdef ENABLE_EFA
namespace deep_ep {
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>
moe_routing_sort(const torch::Tensor& is_token_in_rank);

std::tuple<torch::Tensor, torch::Tensor>
topk_remap(const torch::Tensor& topk_idx,
           const torch::Tensor& topk_weights,
           const torch::Tensor& sorted_token_ids,
           const torch::Tensor& send_cumsum,
           int total_send,
           int num_ranks,
           int num_local_experts);

void efa_permute(const torch::Tensor& src,
                 torch::Tensor& dst,
                 const torch::Tensor& src_offsets,
                 const torch::Tensor& dst_offsets,
                 const torch::Tensor& copy_sizes,
                 int64_t total_bytes);

torch::Tensor build_recv_src_meta(
    const torch::Tensor& recv_counts,
    const torch::Tensor& recv_cumsum,
    int num_recv_tokens,
    int num_ranks,
    int num_local_ranks);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
ll_dispatch_route_and_pack(
    const torch::Tensor& topk_idx,
    const torch::Tensor& x_data,
    const c10::optional<torch::Tensor>& x_scales_opt,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ll_recv_unpack(
    const torch::Tensor& recv_packed,
    const torch::Tensor& recv_counts_tensor,
    torch::Tensor& packed_recv_x,
    c10::optional<torch::Tensor> packed_recv_scales_opt,
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t N,
    int hidden_bytes,
    int scale_cols_bytes);

void ll_combine_weighted_reduce(
    const torch::Tensor& combine_recv_flat,
    const torch::Tensor& sorted_send_token_ids,
    const torch::Tensor& global_expert_ids,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    torch::Tensor& combined_x,
    int total_send,
    int num_tokens,
    int hidden,
    int num_topk);

void ll_combine_weighted_reduce_v2(
    const torch::Tensor& combine_recv_flat,
    const torch::Tensor& sorted_send_token_ids,
    const torch::Tensor& global_expert_ids,
    const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights,
    torch::Tensor& combined_x,
    torch::Tensor& combined_x_f32,
    const c10::optional<torch::Tensor>& send_cumsum_opt,
    int total_send,
    int num_tokens,
    int hidden,
    int num_topk,
    int num_ranks,
    int64_t slot_size);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int>
ll_dispatch_route_and_pack_v2(
    const torch::Tensor& topk_idx,
    const torch::Tensor& x_data,
    const c10::optional<torch::Tensor>& x_scales_opt,
    torch::Tensor& send_counts,
    torch::Tensor& send_cumsum,
    torch::Tensor& total_send_out,
    torch::Tensor& send_packed_max,
    torch::Tensor& sorted_token_ids_max,
    torch::Tensor& sorted_local_eids_max,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ll_recv_unpack_v2(
    const torch::Tensor& recv_packed,
    const torch::Tensor& recv_counts_tensor,
    torch::Tensor& packed_recv_x,
    c10::optional<torch::Tensor> packed_recv_scales_opt,
    torch::Tensor& recv_cumsum,
    torch::Tensor& pair_counts,
    torch::Tensor& packed_recv_count,
    torch::Tensor& packed_recv_layout_range,
    torch::Tensor& expert_cumsum,
    torch::Tensor& pair_cumsum,
    torch::Tensor& packed_recv_src_info,
    torch::Tensor& recv_expert_ids_max,
    torch::Tensor& recv_expert_pos_max,
    torch::Tensor& sort_order_recv_max,
    torch::Tensor& pair_write_counters,
    int total_recv,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token,
    int64_t N,
    int hidden_bytes,
    int scale_cols_bytes,
    int64_t slot_size);

std::tuple<torch::Tensor, torch::Tensor>
ll_compute_combine_helpers(
    const torch::Tensor& sort_order_recv,
    const torch::Tensor& send_cumsum,
    const torch::Tensor& sorted_local_eids,
    torch::Tensor& inverse_sort_max,
    torch::Tensor& global_expert_ids_max,
    int total_recv,
    int total_send,
    int num_ranks,
    int num_local_experts);

std::tuple<torch::Tensor, torch::Tensor>
per_token_cast_to_fp8(
    const torch::Tensor& x,
    bool round_scale,
    bool use_ue8m0);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ll_dispatch_route_and_pack_v3(
    const torch::Tensor& topk_idx,
    const torch::Tensor& x_data,
    const c10::optional<torch::Tensor>& x_scales_opt,
    torch::Tensor& send_counts,
    torch::Tensor& send_cumsum,
    torch::Tensor& total_send_out,
    torch::Tensor& send_packed_max,
    torch::Tensor& sorted_token_ids_max,
    torch::Tensor& sorted_local_eids_max,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token);

void ll_dispatch_count_only(
    const torch::Tensor& topk_idx,
    torch::Tensor& send_counts,
    torch::Tensor& send_cumsum,
    torch::Tensor& total_send_out,
    int num_ranks,
    int num_local_experts);

void ll_dispatch_pack_only(
    const torch::Tensor& topk_idx,
    const torch::Tensor& x_data,
    const c10::optional<torch::Tensor>& x_scales_opt,
    const torch::Tensor& send_cumsum,
    torch::Tensor& send_packed_max,
    torch::Tensor& sorted_token_ids_max,
    torch::Tensor& sorted_local_eids_max,
    int num_ranks,
    int num_local_experts,
    int data_bytes_per_token,
    int scale_bytes_per_token,
    int packed_bytes_per_token);

void gdr_signal_counts(
    const torch::Tensor& send_counts_tensor,
    int64_t gdr_send_counts_ptr,
    int64_t gdr_pack_done_ptr,
    int num_ranks);

}  // namespace deep_ep
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_nvl_chunked_send_tokens") = 6,
             py::arg("num_max_nvl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6,
             py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def_readonly("num_sms", &deep_ep::Config::num_sms)
        .def_readonly("num_max_nvl_chunked_send_tokens", &deep_ep::Config::num_max_nvl_chunked_send_tokens)
        .def_readonly("num_max_nvl_chunked_recv_tokens", &deep_ep::Config::num_max_nvl_chunked_recv_tokens)
        .def_readonly("num_max_rdma_chunked_send_tokens", &deep_ep::Config::num_max_rdma_chunked_send_tokens)
        .def_readonly("num_max_rdma_chunked_recv_tokens", &deep_ep::Config::num_max_rdma_chunked_recv_tokens)
        .def("get_nvl_buffer_size_hint", &deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool, bool, bool, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_nvshmem_unique_id", &deep_ep::Buffer::get_local_nvshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("get_comm_stream", &deep_ep::Buffer::get_comm_stream)
        .def("sync", &deep_ep::Buffer::sync)
        .def("destroy", &deep_ep::Buffer::destroy)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine)
        .def("low_latency_update_mask_buffer", &deep_ep::Buffer::low_latency_update_mask_buffer)
        .def("low_latency_query_mask_buffer", &deep_ep::Buffer::low_latency_query_mask_buffer)
        .def("low_latency_clean_mask_buffer", &deep_ep::Buffer::low_latency_clean_mask_buffer)
        .def("get_next_low_latency_combine_buffer", &deep_ep::Buffer::get_next_low_latency_combine_buffer)
#ifdef ENABLE_EFA
        .def("init_efa", &deep_ep::Buffer::init_efa)
        .def("efa_all_to_all", [](deep_ep::Buffer& self,
                                    const torch::Tensor& send_buf,
                                    const std::vector<int64_t>& send_sizes,
                                    const std::vector<int64_t>& send_offsets,
                                    const torch::Tensor& recv_buf,
                                    const std::vector<int64_t>& recv_sizes,
                                    const std::vector<int64_t>& recv_offsets,
                                    int override_tag,
                                    int override_baseline,
                                    std::optional<std::vector<int32_t>> send_token_counts,
                                    std::optional<std::vector<int32_t>> recv_token_counts,
                                    int packed_bytes_per_token) {
                 // Convert optional vectors to raw pointers for the C++ API
                 const std::vector<int32_t>* stc_ptr = send_token_counts ? &*send_token_counts : nullptr;
                 std::vector<int32_t>* rtc_ptr = recv_token_counts ? &*recv_token_counts : nullptr;
                 self.efa_all_to_all(send_buf, send_sizes, send_offsets,
                                     recv_buf, recv_sizes, recv_offsets,
                                     override_tag, override_baseline,
                                     stc_ptr, rtc_ptr, packed_bytes_per_token);
             },
             py::arg("send_buf"), py::arg("send_sizes"), py::arg("send_offsets"),
             py::arg("recv_buf"), py::arg("recv_sizes"), py::arg("recv_offsets"),
             py::arg("override_tag") = -1, py::arg("override_baseline") = -1,
             py::arg("send_token_counts") = py::none(),
             py::arg("recv_token_counts") = py::none(),
             py::arg("packed_bytes_per_token") = 0)
        .def("efa_count_exchange", &deep_ep::Buffer::efa_count_exchange)
        .def("init_ll_pipeline", &deep_ep::Buffer::init_ll_pipeline)
        .def("ll_pipeline_start_dispatch", &deep_ep::Buffer::ll_pipeline_start_dispatch)
        .def("ll_pipeline_wait_dispatch", &deep_ep::Buffer::ll_pipeline_wait_dispatch)
        .def("ll_pipeline_start_combine", &deep_ep::Buffer::ll_pipeline_start_combine)
        .def("ll_pipeline_wait_combine", &deep_ep::Buffer::ll_pipeline_wait_combine)
        .def("ll_pipeline_barrier", &deep_ep::Buffer::ll_pipeline_barrier)
        .def("ll_pipeline_get_dispatch_gdr_ptrs", &deep_ep::Buffer::ll_pipeline_get_dispatch_gdr_ptrs)
        .def("ll_pipeline_get_combine_gdr_ptrs", &deep_ep::Buffer::ll_pipeline_get_combine_gdr_ptrs)
        .def("ll_efa_dispatch_data", &deep_ep::Buffer::ll_efa_dispatch_data)
        .def("ll_efa_dispatch_data_v2", &deep_ep::Buffer::ll_efa_dispatch_data_v2)
        .def("ll_efa_dispatch_data_v3", &deep_ep::Buffer::ll_efa_dispatch_data_v3)
        .def("ll_efa_combine_data", &deep_ep::Buffer::ll_efa_combine_data,
             py::arg("send_counts"), py::arg("recv_counts"),
             py::arg("send_buf"), py::arg("recv_buf"),
             py::arg("hidden"), py::arg("bytes_per_elem"),
             py::arg("combine_slot_size"),
             py::arg("override_tag") = -1, py::arg("override_baseline") = -1)
        .def("ll_efa_combine_data_v2", &deep_ep::Buffer::ll_efa_combine_data_v2)
        .def("efa_barrier", &deep_ep::Buffer::efa_barrier)
        .def("efa_imm_barrier", &deep_ep::Buffer::efa_imm_barrier)
        .def("ll_efa_rdma_dispatch", &deep_ep::Buffer::ll_efa_rdma_dispatch)
        // Cooperative kernel wrappers (Iter 48)
        .def("coop_init", &deep_ep::Buffer::coop_init)
        .def("coop_get_ptrs", &deep_ep::Buffer::coop_get_ptrs)
        .def("coop_dispatch_send", &deep_ep::Buffer::coop_dispatch_send)
        .def("coop_dispatch_recv", &deep_ep::Buffer::coop_dispatch_recv)
        .def("coop_set_recv_metadata", &deep_ep::Buffer::coop_set_recv_metadata)
        .def("coop_combine_send", &deep_ep::Buffer::coop_combine_send)
        .def("coop_combine_recv", &deep_ep::Buffer::coop_combine_recv)
        .def("coop_read_send_counts", &deep_ep::Buffer::coop_read_send_counts)
        .def("coop_get_num_routed_tensor", &deep_ep::Buffer::coop_get_num_routed_tensor)
        .def("coop_signal_tx_ready", &deep_ep::Buffer::coop_signal_tx_ready)
        .def("coop_wait_route_done", &deep_ep::Buffer::coop_wait_route_done)
        .def("coop_signal_recv_metadata_ready", &deep_ep::Buffer::coop_signal_recv_metadata_ready)
        .def("coop_signal_dispatch_recv_ready", &deep_ep::Buffer::coop_signal_dispatch_recv_ready)
        .def("coop_signal_combine_recv_ready", &deep_ep::Buffer::coop_signal_combine_recv_ready)
        .def("coop_wait_dispatch_recv_done", &deep_ep::Buffer::coop_wait_dispatch_recv_done)
        .def("coop_wait_combine_send_done", &deep_ep::Buffer::coop_wait_combine_send_done)
        .def("coop_wait_combine_recv_done", &deep_ep::Buffer::coop_wait_combine_recv_done)
        .def("coop_reset_flags", &deep_ep::Buffer::coop_reset_flags)
        .def("coop_reset_tags", &deep_ep::Buffer::coop_reset_tags)
        .def("coop_efa_dispatch_rdma", &deep_ep::Buffer::coop_efa_dispatch_rdma)
        .def("coop_efa_combine_rdma", &deep_ep::Buffer::coop_efa_combine_rdma)
        // Iter 50: Cooperative worker thread API
        .def("coop_worker_init", &deep_ep::Buffer::coop_worker_init)
        .def("start_coop_dispatch", &deep_ep::Buffer::start_coop_dispatch)
        .def("wait_coop_dispatch_done", &deep_ep::Buffer::wait_coop_dispatch_done)
        .def("start_coop_combine", &deep_ep::Buffer::start_coop_combine)
        .def("wait_coop_combine_done", &deep_ep::Buffer::wait_coop_combine_done)
        .def("coop_worker_get_counts", &deep_ep::Buffer::coop_worker_get_counts)
#endif
        ;

    m.def("is_sm90_compiled", deep_ep::is_sm90_compiled);
    m.attr("topk_idx_t") =
        py::reinterpret_borrow<py::object>((PyObject*)torch::getTHPDtype(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value));

#ifdef ENABLE_EFA
    // EFA fused kernel functions
    m.def("moe_routing_sort", &deep_ep::moe_routing_sort,
          "Fused nonzero + argsort + bincount for MoE token routing");
    m.def("topk_remap", &deep_ep::topk_remap,
          "Fused topk remapping for internode dispatch");
    m.def("efa_permute", &deep_ep::efa_permute,
          "Fused gather/scatter for EFA pack/unpack");
    m.def("build_recv_src_meta", &deep_ep::build_recv_src_meta,
          "Fused construction of recv_src_meta from recv_counts");
    m.def("ll_dispatch_route_and_pack", &deep_ep::ll_dispatch_route_and_pack,
          "Fused routing + packing for LL dispatch");
    m.def("ll_recv_unpack", &deep_ep::ll_recv_unpack,
          "Fused unpack + scatter for LL dispatch receive side");
    m.def("ll_combine_weighted_reduce", &deep_ep::ll_combine_weighted_reduce,
          "Fused weighted reduction for LL combine");
    m.def("ll_dispatch_route_and_pack_v2", &deep_ep::ll_dispatch_route_and_pack_v2,
          "Fused routing + packing for LL dispatch with pre-allocated buffers");
    m.def("ll_recv_unpack_v2", &deep_ep::ll_recv_unpack_v2,
          "Fused unpack + scatter for LL dispatch receive side with pre-allocated buffers");
    m.def("ll_combine_weighted_reduce_v2", &deep_ep::ll_combine_weighted_reduce_v2,
          "Fused weighted reduction for LL combine with pre-allocated f32 accumulator");
    m.def("ll_compute_combine_helpers", &deep_ep::ll_compute_combine_helpers,
          "Compute inverse_sort + global_expert_ids for LL combine");
    m.def("per_token_cast_to_fp8", &deep_ep::per_token_cast_to_fp8,
          "Fused per-token FP8 E4M3 quantization with group_size=128");
    m.def("ll_dispatch_route_and_pack_v3", &deep_ep::ll_dispatch_route_and_pack_v3,
          "Fused routing + packing for LL dispatch — no .item() GPU sync");
    m.def("ll_dispatch_count_only", &deep_ep::ll_dispatch_count_only,
          "LL dispatch Phase 1+2: count + prefix sum only");
    m.def("ll_dispatch_pack_only", &deep_ep::ll_dispatch_pack_only,
          "LL dispatch Phase 3: scatter + pack only (uses pre-computed cumsum)");
    m.def("gdr_signal_counts", &deep_ep::gdr_signal_counts,
          "Copy send_counts to GDR memory and signal pack_done flag via MMIO");
#endif
}
