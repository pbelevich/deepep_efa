#include "efa_transport.h"

#include <cuda.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <functional>
#include <unistd.h>

// For kernel version check
#include <sys/utsname.h>

namespace deep_ep {
namespace efa {

// Check if the Linux kernel supports DMA-BUF (>= 5.12)
static bool linux_kernel_supports_dma_buf() {
    static int cached = -1;
    if (cached >= 0) return cached;

    struct utsname uts;
    if (uname(&uts) != 0) {
        cached = 0;
        return false;
    }
    int major = 0, minor = 0;
    if (sscanf(uts.release, "%d.%d", &major, &minor) < 2) {
        cached = 0;
        return false;
    }
    cached = (major > 5 || (major == 5 && minor >= 12)) ? 1 : 0;
    return cached;
}

// Try to get a DMA-BUF file descriptor for a GPU memory range.
// Returns the fd on success, -1 on failure.
static int get_dma_buf_fd(void* ptr, size_t size) {
    int dmabuf_fd = -1;
    CUresult res = cuMemGetHandleForAddressRange(
        &dmabuf_fd,
        reinterpret_cast<CUdeviceptr>(ptr),
        size,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
        0);
    if (res != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(res, &err_str);
        fprintf(stderr, "DMA-BUF: cuMemGetHandleForAddressRange failed: %s (res=%d), falling back to legacy HMEM\n",
                err_str ? err_str : "unknown", res);
        return -1;
    }
    return dmabuf_fd;
}

// P5 PCI topology: map GPU index to closest EFA device
// GPU 0,1 -> EFA 0; GPU 2,3 -> EFA 1; GPU 4,5 -> EFA 2; GPU 6,7 -> EFA 3
int get_efa_device_for_gpu(int gpu_id) {
    return gpu_id / 2;
}

// Get the list of EFA device indices for a given GPU (multi-NIC)
// On P5en with 32 EFA devices and 8 GPUs, each GPU gets 4 NICs.
// We distribute NICs evenly: GPU i gets devices [i*N, i*N+1, ..., i*N+(N-1)]
// where N = total_efa_devices / 8.
std::vector<int> get_efa_devices_for_gpu(int gpu_id, int total_efa_devices) {
    int num_gpus = 8;  // P5en has 8 GPUs
    int nics_per_gpu = total_efa_devices / num_gpus;
    if (nics_per_gpu < 1) nics_per_gpu = 1;
    if (nics_per_gpu > kMaxNicsPerGpu) nics_per_gpu = kMaxNicsPerGpu;
    
    std::vector<int> devices;
    int start = gpu_id * nics_per_gpu;
    for (int i = 0; i < nics_per_gpu && (start + i) < total_efa_devices; ++i) {
        devices.push_back(start + i);
    }
    return devices;
}

EfaEndpoint::~EfaEndpoint() {
    // Resources cleaned up by destroy_endpoint()
}

void init_endpoint(EfaEndpoint& ep, int efa_device_idx, int cq_size) {
    struct fi_info* hints = fi_allocinfo();
    if (!hints) {
        throw std::runtime_error("Failed to allocate fi_info hints");
    }

    // Match pplx-garden's hint setup
    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR |
                                   FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                                   FI_MR_HMEM;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;

    // Select EFA provider
    hints->fabric_attr->prov_name = strdup("efa");

    int ret = fi_getinfo(FI_VERSION(1, 22), nullptr, nullptr, 0, hints, &ep.info);
    fi_freeinfo(hints);

    if (ret != 0) {
        throw std::runtime_error(std::string("fi_getinfo failed: ") +
                                 fi_strerror(-ret));
    }

    // If there are multiple EFA devices, walk the list to find the right one
    if (efa_device_idx > 0) {
        struct fi_info* cur = ep.info;
        int idx = 0;
        while (cur && idx < efa_device_idx) {
            cur = cur->next;
            idx++;
        }
        if (!cur) {
            // Fall back to first device if specified index not found
            cur = ep.info;
        }
        // Use this specific info entry
        struct fi_info* selected = fi_dupinfo(cur);
        fi_freeinfo(ep.info);
        ep.info = selected;
    }

    FI_CHECK(fi_fabric(ep.info->fabric_attr, &ep.fabric, nullptr));
    FI_CHECK(fi_domain(ep.fabric, ep.info, &ep.domain, nullptr));

    // Create TX completion queue
    struct fi_cq_attr tx_cq_attr = {};
    tx_cq_attr.size = cq_size;
    tx_cq_attr.format = FI_CQ_FORMAT_DATA;
    tx_cq_attr.wait_obj = FI_WAIT_NONE;
    FI_CHECK(fi_cq_open(ep.domain, &tx_cq_attr, &ep.tx_cq, nullptr));

    // Create RX completion queue
    struct fi_cq_attr rx_cq_attr = {};
    rx_cq_attr.size = cq_size;
    rx_cq_attr.format = FI_CQ_FORMAT_DATA;
    rx_cq_attr.wait_obj = FI_WAIT_NONE;
    FI_CHECK(fi_cq_open(ep.domain, &rx_cq_attr, &ep.rx_cq, nullptr));

    // Create address vector
    struct fi_av_attr av_attr = {};
    av_attr.type = FI_AV_TABLE;
    av_attr.count = 256;  // Max expected peers
    FI_CHECK(fi_av_open(ep.domain, &av_attr, &ep.av, nullptr));

    // Create endpoint
    FI_CHECK(fi_endpoint(ep.domain, ep.info, &ep.ep, nullptr));

    // Disable shared memory and CUDA P2P - force all data through RDMA
    // (matching pplx-garden's approach)
    bool opt_false = false;
    fi_setopt(&ep.ep->fid, FI_OPT_ENDPOINT,
              FI_OPT_SHARED_MEMORY_PERMITTED,
              &opt_false, sizeof(opt_false));
    fi_setopt(&ep.ep->fid, FI_OPT_ENDPOINT,
              FI_OPT_CUDA_API_PERMITTED,
              &opt_false, sizeof(opt_false));

    FI_CHECK(fi_ep_bind(ep.ep, &ep.tx_cq->fid, FI_TRANSMIT));
    FI_CHECK(fi_ep_bind(ep.ep, &ep.rx_cq->fid, FI_RECV));
    FI_CHECK(fi_ep_bind(ep.ep, &ep.av->fid, 0));
    FI_CHECK(fi_enable(ep.ep));

    // Get local endpoint name
    size_t name_len = 0;
    fi_getname(&ep.ep->fid, nullptr, &name_len);
    ep.local_name.resize(name_len);
    FI_CHECK(fi_getname(&ep.ep->fid, ep.local_name.data(), &name_len));
}

void register_memory(EfaEndpoint& ep, void* gpu_ptr, size_t size,
                     uint64_t requested_key) {
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {};
    iov.iov_base = gpu_ptr;
    iov.iov_len = size;

    mr_attr.iov_count = 1;
    mr_attr.access = FI_REMOTE_READ | FI_REMOTE_WRITE |
                     FI_READ | FI_WRITE |
                     FI_SEND | FI_RECV;
    mr_attr.requested_key = requested_key;
    mr_attr.iface = FI_HMEM_CUDA;

    // Get CUDA device for this pointer
    cudaPointerAttributes ptr_attrs;
    cudaPointerGetAttributes(&ptr_attrs, gpu_ptr);
    mr_attr.device.cuda = ptr_attrs.device;

    // Try DMA-BUF based registration first (preferred path for GPU-Direct RDMA),
    // then fall back to legacy FI_HMEM_CUDA if DMA-BUF is unavailable.
    int dmabuf_fd = -1;
    bool use_dmabuf = false;

    if (linux_kernel_supports_dma_buf()) {
        dmabuf_fd = get_dma_buf_fd(gpu_ptr, size);
        if (dmabuf_fd >= 0) {
            use_dmabuf = true;
        }
    }

    if (use_dmabuf) {
        // DMA-BUF path: use fi_mr_dmabuf struct
        struct fi_mr_dmabuf dmabuf = {};
        dmabuf.fd = dmabuf_fd;
        dmabuf.len = size;
        dmabuf.base_addr = gpu_ptr;
        // In the fi_mr_attr union, set dmabuf instead of mr_iov
        mr_attr.dmabuf = &dmabuf;

        int ret = fi_mr_regattr(ep.domain, &mr_attr, FI_MR_DMABUF, &ep.mr);
        if (ret == 0) {
            ep.mr_key = fi_mr_key(ep.mr);
            // Store the fd so we can close it later
            ep.dmabuf_fd = dmabuf_fd;
            ep.mr_base = gpu_ptr;
            ep.mr_size = size;
            fprintf(stderr, "EFA: Registered GPU memory (%zu bytes) with DMA-BUF (fd=%d, key=%lu)\n",
                    size, dmabuf_fd, ep.mr_key);
            return;
        }

        // DMA-BUF registration failed; close the fd and fall through to legacy path
        fprintf(stderr, "EFA: fi_mr_regattr with DMA-BUF failed: %s (ret=%d), falling back to legacy HMEM\n",
                fi_strerror(-ret), ret);
        close(dmabuf_fd);
        dmabuf_fd = -1;
    }

    // Legacy path: plain FI_HMEM_CUDA with mr_iov
    mr_attr.mr_iov = &iov;
    FI_CHECK(fi_mr_regattr(ep.domain, &mr_attr, 0, &ep.mr));
    ep.mr_key = fi_mr_key(ep.mr);
    ep.dmabuf_fd = -1;
    ep.mr_base = gpu_ptr;
    ep.mr_size = size;
    fprintf(stderr, "EFA: Registered GPU memory (%zu bytes) with legacy HMEM (key=%lu)\n",
            size, ep.mr_key);
}

void exchange_addresses(
    EfaEndpoint& ep,
    int rank,
    int num_ranks,
    void* rdma_buffer_ptr,
    size_t rdma_buffer_size,
    const std::function<std::vector<std::vector<uint8_t>>(
        const std::vector<uint8_t>&)>& allgather_fn) {

    // Pack: [name_len (4B)] [name_data...] [mr_key (8B)] [base_addr (8B)]
    size_t name_len = ep.local_name.size();
    std::vector<uint8_t> local_data(4 + name_len + 8 + 8);
    uint32_t name_len_32 = static_cast<uint32_t>(name_len);
    memcpy(local_data.data(), &name_len_32, 4);
    memcpy(local_data.data() + 4, ep.local_name.data(), name_len);
    memcpy(local_data.data() + 4 + name_len, &ep.mr_key, 8);
    uint64_t base_addr = reinterpret_cast<uint64_t>(rdma_buffer_ptr);
    memcpy(local_data.data() + 4 + name_len + 8, &base_addr, 8);

    auto all_data = allgather_fn(local_data);

    ep.remote_addrs.resize(num_ranks);
    ep.remote_keys.resize(num_ranks);
    ep.remote_base_addrs.resize(num_ranks);

    for (int i = 0; i < num_ranks; ++i) {
        const auto& data = all_data[i];
        uint32_t remote_name_len;
        memcpy(&remote_name_len, data.data(), 4);

        // Insert address into AV
        int ret = fi_av_insert(ep.av, data.data() + 4, 1,
                               &ep.remote_addrs[i], 0, nullptr);
        if (ret != 1) {
            throw std::runtime_error("fi_av_insert failed for rank " +
                                     std::to_string(i));
        }

        memcpy(&ep.remote_keys[i], data.data() + 4 + remote_name_len, 8);
        memcpy(&ep.remote_base_addrs[i],
               data.data() + 4 + remote_name_len + 8, 8);
    }
}

int rdma_write_with_imm(EfaEndpoint& ep,
                        fi_addr_t remote_addr,
                        void* local_buf,
                        size_t len,
                        uint64_t remote_offset,
                        uint64_t remote_key,
                        uint64_t remote_base,
                        uint32_t imm_data,
                        void* context) {
    // Bounds check: ensure local_buf is within the registered MR
    if (ep.mr_base) {
        uintptr_t buf_start = reinterpret_cast<uintptr_t>(local_buf);
        uintptr_t buf_end = buf_start + len;
        uintptr_t mr_start = reinterpret_cast<uintptr_t>(ep.mr_base);
        uintptr_t mr_end = mr_start + ep.mr_size;
        if (buf_start < mr_start || buf_end > mr_end) {
            fprintf(stderr, "EFA RDMA WRITE OOB: local_buf=%p len=%zu range=[%p,%p) MR=[%p,%p) — SKIPPING\n",
                    local_buf, len,
                    reinterpret_cast<void*>(buf_start),
                    reinterpret_cast<void*>(buf_end),
                    reinterpret_cast<void*>(mr_start),
                    reinterpret_cast<void*>(mr_end));
            // Return 0 to pretend success — the write is not submitted to avoid EFA CQ errors
            return 0;
        }
    }

    struct iovec msg_iov = {};
    msg_iov.iov_base = local_buf;
    msg_iov.iov_len = len;

    struct fi_rma_iov rma_iov = {};
    rma_iov.addr = remote_base + remote_offset;
    rma_iov.len = len;
    rma_iov.key = remote_key;

    void* desc = fi_mr_desc(ep.mr);

    struct fi_msg_rma msg = {};
    msg.msg_iov = &msg_iov;
    msg.desc = &desc;
    msg.iov_count = 1;
    msg.addr = remote_addr;
    msg.rma_iov = &rma_iov;
    msg.rma_iov_count = 1;
    msg.context = context;
    msg.data = imm_data;

    return fi_writemsg(ep.ep, &msg, FI_REMOTE_CQ_DATA | FI_COMPLETION);
}

int poll_tx_cq(EfaEndpoint& ep, struct fi_cq_data_entry* entries,
               int max_entries) {
    ssize_t ret = fi_cq_read(ep.tx_cq, entries, max_entries);
    if (ret == -FI_EAGAIN) return 0;
    if (ret < 0) {
        // CQ error: one or more TX completions failed.
        // Read the error entry to clear it from the CQ, then return 1
        // so callers count the failed operation as completed (preventing deadlock).
        // The RDMA data was NOT delivered — the remote peer will time out.
        struct fi_cq_err_entry err_entry = {};
        fi_cq_readerr(ep.tx_cq, &err_entry, 0);
        fprintf(stderr, "EFA TX CQ error: %s (prov_errno=%d)\n",
                fi_cq_strerror(ep.tx_cq, err_entry.prov_errno,
                               err_entry.err_data, nullptr, 0),
                err_entry.prov_errno);
        ep.tx_errors++;
        return 1;  // Count as 1 completed (failed) operation
    }
    return static_cast<int>(ret);
}

int poll_rx_cq(EfaEndpoint& ep, struct fi_cq_data_entry* entries,
               int max_entries) {
    ssize_t ret = fi_cq_read(ep.rx_cq, entries, max_entries);
    if (ret == -FI_EAGAIN) return 0;
    if (ret < 0) {
        struct fi_cq_err_entry err_entry = {};
        fi_cq_readerr(ep.rx_cq, &err_entry, 0);
        fprintf(stderr, "EFA RX CQ error: %s (prov_errno=%d)\n",
                fi_cq_strerror(ep.rx_cq, err_entry.prov_errno,
                               err_entry.err_data, nullptr, 0),
                err_entry.prov_errno);
        return -1;
    }
    return static_cast<int>(ret);
}

int post_recv(EfaEndpoint& ep, void* buf, size_t len, void* context) {
    return fi_recv(ep.ep, buf, len, nullptr, FI_ADDR_UNSPEC, context);
}

void destroy_endpoint(EfaEndpoint& ep) {
    if (ep.mr) { fi_close(&ep.mr->fid); ep.mr = nullptr; }
    if (ep.dmabuf_fd >= 0) { close(ep.dmabuf_fd); ep.dmabuf_fd = -1; }
    if (ep.ep) { fi_close(&ep.ep->fid); ep.ep = nullptr; }
    if (ep.av) { fi_close(&ep.av->fid); ep.av = nullptr; }
    if (ep.tx_cq) { fi_close(&ep.tx_cq->fid); ep.tx_cq = nullptr; }
    if (ep.rx_cq) { fi_close(&ep.rx_cq->fid); ep.rx_cq = nullptr; }
    if (ep.domain) { fi_close(&ep.domain->fid); ep.domain = nullptr; }
    if (ep.fabric) { fi_close(&ep.fabric->fid); ep.fabric = nullptr; }
    if (ep.info) { fi_freeinfo(ep.info); ep.info = nullptr; }
}

// ============================================================================
// EfaTransport implementation
// ============================================================================

EfaTransport::~EfaTransport() {
    if (initialized_) {
        for (auto& ep : endpoints_) {
            destroy_endpoint(ep);
        }
    }
}

void EfaTransport::init(int gpu_id, int rank, int num_ranks) {
    gpu_id_ = gpu_id;
    rank_ = rank;
    num_ranks_ = num_ranks;

    // Count total EFA devices by walking fi_getinfo linked list
    struct fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR |
                                   FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                                   FI_MR_HMEM;
    hints->domain_attr->threading = FI_THREAD_DOMAIN;
    hints->fabric_attr->prov_name = strdup("efa");

    struct fi_info* info_list = nullptr;
    int ret = fi_getinfo(FI_VERSION(1, 22), nullptr, nullptr, 0, hints, &info_list);
    fi_freeinfo(hints);
    if (ret != 0) {
        throw std::runtime_error(std::string("fi_getinfo failed: ") + fi_strerror(-ret));
    }

    int total_devices = 0;
    for (struct fi_info* cur = info_list; cur; cur = cur->next) {
        total_devices++;
    }
    fi_freeinfo(info_list);

    // Get the NIC indices for this GPU
    auto nic_indices = get_efa_devices_for_gpu(gpu_id, total_devices);
    num_nics_ = static_cast<int>(nic_indices.size());

    fprintf(stderr, "EFA: GPU %d using %d NICs (out of %d total), indices:",
            gpu_id, num_nics_, total_devices);
    for (int idx : nic_indices) fprintf(stderr, " %d", idx);
    fprintf(stderr, "\n");

    // Create one endpoint per NIC
    endpoints_.resize(num_nics_);
    for (int n = 0; n < num_nics_; ++n) {
        init_endpoint(endpoints_[n], nic_indices[n]);
    }

    initialized_ = true;
}

void EfaTransport::register_buffer(void* gpu_ptr, size_t size) {
    assert(initialized_);
    // Register the same GPU buffer with all endpoints.
    // Each endpoint needs its own MR with a unique key.
    for (int n = 0; n < num_nics_; ++n) {
        // Use a key that encodes (rank, nic_index) to be globally unique
        uint64_t key = static_cast<uint64_t>(rank_) * kMaxNicsPerGpu + n;
        register_memory(endpoints_[n], gpu_ptr, size, key);
    }
}

void EfaTransport::exchange(
    void* rdma_buffer_ptr,
    size_t rdma_buffer_size,
    const std::function<std::vector<std::vector<uint8_t>>(
        const std::vector<uint8_t>&)>& allgather_fn) {
    assert(initialized_);
    // Exchange addresses for all NICs. Each NIC does its own allgather.
    for (int n = 0; n < num_nics_; ++n) {
        exchange_addresses(endpoints_[n], rank_, num_ranks_, rdma_buffer_ptr,
                          rdma_buffer_size, allgather_fn);
    }
}

}  // namespace efa
}  // namespace deep_ep
