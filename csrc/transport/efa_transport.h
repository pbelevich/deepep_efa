#pragma once

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace deep_ep {
namespace efa {

// Throw on libfabric errors
#define FI_CHECK(call)                                                          \
    do {                                                                         \
        int ret_ = (call);                                                       \
        if (ret_ < 0) {                                                          \
            fprintf(stderr, "EFA error at %s:%d: %s (ret=%d)\n",                 \
                    __FILE__, __LINE__, fi_strerror(-ret_), ret_);               \
            fflush(stderr);                                                      \
            throw std::runtime_error(std::string("EFA error: ") +                \
                                     fi_strerror(-ret_));                         \
        }                                                                        \
    } while (0)

// Maximum number of EFA devices per GPU on P5en (32 NICs / 8 GPUs = 4)
static constexpr int kMaxNicsPerGpu = 4;

// Each GPU maps to a specific set of EFA devices based on PCI topology
// On P5en: GPU0,1 -> EFA 0-3; GPU2,3 -> EFA 4-7; etc.
// Actually on P5en, the mapping is detected via fi_getinfo index.
// We use multiple NICs per GPU to aggregate bandwidth.
int get_efa_device_for_gpu(int gpu_id);

// Get the list of EFA device indices for a given GPU (multi-NIC)
// Returns up to kMaxNicsPerGpu indices
std::vector<int> get_efa_devices_for_gpu(int gpu_id, int total_efa_devices);

// Represents a single EFA endpoint (fabric + domain + CQ + AV + EP)
struct EfaEndpoint {
    struct fi_info* info = nullptr;
    struct fid_fabric* fabric = nullptr;
    struct fid_domain* domain = nullptr;
    struct fid_cq* tx_cq = nullptr;
    struct fid_cq* rx_cq = nullptr;
    struct fid_av* av = nullptr;
    struct fid_ep* ep = nullptr;

    // Memory region for RDMA buffer
    struct fid_mr* mr = nullptr;
    uint64_t mr_key = 0;

    // DMA-BUF file descriptor (>= 0 if DMA-BUF registration was used, -1 otherwise)
    int dmabuf_fd = -1;

    // Cumulative TX CQ error count (for diagnostics)
    uint64_t tx_errors = 0;

    // Registered MR range (for bounds checking)
    void* mr_base = nullptr;
    size_t mr_size = 0;

    // Local endpoint name for address exchange
    std::vector<uint8_t> local_name;

    // Remote addresses (indexed by remote rank)
    std::vector<fi_addr_t> remote_addrs;

    // Remote memory keys (indexed by remote rank)
    std::vector<uint64_t> remote_keys;

    // Remote buffer base addresses (indexed by remote rank)
    std::vector<uint64_t> remote_base_addrs;

    ~EfaEndpoint();
};

// Initialize an EFA endpoint on a specific EFA device
// Returns a fully setup endpoint ready for RDMA operations
void init_endpoint(EfaEndpoint& ep,
                   int efa_device_idx,
                   int cq_size = 8192);

// Register a GPU memory region for RDMA
// Uses FI_HMEM_CUDA with optional DMA-BUF support
void register_memory(EfaEndpoint& ep,
                     void* gpu_ptr,
                     size_t size,
                     uint64_t requested_key = 0);

// Exchange endpoint addresses and memory keys between all ranks
// Uses the provided allgather function (wraps torch.dist or MPI)
void exchange_addresses(EfaEndpoint& ep,
                        int rank,
                        int num_ranks,
                        void* rdma_buffer_ptr,
                        size_t rdma_buffer_size,
                        const std::function<std::vector<std::vector<uint8_t>>(
                            const std::vector<uint8_t>&)>& allgather_fn);

// Post an RDMA write with immediate data
// The immediate data is used for completion tracking on the remote side
int rdma_write_with_imm(EfaEndpoint& ep,
                        fi_addr_t remote_addr,
                        void* local_buf,
                        size_t len,
                        uint64_t remote_offset,
                        uint64_t remote_key,
                        uint64_t remote_base,
                        uint32_t imm_data,
                        void* context);

// Poll the TX completion queue for send completions
// Returns number of completed entries
int poll_tx_cq(EfaEndpoint& ep, struct fi_cq_data_entry* entries, int max_entries);

// Poll the RX completion queue for remote write completions (with imm data)
// Returns number of completed entries
int poll_rx_cq(EfaEndpoint& ep, struct fi_cq_data_entry* entries, int max_entries);

// Post receive buffers for incoming RDMA writes with immediate data
// EFA RDM requires pre-posted receives to get CQ entries for writes with imm
int post_recv(EfaEndpoint& ep, void* buf, size_t len, void* context);

// Cleanup an endpoint
void destroy_endpoint(EfaEndpoint& ep);

// ============================================================================
// EFA Transport Manager
// Manages multiple endpoints (one per EFA NIC) for multi-NIC bandwidth aggregation
// ============================================================================
class EfaTransport {
public:
    EfaTransport() = default;
    ~EfaTransport();

    // Initialize with multiple endpoints for the given GPU (multi-NIC)
    void init(int gpu_id, int rank, int num_ranks);

    // Register RDMA buffer memory with all endpoints
    void register_buffer(void* gpu_ptr, size_t size);

    // Exchange addresses with all ranks for all endpoints
    void exchange(void* rdma_buffer_ptr,
                  size_t rdma_buffer_size,
                  const std::function<std::vector<std::vector<uint8_t>>(
                      const std::vector<uint8_t>&)>& allgather_fn);

    // Get the primary endpoint (first NIC)
    EfaEndpoint& endpoint() { return endpoints_[0]; }
    const EfaEndpoint& endpoint() const { return endpoints_[0]; }

    // Get endpoint by NIC index
    EfaEndpoint& endpoint(int nic_idx) { return endpoints_[nic_idx]; }
    const EfaEndpoint& endpoint(int nic_idx) const { return endpoints_[nic_idx]; }

    // Get number of NICs being used
    int num_nics() const { return num_nics_; }

    int rank() const { return rank_; }
    int num_ranks() const { return num_ranks_; }

    bool is_initialized() const { return initialized_; }

private:
    std::vector<EfaEndpoint> endpoints_;
    int num_nics_ = 0;
    int rank_ = -1;
    int num_ranks_ = 0;
    int gpu_id_ = -1;
    bool initialized_ = false;
};

}  // namespace efa
}  // namespace deep_ep
