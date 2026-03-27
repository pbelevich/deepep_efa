#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "efa_transport.h"
#include "gdr_signal.h"
#include "imm_counter.h"

namespace deep_ep {
namespace efa {

// ============================================================================
// WorkItem: Represents a single RDMA write request
// The GPU kernel writes data to a staging buffer and sets a GdrFlag.
// The EfaWorker thread detects the flag, then initiates fi_writemsg.
// ============================================================================
struct WorkItem {
    // Source (local GPU buffer) for the RDMA write
    void* src_buf;
    size_t src_len;

    // Destination info
    int dst_rank;
    uint64_t dst_offset;

    // Immediate data for completion tracking
    uint32_t imm_data;

    // Flag that GPU sets when data is ready
    volatile uint8_t* ready_flag;

    // Sequence number for ordering
    uint64_t seq;
};

// ============================================================================
// EfaWorker: CPU-side worker thread that polls GdrFlags and initiates RDMA
//
// Architecture (following pplx-garden's pattern):
// 1. GPU kernel writes token data to local RDMA buffer
// 2. GPU kernel sets GdrFlag via MMIO store (1 byte)
// 3. EfaWorker polls GdrFlag via GDRCopy mapping (~1us)
// 4. EfaWorker calls fi_writemsg to issue RDMA write
// 5. EfaWorker polls TX CQ for local completion
// 6. Remote RX CQ receives immediate data -> ImmCounter tracks completion
// ============================================================================
class EfaWorker {
public:
    EfaWorker() = default;
    ~EfaWorker();

    // Initialize the worker thread
    void init(EfaTransport* transport, ImmCounterMap* imm_counters);

    // Submit a batch of work items (called from the host during kernel execution)
    // The worker will poll the ready_flags and issue RDMA writes as they become ready
    void submit_batch(const std::vector<WorkItem>& items);

    // Submit a single work item
    void submit(const WorkItem& item);

    // Wait for all submitted work to complete (all TX CQ entries received)
    void wait_all_tx_complete();

    // Poll the RX CQ for incoming RDMA writes and record immediate data
    // Returns number of completions processed
    int poll_rx_completions(int max_completions = 64);

    // Post receive buffers for incoming writes with imm data
    void post_recv_buffers(int count);

    // Stop the worker thread
    void stop();

    // Check if worker is running
    bool is_running() const { return running_.load(); }

    // Get statistics
    uint64_t get_total_tx_completions() const { return total_tx_completions_; }
    uint64_t get_total_rx_completions() const { return total_rx_completions_; }

private:
    void worker_loop();

    EfaTransport* transport_ = nullptr;
    ImmCounterMap* imm_counters_ = nullptr;

    // Work queue
    std::queue<WorkItem> work_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Worker thread
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Completion tracking
    std::atomic<uint64_t> submitted_count_{0};
    std::atomic<uint64_t> tx_completed_count_{0};
    uint64_t total_tx_completions_ = 0;
    uint64_t total_rx_completions_ = 0;

    // Receive buffer for posted receives (dummy - EFA RDM requires it)
    void* recv_buf_ = nullptr;
    static constexpr size_t kRecvBufSize = 64;  // Minimal buffer for imm-only writes
};

// ============================================================================
// EfaWorkerManager: Manages the worker and provides the high-level interface
// for the internode dispatch/combine operations
// ============================================================================
class EfaWorkerManager {
public:
    EfaWorkerManager() = default;
    ~EfaWorkerManager();

    // Initialize: creates transport, worker, counters
    void init(int gpu_id, int rank, int num_ranks);

    // Register RDMA buffer and exchange addresses
    void setup_rdma(void* rdma_buffer_ptr, size_t rdma_buffer_size,
                    const std::function<std::vector<std::vector<uint8_t>>(
                        const std::vector<uint8_t>&)>& allgather_fn);

    // Issue an RDMA write from local GPU buffer to remote rank's RDMA buffer
    // The write happens asynchronously via the worker thread
    void rdma_write(void* local_buf, size_t len,
                    int dst_rank, uint64_t dst_offset,
                    uint32_t imm_data,
                    volatile uint8_t* ready_flag = nullptr);

    // Wait for all pending RDMA writes to complete locally
    void wait_tx_complete();

    // Poll for remote RDMA write completions and update counters
    int poll_completions(int max_completions = 64);

    // Post receive buffers
    void post_recvs(int count);

    // Get the ImmCounter for tracking remote completions
    ImmCounterMap& counters() { return imm_counters_; }

    // Get the transport
    EfaTransport& transport() { return transport_; }

    // Barrier across all ranks (uses the allgather function internally)
    void barrier(const std::function<std::vector<std::vector<uint8_t>>(
                     const std::vector<uint8_t>&)>& allgather_fn);

    // Shutdown
    void shutdown();

    bool is_initialized() const { return initialized_; }

private:
    EfaTransport transport_;
    EfaWorker worker_;
    ImmCounterMap imm_counters_;
    bool initialized_ = false;
    void* recv_staging_buf_ = nullptr;  // Small buffer for pre-posted receives
};

}  // namespace efa
}  // namespace deep_ep
