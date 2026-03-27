#include "efa_worker.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sched.h>

namespace deep_ep {
namespace efa {

// ============================================================================
// EfaWorker implementation
// ============================================================================

EfaWorker::~EfaWorker() {
    stop();
    if (recv_buf_) {
        free(recv_buf_);
        recv_buf_ = nullptr;
    }
}

void EfaWorker::init(EfaTransport* transport, ImmCounterMap* imm_counters) {
    transport_ = transport;
    imm_counters_ = imm_counters;

    // Allocate a small receive buffer for posted receives (for imm data)
    recv_buf_ = calloc(1, kRecvBufSize);

    running_.store(true);
    stop_requested_.store(false);

    worker_thread_ = std::thread(&EfaWorker::worker_loop, this);
}

void EfaWorker::submit_batch(const std::vector<WorkItem>& items) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        for (const auto& item : items) {
            work_queue_.push(item);
        }
        submitted_count_.fetch_add(items.size(), std::memory_order_relaxed);
    }
    queue_cv_.notify_one();
}

void EfaWorker::submit(const WorkItem& item) {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        work_queue_.push(item);
        submitted_count_.fetch_add(1, std::memory_order_relaxed);
    }
    queue_cv_.notify_one();
}

void EfaWorker::wait_all_tx_complete() {
    while (tx_completed_count_.load(std::memory_order_relaxed) <
           submitted_count_.load(std::memory_order_relaxed)) {
        // Busy-wait with a small sleep to avoid burning CPU
        std::this_thread::yield();
    }
}

int EfaWorker::poll_rx_completions(int max_completions) {
    struct fi_cq_data_entry entries[64];
    int count = std::min(max_completions, 64);

    int ret = poll_rx_cq(transport_->endpoint(), entries, count);
    if (ret > 0) {
        for (int i = 0; i < ret; ++i) {
            if (entries[i].flags & FI_REMOTE_CQ_DATA) {
                imm_counters_->record(
                    static_cast<uint32_t>(entries[i].data));
                total_rx_completions_++;
            }
        }
    }
    return ret > 0 ? ret : 0;
}

void EfaWorker::post_recv_buffers(int count) {
    for (int i = 0; i < count; ++i) {
        int ret = post_recv(transport_->endpoint(),
                            recv_buf_, kRecvBufSize, nullptr);
        if (ret != 0) {
            fprintf(stderr, "Warning: post_recv failed: %s\n",
                    fi_strerror(-ret));
        }
    }
}

void EfaWorker::stop() {
    if (running_.load()) {
        stop_requested_.store(true);
        queue_cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        running_.store(false);
    }
}

void EfaWorker::worker_loop() {
    struct fi_cq_data_entry tx_entries[64];

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        WorkItem item;
        bool has_item = false;

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!work_queue_.empty()) {
                item = work_queue_.front();
                work_queue_.pop();
                has_item = true;
            } else {
                // Wait with timeout so we can check stop flag
                queue_cv_.wait_for(lock, std::chrono::microseconds(100),
                                   [this] {
                                       return !work_queue_.empty() ||
                                              stop_requested_.load();
                                   });
                if (!work_queue_.empty()) {
                    item = work_queue_.front();
                    work_queue_.pop();
                    has_item = true;
                }
            }
        }

        if (has_item) {
            // If there's a ready_flag, poll it until the GPU signals data ready
            if (item.ready_flag) {
                auto start = std::chrono::steady_clock::now();
                while (!(*item.ready_flag)) {
                    // Spin with occasional yield
                    auto elapsed = std::chrono::steady_clock::now() - start;
                    if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 100) {
                        fprintf(stderr, "EfaWorker: timeout waiting for GPU ready flag\n");
                        break;
                    }
                    sched_yield();  // yield to other threads
                }
            }

            // Issue RDMA write
            auto& ep = transport_->endpoint();
            int ret = rdma_write_with_imm(
                ep,
                ep.remote_addrs[item.dst_rank],
                item.src_buf,
                item.src_len,
                item.dst_offset,
                ep.remote_keys[item.dst_rank],
                ep.remote_base_addrs[item.dst_rank],
                item.imm_data,
                nullptr);

            if (ret != 0 && ret != -FI_EAGAIN) {
                fprintf(stderr, "EfaWorker: rdma_write_with_imm failed: %s\n",
                        fi_strerror(-ret));
            }

            // Handle EAGAIN by retrying
            while (ret == -FI_EAGAIN) {
                // Drain some TX completions first
                int polled = poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) {
                    total_tx_completions_ += polled;
                    tx_completed_count_.fetch_add(polled, std::memory_order_relaxed);
                }
                ret = rdma_write_with_imm(
                    ep,
                    ep.remote_addrs[item.dst_rank],
                    item.src_buf,
                    item.src_len,
                    item.dst_offset,
                    ep.remote_keys[item.dst_rank],
                    ep.remote_base_addrs[item.dst_rank],
                    item.imm_data,
                    nullptr);
            }
        }

        // Always poll TX completions
        int polled = poll_tx_cq(transport_->endpoint(), tx_entries, 64);
        if (polled > 0) {
            total_tx_completions_ += polled;
            tx_completed_count_.fetch_add(polled, std::memory_order_relaxed);
        }

        // Also poll RX completions
        poll_rx_completions(64);
    }
}

// ============================================================================
// EfaWorkerManager implementation
// ============================================================================

EfaWorkerManager::~EfaWorkerManager() {
    shutdown();
}

void EfaWorkerManager::init(int gpu_id, int rank, int num_ranks) {
    transport_.init(gpu_id, rank, num_ranks);
    imm_counters_.init(num_ranks);
    initialized_ = true;
}

void EfaWorkerManager::setup_rdma(
    void* rdma_buffer_ptr,
    size_t rdma_buffer_size,
    const std::function<std::vector<std::vector<uint8_t>>(
        const std::vector<uint8_t>&)>& allgather_fn) {
    transport_.register_buffer(rdma_buffer_ptr, rdma_buffer_size);
    transport_.exchange(rdma_buffer_ptr, rdma_buffer_size, allgather_fn);

    // NOTE: We do NOT start the worker thread here.
    // For Iteration 2/3, efa_all_to_all() does synchronous RDMA directly
    // from the calling thread. The worker thread will be used in future
    // iterations for the async split-kernel approach.
    
    // Pre-post receive buffers for incoming writes with immediate data
    // EFA RDM requires pre-posted receives to deliver CQ entries for
    // RDMA writes with immediate data.
    // Post receives on ALL endpoints (all NICs).
    void* recv_buf = calloc(1, 64);  // Small buffer for imm-only writes
    int num_nics = transport_.num_nics();
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport_.endpoint(n);
        for (int i = 0; i < transport_.num_ranks() * 256; ++i) {
            post_recv(ep, recv_buf, 64, nullptr);
        }
    }
    // Store for later cleanup
    recv_staging_buf_ = recv_buf;
}

void EfaWorkerManager::rdma_write(
    void* local_buf, size_t len,
    int dst_rank, uint64_t dst_offset,
    uint32_t imm_data,
    volatile uint8_t* ready_flag) {
    WorkItem item;
    item.src_buf = local_buf;
    item.src_len = len;
    item.dst_rank = dst_rank;
    item.dst_offset = dst_offset;
    item.imm_data = imm_data;
    item.ready_flag = ready_flag;
    item.seq = 0;
    worker_.submit(item);
}

void EfaWorkerManager::wait_tx_complete() {
    worker_.wait_all_tx_complete();
}

int EfaWorkerManager::poll_completions(int max_completions) {
    return worker_.poll_rx_completions(max_completions);
}

void EfaWorkerManager::post_recvs(int count) {
    worker_.post_recv_buffers(count);
}

void EfaWorkerManager::barrier(
    const std::function<std::vector<std::vector<uint8_t>>(
        const std::vector<uint8_t>&)>& allgather_fn) {
    // Simple barrier via allgather
    std::vector<uint8_t> dummy = {1};
    allgather_fn(dummy);
}

void EfaWorkerManager::shutdown() {
    if (initialized_) {
        worker_.stop();
        if (recv_staging_buf_) {
            free(recv_staging_buf_);
            recv_staging_buf_ = nullptr;
        }
        initialized_ = false;
    }
}

}  // namespace efa
}  // namespace deep_ep
