#include "ll_pipeline.h"
#include <sched.h>
#include <algorithm>
#include <immintrin.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace deep_ep {
namespace efa {

LLPipelineWorker::~LLPipelineWorker() {
    stop();
}

void LLPipelineWorker::init(EfaTransport* transport, ImmCounterMap* counters,
                             const LLPipelineConfig& config) {
    transport_ = transport;
    counters_ = counters;
    config_ = config;

    // Initialize GDR flags
    dispatch_pack_done_.init();
    dispatch_recv_done_.init();
    combine_pack_done_.init();
    combine_recv_done_.init();

    // Initialize GDR-mapped send/recv count vectors
    dispatch_send_counts_.init(config.num_ranks);
    dispatch_recv_counts_.init(config.num_ranks);
    combine_send_counts_.init(config.num_ranks);
    combine_recv_counts_.init(config.num_ranks);

    // Clear all flags
    dispatch_pack_done_.clear();
    dispatch_recv_done_.clear();
    combine_pack_done_.clear();
    combine_recv_done_.clear();

    // Start worker thread
    running_.store(true, std::memory_order_release);
    stop_requested_.store(false, std::memory_order_release);
    pending_command_.store(0, std::memory_order_release);
    command_done_.store(false, std::memory_order_release);

    worker_thread_ = std::thread(&LLPipelineWorker::worker_loop, this);

    // Try to pin to a nearby CPU core (best effort)
    // On P5en, GPUs 0-7 map to NUMA nodes, we try to pin to cores near the GPU
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    // Use GPU ID (local_rank % 8) to compute nearby cores
    // P5en has 192 cores / 8 GPUs ≈ 24 cores per GPU
    int gpu_local = config.local_rank % 8;
    int target_core = gpu_local * 24 + 20;  // Use a core not too close to 0
    if (target_core < 192) {
        CPU_SET(target_core, &cpuset);
        pthread_setaffinity_np(worker_thread_.native_handle(), sizeof(cpuset), &cpuset);
    }
}

void LLPipelineWorker::update_config(const LLPipelineConfig& config) {
    config_ = config;
}

void LLPipelineWorker::set_combine_recv_counts(const std::vector<int32_t>& counts) {
    for (size_t i = 0; i < counts.size() && i < static_cast<size_t>(config_.num_ranks); ++i) {
        combine_recv_counts_.write(i, counts[i]);
    }
}

void LLPipelineWorker::set_combine_send_counts(const std::vector<int32_t>& counts) {
    for (size_t i = 0; i < counts.size() && i < static_cast<size_t>(config_.num_ranks); ++i) {
        combine_send_counts_.write(i, counts[i]);
    }
}

void LLPipelineWorker::start_dispatch() {
    command_done_.store(false, std::memory_order_release);
    pending_command_.store(static_cast<int>(Command::DISPATCH), std::memory_order_release);
}

void LLPipelineWorker::start_combine() {
    command_done_.store(false, std::memory_order_release);
    pending_command_.store(static_cast<int>(Command::COMBINE), std::memory_order_release);
}

void LLPipelineWorker::start_coop_dispatch() {
    coop_done_.store(false, std::memory_order_release);
    pending_command_.store(static_cast<int>(Command::COOP_DISPATCH), std::memory_order_release);
}

void LLPipelineWorker::start_coop_combine() {
    coop_done_.store(false, std::memory_order_release);
    pending_command_.store(static_cast<int>(Command::COOP_COMBINE), std::memory_order_release);
}

void LLPipelineWorker::wait_dispatch_done() {
    while (!command_done_.load(std::memory_order_acquire)) {
        // Spin
        _mm_pause();
    }
}

void LLPipelineWorker::wait_combine_done() {
    while (!command_done_.load(std::memory_order_acquire)) {
        // Spin
        _mm_pause();
    }
}

void LLPipelineWorker::wait_coop_dispatch_done() {
    while (!coop_done_.load(std::memory_order_acquire)) {
        _mm_pause();
    }
}

void LLPipelineWorker::wait_coop_combine_done() {
    while (!coop_done_.load(std::memory_order_acquire)) {
        _mm_pause();
    }
}

void LLPipelineWorker::efa_barrier() {
    command_done_.store(false, std::memory_order_release);
    pending_command_.store(static_cast<int>(Command::BARRIER), std::memory_order_release);
    // Wait for barrier to complete
    while (!command_done_.load(std::memory_order_acquire)) {
        _mm_pause();
    }
}

void LLPipelineWorker::post_initial_recvs() {
    static char recv_dummy[64] = {};
    int num_nics = transport_->num_nics();
    // Post a large initial pool matching efa_all_to_all: num_ranks * 256 per NIC.
    // This ensures we never run out of recv buffers during multi-stage coop operations.
    int per_nic = config_.num_ranks * 256;
    for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
        auto& ep = transport_->endpoint(n);
        int posted = 0;
        for (int i = 0; i < per_nic; ++i) {
            int ret = post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
            if (ret == 0) {
                posted++;
            } else if (ret == -FI_EAGAIN) {
                // Recv queue full, stop
                break;
            } else {
                fprintf(stderr, "LLPipeline: post_recv failed NIC=%d: %s\n",
                        n, fi_strerror(-ret));
                break;
            }
        }
        posted_per_nic_[n] = posted;
    }
}

void LLPipelineWorker::stop() {
    if (running_.load(std::memory_order_relaxed)) {
        stop_requested_.store(true, std::memory_order_release);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        running_.store(false, std::memory_order_release);
    }
}

// ============================================================================
// Worker thread main loop
// ============================================================================
void LLPipelineWorker::worker_loop() {
    // Set CUDA device for this thread (needed for cudaMemcpy and cudaDeviceSynchronize)
    cudaSetDevice(config_.device_id);

    while (!stop_requested_.load(std::memory_order_relaxed)) {
        int cmd = pending_command_.load(std::memory_order_acquire);
        if (cmd == 0) {
            // No command pending, spin
            _mm_pause();
            continue;
        }

        Command command = static_cast<Command>(cmd);
        pending_command_.store(0, std::memory_order_release);

        switch (command) {
            case Command::DISPATCH: {
                // In cooperative mode, the GPU kernel is still packing when
                // start_dispatch() is called. Wait for dispatch_pack_done_.
                if (coop_mode_) {
                    auto t_start = std::chrono::high_resolution_clock::now();
                    int last_warn_s = 0;
                    while (dispatch_pack_done_.read() == 0) {
                        _mm_pause();
                        auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
                        int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                        if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                            fprintf(stderr, "LLPipeline: dispatch pack_done timeout (coop mode) rank=%d %ds\n",
                                    config_.local_rank, elapsed_s);
                            last_warn_s = elapsed_s;
                        }
                    }
                    dispatch_pack_done_.clear();
                }

                // NOTE: By the time we get here, start_dispatch() has already:
                //   1. Written recv_counts to GDR-mapped dispatch_recv_counts_
                //   2. Posted this command
                // The Python caller guaranteed that:
                //   - NCCL count exchange completed
                //   - .tolist() synced the default stream
                //   - gdr_signal_counts kernel wrote send_counts + pack_done flag
                //     (on default stream, before .tolist() returned)
                // So both send_counts and recv_counts are available, and GPU data
                // is committed to HBM (default stream synced).

                // Step 1: Read send counts and recv counts from GDR-mapped memory
                // (Both are already written: send_counts by GPU kernel, recv_counts
                // by Python via start_dispatch)
                int32_t send_counts[kMaxLLRanks];
                int32_t recv_counts[kMaxLLRanks];
                int total_send = 0;
                for (int i = 0; i < config_.num_ranks; ++i) {
                    send_counts[i] = dispatch_send_counts_.read(i);
                    recv_counts[i] = dispatch_recv_counts_.read(i);
                    total_send += send_counts[i];
                }

                // Step 2: Post receives for incoming data
                post_recvs_for_transfer(config_.num_ranks);

                // Step 3: Issue RDMA writes
                issue_rdma_writes(send_counts, recv_counts,
                                  config_.send_data_base,
                                  config_.recv_data_rdma_offset,
                                  config_.slot_size,
                                  true /* fixed_slot for dispatch */);

                // NOTE: We do NOT call cudaDeviceSynchronize() here.
                // DMA-BUF RDMA writes bypass GPU L2 cache, so subsequent CUDA
                // kernels that read from the recv buffer MUST use ld.global.nc
                // (non-coherent loads) to see the RDMA-written data.
                // The existing ll_recv_unpack kernel already uses nc loads.

                // Step 4: Signal completion
                dispatch_recv_done_.set(1);
                command_done_.store(true, std::memory_order_release);
                break;
            }

            case Command::COMBINE: {
                // Wait for GPU to signal combine data is packed
                auto t_start = std::chrono::high_resolution_clock::now();
                int last_warn_s = 0;
                while (combine_pack_done_.read() == 0) {
                    _mm_pause();
                    auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
                    int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                    if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                        fprintf(stderr, "LLPipeline: combine pack_done timeout rank=%d %ds\n",
                                config_.local_rank, elapsed_s);
                        last_warn_s = elapsed_s;
                    }
                }
                combine_pack_done_.clear();

                // Read counts from GDR-mapped memory
                int32_t send_counts[kMaxLLRanks];
                int32_t recv_counts[kMaxLLRanks];
                for (int i = 0; i < config_.num_ranks; ++i) {
                    send_counts[i] = combine_send_counts_.read(i);
                    recv_counts[i] = combine_recv_counts_.read(i);
                }

                // Post receives
                post_recvs_for_transfer(config_.num_ranks);

                // Issue RDMA writes
                issue_rdma_writes(send_counts, recv_counts,
                                  config_.send_data_base,
                                  config_.recv_data_rdma_offset,
                                  config_.slot_size,
                                  true /* fixed_slot for combine too */);

                combine_recv_done_.set(1);
                command_done_.store(true, std::memory_order_release);
                break;
            }

            case Command::BARRIER: {
                // EFA-native barrier: send imm-only RDMA writes to all remote peers
                // and wait for theirs. This replaces dist.barrier().
                uint16_t tag = current_tag_++;
                counters_->reset_tag(tag);

                // Ensure recv pool has enough for barrier
                ensure_recv_pool(std::max(config_.num_ranks * 4, 256));

                // Send zero-byte RDMA writes with immediate data to all remote peers
                int num_expected = 0;
                int total_writes = 0;
                for (int i = 0; i < config_.num_ranks; ++i) {
                    if (i == config_.local_rank) continue;
                    num_expected++;  // Expect 1 imm from each peer

                    // Use NIC 0 for barrier (tiny messages)
                    auto& ep = transport_->endpoint(0);
                    uint32_t imm = ImmCounterMap::encode(tag, config_.local_rank, 1);

                    // Zero-byte write with imm data (just for the CQ notification)
                    // We need a valid remote address but write 0 bytes
                    int ret = rdma_write_with_imm(
                        ep, ep.remote_addrs[i],
                        config_.rdma_base, 0,  // 0-byte write
                        0, ep.remote_keys[i], ep.remote_base_addrs[i],
                        imm, nullptr);

                    // Handle EAGAIN
                    struct fi_cq_data_entry tx_entries[64];
                    while (ret == -FI_EAGAIN) {
                        poll_tx_cq(ep, tx_entries, 64);
                        ret = rdma_write_with_imm(
                            ep, ep.remote_addrs[i],
                            config_.rdma_base, 0,
                            0, ep.remote_keys[i], ep.remote_base_addrs[i],
                            imm, nullptr);
                    }
                    total_writes++;
                }

                // Wait for all TX completions and all RX completions
                wait_all_completions(total_writes, num_expected, tag);

                command_done_.store(true, std::memory_order_release);
                break;
            }

            case Command::COOP_DISPATCH: {
                handle_coop_dispatch();
                coop_done_.store(true, std::memory_order_release);
                break;
            }

            case Command::COOP_COMBINE: {
                handle_coop_combine();
                coop_done_.store(true, std::memory_order_release);
                break;
            }

            default:
                break;
        }
    }
}

// ============================================================================
// Issue RDMA writes for a transfer
// ============================================================================
void LLPipelineWorker::issue_rdma_writes(
    const int32_t* send_counts, const int32_t* recv_counts,
    uint8_t* send_base, uint64_t recv_data_rdma_offset,
    uint64_t slot_size, bool use_fixed_slot) {

    int num_nics = transport_->num_nics();
    int bpt = config_.packed_bytes_per_token;
    uint16_t tag = current_tag_++;
    counters_->reset_tag(tag);

    // Compute send offsets (cumulative sum)
    int64_t send_offsets[kMaxLLRanks];
    int64_t send_sizes[kMaxLLRanks];
    int64_t s_off = 0;
    for (int i = 0; i < config_.num_ranks; ++i) {
        send_offsets[i] = s_off;
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * bpt;
        s_off += send_sizes[i];
    }

    // Compute recv offsets
    int64_t recv_offsets_arr[kMaxLLRanks];
    int64_t recv_sizes_arr[kMaxLLRanks];
    for (int i = 0; i < config_.num_ranks; ++i) {
        recv_sizes_arr[i] = static_cast<int64_t>(recv_counts[i]) * bpt;
        if (use_fixed_slot) {
            recv_offsets_arr[i] = static_cast<int64_t>(config_.local_rank) * static_cast<int64_t>(slot_size);
        } else {
            recv_offsets_arr[i] = static_cast<int64_t>(config_.local_rank) * static_cast<int64_t>(slot_size);
        }
    }

    // Drain leftover RX CQ entries from previous transfers on ALL NICs
    // (matches efa_all_to_all behavior — prevents stale entries from interfering)
    for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
        auto& ep = transport_->endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = poll_rx_cq(ep, drain_entries, 64);
            if (drained > 0) {
                posted_per_nic_[n] -= drained;
                for (int j = 0; j < drained; ++j) {
                    if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(drain_entries[j].data));
                    }
                }
            }
        } while (drained > 0);
    }

    // Local copy: self-send data needs to go into our own recv buffer
    // (matches efa_all_to_all's i == rank case with cudaMemcpyAsync)
    if (send_sizes[config_.local_rank] > 0) {
        uint8_t* src = send_base + send_offsets[config_.local_rank];
        uint8_t* dst = config_.recv_data_base +
                       static_cast<int64_t>(config_.local_rank) * static_cast<int64_t>(slot_size);
        cudaMemcpy(dst, src, static_cast<size_t>(send_sizes[config_.local_rank]),
                   cudaMemcpyDeviceToDevice);
    }

    // Count expected incoming writes (considering multi-NIC sharding)
    int num_expected_recvs = 0;
    for (int i = 0; i < config_.num_ranks; ++i) {
        if (i == config_.local_rank || recv_sizes_arr[i] <= 0) continue;
        size_t total_bytes = static_cast<size_t>(recv_sizes_arr[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);
        if (bytes_per_shard == 0) bytes_per_shard = kShardAlign;
        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;
            num_expected_recvs++;
            shard_offset += shard_len;
        }
    }

    // Issue RDMA writes, sharding across NICs
    int total_writes = 0;
    int writes_per_nic[4] = {};  // Up to kMaxNicsPerGpu
    int tx_completed[4] = {};
    int total_tx_completed = 0;

    for (int i = 0; i < config_.num_ranks; ++i) {
        if (send_sizes[i] <= 0 || i == config_.local_rank) continue;

        // Shard across NICs
        size_t total_bytes = static_cast<size_t>(send_sizes[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);

        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;

            auto& ep = transport_->endpoint(n);

            // Flow control
            while (writes_per_nic[n] - tx_completed[n] >= kMaxOutstandingPerNic) {
                struct fi_cq_data_entry drain[64];
                int polled = poll_tx_cq(ep, drain, 64);
                if (polled > 0) {
                    tx_completed[n] += polled;
                    total_tx_completed += polled;
                }
                // Also poll RX to avoid deadlock
                struct fi_cq_data_entry rx_drain[64];
                int rx_polled = poll_rx_cq(ep, rx_drain, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_drain[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_drain[j].data));
                        }
                    }
                }
            }

            void* src = send_base + send_offsets[i] + shard_offset;
            uint64_t dst_offset = recv_data_rdma_offset + recv_offsets_arr[i] + shard_offset;
            uint32_t imm = ImmCounterMap::encode(tag, config_.local_rank, 1);

            int ret = rdma_write_with_imm(
                ep, ep.remote_addrs[i], src, shard_len,
                dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                imm, nullptr);

            // Handle EAGAIN
            struct fi_cq_data_entry tx_entries[64];
            while (ret == -FI_EAGAIN) {
                int polled = poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) {
                    tx_completed[n] += polled;
                    total_tx_completed += polled;
                }
                struct fi_cq_data_entry rx_entries[64];
                int rx_polled = poll_rx_cq(ep, rx_entries, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_entries[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_entries[j].data));
                        }
                    }
                }
                ret = rdma_write_with_imm(
                    ep, ep.remote_addrs[i], src, shard_len,
                    dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                    imm, nullptr);
            }

            if (ret != 0) {
                fprintf(stderr, "LLPipeline RDMA write failed: peer=%d nic=%d: %s\n",
                        i, n, fi_strerror(-ret));
            }
            writes_per_nic[n]++;
            total_writes++;
            shard_offset += shard_len;
        }
    }

    // Wait for all completions
    wait_all_completions(total_writes - total_tx_completed, num_expected_recvs, tag);

    // Drain remaining TX CQ entries on all NICs (prevents stale entries for next transfer)
    for (int n = 0; n < num_nics; ++n) {
        auto& ep = transport_->endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = poll_tx_cq(ep, drain_entries, 64);
        } while (drained > 0);
    }
}

// ============================================================================
// Wait for TX and RX completions (with recv pool replenishment)
// ============================================================================
void LLPipelineWorker::wait_all_completions(int num_remaining_tx, int num_expected_rx,
                                             uint16_t tag, int base_rx) {
    int num_nics = transport_->num_nics();
    auto start_time = std::chrono::high_resolution_clock::now();
    struct fi_cq_data_entry cq_entries[64];
    int tx_remaining = num_remaining_tx;
    int target_rx = base_rx + num_expected_rx;
    int last_warn_s = 0;

    // Target recv pool level for replenishment during wait
    int recv_replenish_target = std::max(config_.num_ranks * 4, 256);

    while (tx_remaining > 0 || counters_->get_total(tag) < target_rx) {
        for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
            auto& ep = transport_->endpoint(n);

            // Poll TX CQ
            if (tx_remaining > 0) {
                int polled = poll_tx_cq(ep, cq_entries, 64);
                if (polled > 0) tx_remaining -= polled;
            }

            // Poll RX CQ — decrement posted_per_nic_ for each consumed recv
            int polled = poll_rx_cq(ep, cq_entries, 64);
            if (polled > 0) {
                posted_per_nic_[n] -= polled;
                for (int j = 0; j < polled; ++j) {
                    if (cq_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(cq_entries[j].data));
                    }
                }
            }

            // Replenish recv pool if it drops too low (prevents starvation)
            if (posted_per_nic_[n] < 32) {
                static char recv_dummy[64] = {};
                int to_post = recv_replenish_target - posted_per_nic_[n];
                for (int rp = 0; rp < to_post; ++rp) {
                    int ret = post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
                    if (ret == 0) {
                        posted_per_nic_[n]++;
                    } else {
                        break;
                    }
                }
            }
        }

        // Timeout check
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        if (elapsed_s > 0 && elapsed_s % 30 == 0) {
            if (elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline CQ wait timeout: rank=%d tx_rem=%d rx=%d/%d tag=%u %ds pool=[",
                        config_.local_rank, tx_remaining,
                        counters_->get_total(tag) - base_rx, num_expected_rx,
                        static_cast<unsigned>(tag), elapsed_s);
                for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
                    fprintf(stderr, "%d%s", posted_per_nic_[n], n < num_nics-1 ? "," : "");
                }
                fprintf(stderr, "] per_rank=[");
                for (int r = 0; r < config_.num_ranks; ++r) {
                    fprintf(stderr, "%d%s", counters_->get_count(tag, r),
                            r < config_.num_ranks-1 ? "," : "");
                }
                fprintf(stderr, "]\n");
                last_warn_s = elapsed_s;
            }
        }
    }
}

// ============================================================================
// Ensure enough recvs are posted on all NICs (incremental replenishment)
// Mirrors efa_all_to_all's g_posted_per_nic approach.
// ============================================================================
void LLPipelineWorker::ensure_recv_pool(int target_per_nic) {
    static char recv_dummy[64] = {};
    int num_nics = transport_->num_nics();
    for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
        auto& ep = transport_->endpoint(n);
        int needed = target_per_nic - posted_per_nic_[n];
        for (int i = 0; i < needed; ++i) {
            int ret = post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
            if (ret == -FI_EAGAIN) {
                // Drain RX CQ to free recv slots, then retry
                struct fi_cq_data_entry drain[64];
                int drained = poll_rx_cq(ep, drain, 64);
                if (drained > 0) {
                    posted_per_nic_[n] -= drained;
                    for (int j = 0; j < drained; ++j) {
                        if (drain[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(drain[j].data));
                        }
                    }
                }
                ret = post_recv(ep, recv_dummy, sizeof(recv_dummy), nullptr);
            }
            if (ret == 0) {
                posted_per_nic_[n]++;
            }
        }
    }
}

// Keep old function name as a thin wrapper for compatibility
void LLPipelineWorker::post_recvs_for_transfer(int /* num_peers */) {
    // Target: enough for one full round of transfers + headroom
    int target = std::max(config_.num_ranks * 4, 256);
    ensure_recv_pool(target);
}

// ============================================================================
// Cooperative Dispatch Handler (Iter 56)
//
// Runs on the worker thread, overlapped with cooperative GPU kernels.
//
// Iter 56 optimizations:
// - Overlap dispatch_pack_done wait with route exchange
// - After pack_done: submit dispatch RDMA writes immediately using local send counts
// - Meanwhile: route info arriving overlaps with RDMA round-trip
// - After routes arrive: process metadata, signal GPU
// - After RDMA complete: signal dispatch_recv_flag
//
// Flow:
//   1. Wait dispatch_route_done → read num_routed from GPU
//   2. Compute LOCAL send_counts from our routing data
//   3. Fire route scatter RDMA (with TX wait for buffer safety)
//   4. OVERLAPPED: wait for both route info AND dispatch_pack_done
//   5. After pack_done: process route info if ready, signal metadata
//   6. Issue dispatch RDMA writes + wait completions
//   7. Signal dispatch_recv_flag → wait recv_done → signal tx_ready
// ============================================================================
void LLPipelineWorker::handle_coop_dispatch() {
    auto& cc = coop_config_;
    int nr = config_.num_ranks;
    int lr = config_.local_rank;
    int num_experts = cc.num_experts;
    int experts_per_rank = cc.experts_per_rank;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto us_since = [&t0]() {
        return std::chrono::duration<double, std::micro>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };

    fprintf(stderr, "[COOP r%d] entering dispatch, prealloc=%d, pool=[%d,%d]\n",
            lr, (int)tags_preallocated_,
            posted_per_nic_[0], posted_per_nic_[1]);

    // Step 1: Wait for dispatch_route_done (GPU counted tokens per expert)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int last_warn_s = 0;
        while (cc.dispatch_route_done->read() == 0) {
            _mm_pause();
            auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
            int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline COOP: dispatch_route_done waiting rank=%d %ds\n", lr, elapsed_s);
                last_warn_s = elapsed_s;
            }
        }
        cc.dispatch_route_done->clear();
    }
    if (lr == 0) fprintf(stderr, "[COOP r%d] step1 route_done %.0fus\n", lr, us_since());

     // TAG MANAGEMENT: Absolute-target counting (no reset, no snapshot baseline).
     // Iter 61: Tags are reset between bench phases via coop_reset_tags() to handle
     // the kineto-phase tag wrapping issue (tags > 1024).
     uint16_t route_scatter_tag, dispatch_data_tag;
     int route_scatter_base = 0, dispatch_data_base = 0;

     if (!tags_preallocated_) {
         route_scatter_tag = current_tag_++;
         dispatch_data_tag = current_tag_++;
         preallocated_combine_data_tag_ = current_tag_++;
         preallocated_combine_data_base_ = 0;
     } else {
         route_scatter_tag = preallocated_route_scatter_tag_;
         dispatch_data_tag = preallocated_dispatch_data_tag_;
         preallocated_combine_data_base_ = 0;

         if (lr == 0) fprintf(stderr, "[COOP r%d] tag status: route=%u(total=%d) disp=%u(total=%d) comb=%u(total=%d)\n",
                              lr, (unsigned)route_scatter_tag, counters_->get_total(route_scatter_tag),
                              (unsigned)dispatch_data_tag, counters_->get_total(dispatch_data_tag),
                              (unsigned)preallocated_combine_data_tag_, counters_->get_total(preallocated_combine_data_tag_));
     }

    // Step 2: Read num_routed from GPU via cudaMemcpy D2H.
    uint32_t local_num_routed[kMaxExperts];
    cudaMemcpy(local_num_routed, cc.gdr_num_routed->device_ptr(),
               num_experts * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Step 3: RDMA scatter routing info (with TX wait for buffer safety)
    rdma_scatter_route_info_async(local_num_routed, num_experts, route_scatter_tag);
    if (lr == 0) fprintf(stderr, "[COOP r%d] step3 scatter %.0fus\n", lr, us_since());

    // Iter 57: Arm GdrCounter BEFORE the overlapped wait loop so that any dispatch_data_tag
    // CQ entries that arrive early (during route/pack_done wait) trigger inc() immediately.
    if (cc.dispatch_rdma_counter) {
        cc.dispatch_rdma_counter->reset();  // Zero counter + clear dispatch_recv_flag
        active_gdr_counter_ = cc.dispatch_rdma_counter;
        active_counter_tag_ = dispatch_data_tag;

        // Iter 58: Replay pre-consumed dispatch CQ entries.
        // If entries for dispatch_data_tag were polled during a previous phase (e.g.,
        // combine CQ polling of the prior iteration), they were recorded in ImmCounterMap
        // but the GdrCounter wasn't armed yet. Replay them now so the counter is accurate.
        int pre_consumed = counters_->get_total(dispatch_data_tag);
        if (pre_consumed > 0) {
            for (int k = 0; k < pre_consumed; ++k) {
                cc.dispatch_rdma_counter->inc();
            }
            if (lr == 0) fprintf(stderr, "[COOP r%d] replayed %d pre-consumed dispatch CQ entries\n",
                                 lr, pre_consumed);
        }
    }

    // Step 4: OVERLAPPED WAIT — wait for BOTH route info AND dispatch_pack_done.
    int route_num_expected = nr - 1;
    int route_target_rx = route_scatter_base + route_num_expected;
    bool route_done = false;
    bool pack_done = false;

    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int last_warn_s = 0;
        int num_nics = transport_->num_nics();

        while (!route_done || !pack_done) {
            for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
                struct fi_cq_data_entry entries[64];
                int polled = poll_rx_cq(transport_->endpoint(n), entries, 64);
                if (polled > 0) {
                    posted_per_nic_[n] -= polled;
                    for (int j = 0; j < polled; ++j) {
                        if (entries[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(entries[j].data));
                        }
                    }
                }
                struct fi_cq_data_entry tx_entries[64];
                poll_tx_cq(transport_->endpoint(n), tx_entries, 64);
            }

            if (!route_done && counters_->get_total(route_scatter_tag) >= route_target_rx) {
                route_done = true;
                if (lr == 0) fprintf(stderr, "[COOP r%d] step4a route_done %.0fus\n", lr, us_since());
            }
            if (!pack_done && cc.dispatch_pack_done->read() != 0) {
                pack_done = true;
                cc.dispatch_pack_done->clear();
                if (lr == 0) fprintf(stderr, "[COOP r%d] step4b pack_done %.0fus\n", lr, us_since());
            }

            auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
            int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline COOP: overlapped wait rank=%d route=%d/%d pack=%d %ds pool=[",
                        lr, counters_->get_total(route_scatter_tag) - route_scatter_base,
                        route_num_expected, (int)pack_done, elapsed_s);
                int num_nics_log = transport_->num_nics();
                for (int n = 0; n < num_nics_log && n < kMaxNicsPerGpu; ++n) {
                    fprintf(stderr, "%d%s", posted_per_nic_[n], n < num_nics_log-1 ? "," : "");
                }
                fprintf(stderr, "] per_rank=[");
                for (int r = 0; r < nr; ++r) {
                    fprintf(stderr, "%d%s", counters_->get_count(route_scatter_tag, r),
                            r < nr-1 ? "," : "");
                }
                fprintf(stderr, "]\n");
                last_warn_s = elapsed_s;
            }
            _mm_pause();
        }
    }
    if (lr == 0) fprintf(stderr, "[COOP r%d] step4 overlapped_done %.0fus\n", lr, us_since());

    // Step 5: Read route data D2H and compute send/recv counts (fast, ~12us)
    // Iter 62: If any route value is corrupt, ALL counts are zeroed (symmetric skip).
    // With route send buffer moved to tail of send half, corruption should no longer occur.
    {
        size_t route_bytes = num_experts * sizeof(uint32_t);
        uint8_t* route_recv_base = config_.rdma_base + config_.half_rdma;
        if (nr <= 16) {
            cudaMemcpy(route_recv_buf_, route_recv_base,
                       nr * route_bytes, cudaMemcpyDeviceToHost);
        } else {
            for (int i = 0; i < nr; ++i) {
                if (i == lr) continue;
                cudaMemcpy(&route_recv_buf_[i * num_experts],
                           route_recv_base + i * route_bytes,
                           route_bytes, cudaMemcpyDeviceToHost);
            }
        }
        memcpy(&route_recv_buf_[lr * num_experts], local_num_routed,
               num_experts * sizeof(uint32_t));

        compute_send_recv_counts(route_recv_buf_, num_experts, experts_per_rank, nr, lr);
    }
    if (lr == 0) fprintf(stderr, "[COOP r%d] step5a counts_ready, send=%d recv=%d corrupted=%d %.0fus\n",
                         lr, coop_total_send_, coop_total_recv_, (int)coop_data_corrupted_, us_since());

    // Iter 62: If route data was corrupt, ALL counts are zero. All ranks skip this
    // iteration symmetrically (nobody sends, nobody waits). Wrong results for this
    // iteration but prevents deadlocks.

    // Step 6: Submit dispatch RDMA writes (non-blocking).
    // GdrCounter was armed in step 3 and replay applied. submit_coop_dispatch_rdma
    // sets the GdrCounter target via wait(N). When the last CQ inc() fires,
    // dispatch_recv_flag auto-sets via MMIO — potentially while we're still
    // doing metadata upload below.
    auto rdma_state = submit_coop_dispatch_rdma(coop_send_counts_, coop_recv_counts_,
                                                 cc.coop_token_stride, cc.max_tokens_per_rank,
                                                 dispatch_data_tag, dispatch_data_base);
    if (lr == 0) fprintf(stderr, "[COOP r%d] step6a rdma_submitted, writes=%d expected_rx=%d %.0fus\n",
                         lr, rdma_state.total_writes, rdma_state.num_expected_recvs, us_since());

    // Step 7: Upload recv metadata to GPU (overlapped with RDMA round-trip).
    // This computes combine_remote_recv_offset, per-token metadata arrays, and
    // uploads everything via GDRCopy MMIO. The dispatch_recv kernel on GPU needs
    // this metadata AFTER dispatch_recv_flag fires (which is after RDMA completes),
    // so this overlap is safe.
    upload_recv_metadata(route_recv_buf_, num_experts, experts_per_rank,
                         nr, lr, cc.max_tokens_per_rank);
    cc.num_recv_tokens_flag->set(1);
    if (lr == 0) fprintf(stderr, "[COOP r%d] step7 metadata_uploaded %.0fus\n", lr, us_since());

    // Step 8: Wait for dispatch RDMA completions.
    complete_coop_dispatch_rdma(rdma_state);

    // Disarm GdrCounter (all RX entries processed, flag already set)
    if (cc.dispatch_rdma_counter) {
        active_gdr_counter_ = nullptr;
    } else {
        // Fallback: manual flag set
        cc.dispatch_recv_flag->set(1);
    }
    if (lr == 0) fprintf(stderr, "[COOP r%d] step8 dispatch_rdma_done+flag %.0fus\n", lr, us_since());

    // Step 9: Wait dispatch_recv_done
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int last_warn_s = 0;
        while (cc.dispatch_recv_done->read() == 0) {
            _mm_pause();
            auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
            int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline COOP: dispatch_recv_done waiting rank=%d %ds %.0fus\n", lr, elapsed_s, us_since());
                last_warn_s = elapsed_s;
            }
        }
        cc.dispatch_recv_done->clear();
    }
    fprintf(stderr, "[COOP r%d] step9 recv_done %.0fus\n", lr, us_since());

    // Step 10: Signal tx_ready
    cc.tx_ready->set(1);
}

// ============================================================================
// Cooperative Combine Handler (Iter 50)
//
// Flow:
//   1. Wait combine_send_done (GPU packed combine data in send buffer)
//   2. Issue combine RDMA writes
//   3. Wait completions → signal combine_recv_flag
//   4. Wait combine_recv_done → barrier → set tx_ready
// ============================================================================
void LLPipelineWorker::handle_coop_combine() {
    auto& cc = coop_config_;
    int nr = config_.num_ranks;
    int lr = config_.local_rank;

    fprintf(stderr, "[COOP r%d] entering combine, comb_tag=%u comb_base=%d corrupted=%d\n",
            lr, (unsigned)preallocated_combine_data_tag_, preallocated_combine_data_base_,
            (int)coop_data_corrupted_);

    // Iter 62: If route data was corrupt, all counts are zero (symmetric skip).

     // Use pre-allocated combine tag with absolute-target counting (base always 0).
     // Some entries for this tag may have been pre-polled during dispatch data wait,
     // but get_total will still reach num_expected when all entries arrive.
     uint16_t combine_data_tag = preallocated_combine_data_tag_;
     int combine_data_base = 0;  // Absolute counting: base always 0

    // Step 1: Wait for combine_send_done
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int last_warn_s = 0;
        while (cc.combine_send_done->read() == 0) {
            _mm_pause();
            auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
            int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline COOP: combine_send_done timeout rank=%d %ds\n", lr, elapsed_s);
                last_warn_s = elapsed_s;
            }
        }
        cc.combine_send_done->clear();
    }

    // Step 2: Arm GdrCounter for auto-signaling combine_recv_flag, then issue combine RDMA.
    // In combine: what we received in dispatch, we now send back.
    // combine_send_counts = coop_recv_counts_ (from dispatch)
    // combine_recv_counts = coop_send_counts_ (from dispatch)
    if (cc.combine_rdma_counter) {
        cc.combine_rdma_counter->reset();
        active_gdr_counter_ = cc.combine_rdma_counter;
        active_counter_tag_ = combine_data_tag;

        // Iter 57: Replay pre-consumed CQ entries. During the dispatch phase,
        // CQ entries for combine_data_tag may have been polled (indiscriminate CQ)
        // and recorded in counters_ but NOT counted by GdrCounter (which was armed
        // for dispatch_data_tag at that time). Replay those missed inc() calls now.
        int pre_consumed = counters_->get_total(combine_data_tag);
        for (int k = 0; k < pre_consumed; ++k) {
            cc.combine_rdma_counter->inc();
        }
    }

    issue_coop_combine_rdma(coop_recv_counts_, coop_send_counts_,
                            cc.combine_token_dim, combine_data_tag, combine_data_base);

    // Disarm GdrCounter
    if (cc.combine_rdma_counter) {
        active_gdr_counter_ = nullptr;
    } else {
        // Fallback: manual flag set
        cc.combine_recv_flag->set(1);
    }

    // Step 3: Wait combine_recv_done
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int last_warn_s = 0;
        while (cc.combine_recv_done->read() == 0) {
            _mm_pause();
            auto elapsed = std::chrono::high_resolution_clock::now() - t_start;
            int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline COOP: combine_recv_done timeout rank=%d %ds\n", lr, elapsed_s);
                last_warn_s = elapsed_s;
            }
        }
        cc.combine_recv_done->clear();
    }

    // Step 6: Pre-allocate NEXT iteration's tags.
    // Iter 61: No reset here — baselines will be snapshotted at the start of next dispatch.
    preallocated_route_scatter_tag_ = current_tag_++;
    preallocated_dispatch_data_tag_ = current_tag_++;
    preallocated_combine_data_tag_ = current_tag_++;
    tags_preallocated_ = true;

    fprintf(stderr, "[COOP r%d] combine done, prealloc route=%u disp=%u comb=%u\n",
            lr, (unsigned)preallocated_route_scatter_tag_,
            (unsigned)preallocated_dispatch_data_tag_,
            (unsigned)preallocated_combine_data_tag_);

    // Step 7: Signal tx_ready for next iteration
    cc.tx_ready->set(1);
}

// ============================================================================
// Reset Tags (Iter 61)
//
// Resets current_tag_ back to 128 and clears ALL tag counter slots.
// Must be called when NO coop operations are in-flight (after dist.barrier).
// This prevents tag wrapping past kMaxTags (1024), which causes the kineto-phase hang.
// ============================================================================
void LLPipelineWorker::reset_tags() {
    current_tag_ = 128;
    tags_preallocated_ = false;
    counters_->reset_all();
    fprintf(stderr, "[COOP r%d] tags reset to 128\n", config_.local_rank);
}

// ============================================================================
// RDMA Scatter Routing Info — Async (Iter 56)
//
// Fire-and-forget version: posts all route scatter RDMA writes but does NOT
// wait for TX completions. TX completions will be drained by subsequent
// CQ polling in the overlapped wait loop.
//
// Route scatter messages are tiny (~1152 bytes for 288 experts), so TX
// queue depth is not a concern for EP16 (15 writes).
//
// RDMA buffer layout for route exchange:
//   Send half [0, kRouteRegionSize): our num_routed data (copied H2D)
//   Recv half [half_rdma, half_rdma + kRouteRegionSize): recv region
//     Within recv region: rank i's data at offset [i * num_experts * 4, (i+1) * num_experts * 4)
// ============================================================================
void LLPipelineWorker::rdma_scatter_route_info_async(const uint32_t* num_routed, int num_experts, uint16_t tag) {
    int nr = config_.num_ranks;
    int lr = config_.local_rank;
     size_t route_bytes = num_experts * sizeof(uint32_t);

    // Iter 62: Place route send data at the TAIL of the send half to avoid overlap
    // with the cooperative dispatch_send kernel's pack region which starts at rdma_base+0.
    // The kernel writes packed token data to [rdma_base, rdma_base + <pack_size>).
    // Previously, route data was also at rdma_base+0, causing a race: the kernel could
    // overwrite route data before the EFA NIC DMA-reads it for the RDMA write.
    uint8_t* route_send = config_.rdma_base + config_.half_rdma - kRouteRegionSize;
    cudaMemcpy(route_send, num_routed, route_bytes, cudaMemcpyHostToDevice);

    // Post receives
    post_recvs_for_transfer(nr);

    // Send our route info to all peers with TX flow control
    // Use NIC 0 only (multi-NIC sharding reverted due to hang in Iter 53)
    uint64_t half_rdma = config_.half_rdma;
    int total_writes = 0;
    int tx_completed = 0;
    static constexpr int kMaxOutstandingRoute = 32;

    for (int i = 0; i < nr; ++i) {
        if (i == lr) continue;

        auto& ep = transport_->endpoint(0);
        uint32_t imm = ImmCounterMap::encode(tag, lr, 1);
        uint64_t dst_offset = half_rdma + lr * route_bytes;

        // Proactive flow control: drain TX CQ if too many outstanding
        while (total_writes - tx_completed >= kMaxOutstandingRoute) {
            struct fi_cq_data_entry drain[64];
            int polled = poll_tx_cq(ep, drain, 64);
            if (polled > 0) tx_completed += polled;
            // Also poll RX to record any incoming completions
            struct fi_cq_data_entry rx_drain[64];
            int rx_polled = poll_rx_cq(ep, rx_drain, 64);
            if (rx_polled > 0) {
                posted_per_nic_[0] -= rx_polled;
                for (int j = 0; j < rx_polled; ++j) {
                    if (rx_drain[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(rx_drain[j].data));
                    }
                }
            }
        }

        int ret = rdma_write_with_imm(
            ep, ep.remote_addrs[i], route_send, route_bytes,
            dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
            imm, nullptr);

        while (ret == -FI_EAGAIN) {
            struct fi_cq_data_entry tx_entries[64];
            int polled = poll_tx_cq(ep, tx_entries, 64);
            if (polled > 0) tx_completed += polled;
            ret = rdma_write_with_imm(
                ep, ep.remote_addrs[i], route_send, route_bytes,
                dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                imm, nullptr);
        }
        if (ret != 0) {
            fprintf(stderr, "[COOP-ROUTE r%d] route scatter write FAILED: peer=%d ret=%d\n",
                    lr, i, ret);
        }
        total_writes++;
    }

    // NOTE: Must wait for TX completions before returning. Even though route send
    // data is now at tail of send half (not overlapping with pack region), we still
    // need TX completions so the EFA NIC has finished reading before any future use.
    {
        int tx_remaining = total_writes - tx_completed;
        int num_nics = transport_->num_nics();
        auto start = std::chrono::high_resolution_clock::now();
        int last_warn_s = 0;
        while (tx_remaining > 0) {
            for (int n = 0; n < num_nics; ++n) {
                struct fi_cq_data_entry entries[64];
                int polled = poll_tx_cq(transport_->endpoint(n), entries, 64);
                if (polled > 0) tx_remaining -= polled;
            }
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
                fprintf(stderr, "LLPipeline COOP: route scatter TX timeout rank=%d tx_rem=%d %ds\n",
                        lr, tx_remaining, elapsed_s);
                last_warn_s = elapsed_s;
            }
        }
    }
    route_scatter_outstanding_tx_ = 0;
}

// ============================================================================
// Wait for all peers' routing info via RDMA (Iter 50)
// ============================================================================
void LLPipelineWorker::rdma_wait_route_info(int num_experts, uint16_t tag, int base_rx) {
    int nr = config_.num_ranks;
    int lr = config_.local_rank;
    int num_expected = nr - 1;  // Expect imm from each peer
    int target_rx = base_rx + num_expected;

    // Tag was passed in from caller (pre-allocated in handle_coop_dispatch)
    // No reset — using snapshot-based counting

    // Poll until we've received all route info
    int num_nics = transport_->num_nics();
    auto start = std::chrono::high_resolution_clock::now();
    struct fi_cq_data_entry entries[64];
    int last_warn_s = 0;

    while (counters_->get_total(tag) < target_rx) {
        for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
            int polled = poll_rx_cq(transport_->endpoint(n), entries, 64);
            if (polled > 0) {
                posted_per_nic_[n] -= polled;
                for (int j = 0; j < polled; ++j) {
                    if (entries[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(entries[j].data));
                    }
                }
            }
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        int elapsed_s = (int)std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
        if (elapsed_s > 0 && elapsed_s % 30 == 0 && elapsed_s != last_warn_s) {
            fprintf(stderr, "LLPipeline COOP: route info wait timeout rank=%d got=%d/%d tag=%u base=%d %ds per_rank=[",
                    lr, counters_->get_total(tag) - base_rx, num_expected, static_cast<unsigned>(tag), base_rx, elapsed_s);
            for (int r = 0; r < nr; ++r) {
                fprintf(stderr, "%d%s", counters_->get_count(tag, r), r < nr-1 ? "," : "");
            }
            fprintf(stderr, "]\n");
            last_warn_s = elapsed_s;
        }
    }

    // Iter 54: Read received route info.
    // For EP16: single bulk D2H copy (all peers + self slot overwritten after).
    // For EP32+: per-peer serial copies (bulk copy shows corruption at higher EP counts,
    // likely due to GPU memory ordering with many concurrent RDMA writers).
    size_t route_bytes = num_experts * sizeof(uint32_t);
    uint8_t* route_recv_base = config_.rdma_base + config_.half_rdma;

    if (nr <= 16) {
        // Bulk D2H: single cudaMemcpy of all route data (~10us vs 15*10us = 150us)
        size_t total_route_bytes = nr * route_bytes;
        cudaMemcpy(route_recv_buf_, route_recv_base, total_route_bytes, cudaMemcpyDeviceToHost);
    } else {
        // Per-peer D2H: safer for large EP counts
        for (int i = 0; i < nr; ++i) {
            if (i == lr) continue;
            cudaMemcpy(&route_recv_buf_[i * num_experts],
                       route_recv_base + i * route_bytes,
                       route_bytes, cudaMemcpyDeviceToHost);
        }
    }
}

// ============================================================================
// Compute Send/Recv Counts (Iter 58, Iter 60: per-rank validation with clamping)
//
// Fast extraction of per-rank send_counts and recv_counts from routing info.
// Extracted from process_routing_info for early computation — allows submitting
// dispatch RDMA writes before the full metadata is computed and uploaded.
//
// Iter 60: Validates raw route data and per-rank counts. Corrupted values are
// clamped to 0 individually (not the entire iteration), so the system remains
// synchronized and doesn't deadlock. Clamped iterations produce wrong results
// but don't hang or crash.
// ============================================================================
void LLPipelineWorker::compute_send_recv_counts(
    const uint32_t* all_num_routed, int num_experts, int experts_per_rank,
    int num_ranks, int local_rank) {

    memset(coop_send_counts_, 0, sizeof(int32_t) * kMaxLLRanks);
    memset(coop_recv_counts_, 0, sizeof(int32_t) * kMaxLLRanks);
    coop_total_send_ = 0;
    coop_total_recv_ = 0;
    coop_data_corrupted_ = false;

    // Per-expert max: a single num_routed value cannot exceed num_tokens * num_topk / 1.
    // Use max_tokens_per_rank as a generous per-expert upper bound.
    int max_per_expert = coop_config_.max_tokens_per_rank;
    if (max_per_expert <= 0) max_per_expert = 4096;

    // Iter 62: Scan for corrupt values first. If ANY value is corrupt, zero ALL counts.
    // This ensures symmetric behavior: all ranks agree that nobody sends or receives.
    // Previously, per-value clamping created asymmetry: corrupted rank sent 0, but
    // clean receiver expected data → deadlock.
    bool any_corrupt = false;
    for (int r = 0; r < num_ranks; ++r) {
        const uint32_t* row = &all_num_routed[r * num_experts];
        for (int e = 0; e < num_experts; ++e) {
            if (row[e] > static_cast<uint32_t>(max_per_expert)) {
                if (!any_corrupt) {
                    fprintf(stderr, "[COOP r%d] WARNING: corrupt route data detected: "
                            "all_num_routed[%d][%d]=%u (max=%d), hex=0x%08x\n",
                            local_rank, r, e, row[e], max_per_expert, row[e]);
                }
                any_corrupt = true;
            }
        }
    }

    if (any_corrupt) {
        // Zero ALL counts — skip this iteration entirely. All ranks should see the same
        // corruption (route data from the same RDMA writes), so they should all zero out.
        coop_data_corrupted_ = true;
        fprintf(stderr, "[COOP r%d] WARNING: route data corrupt — zeroing ALL send/recv counts "
                "for this iteration\n", local_rank);
        return;  // counts are already zeroed from memset above
    }

    // Our send counts: sum of our row's experts per destination rank
    const uint32_t* our_row = &all_num_routed[local_rank * num_experts];
    for (int e = 0; e < num_experts; ++e) {
        int dst_rank = e / experts_per_rank;
        if (dst_rank < num_ranks) {
            coop_send_counts_[dst_rank] += static_cast<int32_t>(our_row[e]);
        }
    }
    for (int r = 0; r < num_ranks; ++r) {
        coop_total_send_ += coop_send_counts_[r];
    }

    // Recv counts: for each peer rank r, sum their num_routed for our local experts
    int first_expert = local_rank * experts_per_rank;
    int last_expert = std::min(first_expert + experts_per_rank, num_experts);
    for (int r = 0; r < num_ranks; ++r) {
        const uint32_t* their_row = &all_num_routed[r * num_experts];
        for (int e = first_expert; e < last_expert; ++e) {
            coop_recv_counts_[r] += static_cast<int32_t>(their_row[e]);
        }
        coop_total_recv_ += coop_recv_counts_[r];
    }
}

// ============================================================================
// Upload Recv Metadata to GPU (Iter 58)
//
// Computes combine_remote_recv_offset, per-token metadata arrays, and uploads
// everything to GPU via GDRCopy MMIO. Must be called AFTER compute_send_recv_counts.
// ============================================================================
void LLPipelineWorker::upload_recv_metadata(
    const uint32_t* all_num_routed, int num_experts, int experts_per_rank,
    int num_ranks, int local_rank, int max_tokens_per_rank) {

    // Step 1b: Compute combine_remote_recv_offset_ for each peer
    for (int j = 0; j < num_ranks; ++j) {
        const uint32_t* peer_row = &all_num_routed[j * num_experts];
        int64_t offset = 0;
        for (int r = 0; r < local_rank; ++r) {
            int r_first_expert = r * experts_per_rank;
            int r_last_expert = std::min(r_first_expert + experts_per_rank, num_experts);
            for (int e = r_first_expert; e < r_last_expert; ++e) {
                offset += static_cast<int64_t>(peer_row[e]);
            }
        }
        combine_remote_recv_offset_[j] = offset;
    }

    // Step 2: Build per-token metadata arrays
    std::vector<uint32_t> source_rank_vec;
    std::vector<uint32_t> source_offset_vec;
    std::vector<uint32_t> padded_index_vec;
    std::vector<uint32_t> combine_send_offset_vec;
    uint32_t tokens_per_expert_arr[kMaxExperts] = {};

    // Safety check
    int max_possible = num_ranks * max_tokens_per_rank * 8;
    if (coop_total_recv_ < 0 || coop_total_recv_ > max_possible) {
        fprintf(stderr, "[COOP r%d] ERROR: coop_total_recv_=%d out of range (max=%d), clamping to 0\n",
                local_rank, coop_total_recv_, max_possible);
        coop_total_recv_ = 0;
    }

    source_rank_vec.reserve(coop_total_recv_);
    source_offset_vec.reserve(coop_total_recv_);
    padded_index_vec.reserve(coop_total_recv_);
    combine_send_offset_vec.reserve(coop_total_recv_);

    // src_group_offset[r] = cumulative recv tokens from ranks < r
    int src_group_offset[kMaxLLRanks];
    int off = 0;
    for (int r = 0; r < num_ranks; ++r) {
        src_group_offset[r] = off;
        off += coop_recv_counts_[r];
    }

    int rank_counter[kMaxLLRanks] = {};
    int token_idx = 0;
    for (int r = 0; r < num_ranks; ++r) {
        int cnt = coop_recv_counts_[r];
        if (cnt <= 0) continue;
        int base_offset = r * max_tokens_per_rank;
        for (int t = 0; t < cnt; ++t) {
            source_rank_vec.push_back(static_cast<uint32_t>(r));
            source_offset_vec.push_back(static_cast<uint32_t>(base_offset + t));
            padded_index_vec.push_back(static_cast<uint32_t>(token_idx));
            combine_send_offset_vec.push_back(
                static_cast<uint32_t>(src_group_offset[r] + rank_counter[r]));
            rank_counter[r]++;
            token_idx++;
        }
    }

    int num_efa_tokens = static_cast<int>(source_rank_vec.size());
    memset(tokens_per_expert_arr, 0, experts_per_rank * sizeof(uint32_t));

    // Step 3: Upload metadata to GPU via GDRCopy MMIO
    auto& cc = coop_config_;
    if (!source_rank_vec.empty()) {
        size_t n = source_rank_vec.size();
        cc.gdr_source_rank->copy(source_rank_vec.data(), n);
        cc.gdr_source_offset->copy(source_offset_vec.data(), n);
        cc.gdr_padded_index->copy(padded_index_vec.data(), n);
        cc.gdr_combine_send_offset->copy(combine_send_offset_vec.data(), n);
    }

    cc.gdr_tokens_per_expert->copy(tokens_per_expert_arr, experts_per_rank);

    // Write num_recv_tokens: [total, efa_only]
    cc.gdr_num_recv_tokens->write(0, static_cast<uint32_t>(coop_total_recv_));
    cc.gdr_num_recv_tokens->write(1, static_cast<uint32_t>(num_efa_tokens));

    // Flush WC buffer
    _mm_sfence();
}

// ============================================================================
// Process Routing Info (Iter 50)
//
// C++ port of the Python metadata computation in buffer.py.
// For NODE_SIZE=1, DP_SIZE=1: all tokens are EFA tokens.
//
// Computes and uploads to GPU:
//   - source_rank[i], source_offset[i], padded_index[i]: per-token metadata
//   - combine_send_offset[i]: where to write in combine send buffer
//   - tokens_per_expert[e]: per-expert token counts
//   - num_recv_tokens = [total, efa_only]
//   - coop_recv_counts_[r]: per-rank recv counts (stashed for combine)
//   - coop_send_counts_[r]: per-rank send counts (stashed for combine)
// ============================================================================
void LLPipelineWorker::process_routing_info(
    const uint32_t* all_num_routed, int num_experts, int experts_per_rank,
    int num_ranks, int local_rank, int max_tokens_per_rank) {

    // Step 1: Compute per-rank send_counts and recv_counts
    memset(coop_send_counts_, 0, sizeof(int32_t) * kMaxLLRanks);
    memset(coop_recv_counts_, 0, sizeof(int32_t) * kMaxLLRanks);
    coop_total_send_ = 0;
    coop_total_recv_ = 0;

    // Our send counts: sum of our row's experts per destination rank
    // our_num_routed = all_num_routed[local_rank * num_experts ..]
    const uint32_t* our_row = &all_num_routed[local_rank * num_experts];
    for (int e = 0; e < num_experts; ++e) {
        int dst_rank = e / experts_per_rank;
        if (dst_rank < num_ranks) {
            coop_send_counts_[dst_rank] += static_cast<int32_t>(our_row[e]);
        }
    }
    for (int r = 0; r < num_ranks; ++r) {
        coop_total_send_ += coop_send_counts_[r];
    }

    // Recv counts: for each peer rank r, sum their num_routed for our local experts
    int first_expert = local_rank * experts_per_rank;
    int last_expert = std::min(first_expert + experts_per_rank, num_experts);
    for (int r = 0; r < num_ranks; ++r) {
        const uint32_t* their_row = &all_num_routed[r * num_experts];
        for (int e = first_expert; e < last_expert; ++e) {
            coop_recv_counts_[r] += static_cast<int32_t>(their_row[e]);
        }
        coop_total_recv_ += coop_recv_counts_[r];
    }

    // Step 1b: Compute combine_remote_recv_offset_ for each peer
    // For combine, we send data back to peer j. On peer j, the combine recv buffer
    // is indexed by position = expert_offsets[expert-1] + token_offset.
    // For tokens from our rank, they occupy a contiguous range starting at
    // dst_group_offset[local_rank] on peer j.
    // dst_group_offset[r] on peer j = sum of peer_j_send_counts[0..r-1]
    // where peer_j_send_counts[r] = sum of num_routed[j][e] for experts belonging to rank r.
    for (int j = 0; j < num_ranks; ++j) {
        const uint32_t* peer_row = &all_num_routed[j * num_experts];
        int64_t offset = 0;
        for (int r = 0; r < local_rank; ++r) {
            // Count tokens peer j sends to rank r
            int r_first_expert = r * experts_per_rank;
            int r_last_expert = std::min(r_first_expert + experts_per_rank, num_experts);
            for (int e = r_first_expert; e < r_last_expert; ++e) {
                offset += static_cast<int64_t>(peer_row[e]);
            }
        }
        combine_remote_recv_offset_[j] = offset;
    }

    // Step 2: Build per-token metadata arrays
    // For NODE_SIZE=1: ALL tokens are EFA tokens (including self-rank, via D2D in issue_coop_dispatch_rdma)
    std::vector<uint32_t> source_rank_vec;
    std::vector<uint32_t> source_offset_vec;
    std::vector<uint32_t> padded_index_vec;
    std::vector<uint32_t> combine_send_offset_vec;
    uint32_t tokens_per_expert_arr[kMaxExperts] = {};

    // Safety check: coop_total_recv_ must be reasonable
    int max_possible = num_ranks * max_tokens_per_rank * 8;  // topk=8 max
    if (coop_total_recv_ < 0 || coop_total_recv_ > max_possible) {
        fprintf(stderr, "[COOP r%d] ERROR: coop_total_recv_=%d out of range (max=%d), clamping to 0\n",
                config_.local_rank, coop_total_recv_, max_possible);
        coop_total_recv_ = 0;
    }

    source_rank_vec.reserve(coop_total_recv_);
    source_offset_vec.reserve(coop_total_recv_);
    padded_index_vec.reserve(coop_total_recv_);
    combine_send_offset_vec.reserve(coop_total_recv_);

    // src_group_offset[r] = cumulative recv tokens from ranks < r
    int src_group_offset[kMaxLLRanks];
    int off = 0;
    for (int r = 0; r < num_ranks; ++r) {
        src_group_offset[r] = off;
        off += coop_recv_counts_[r];
    }

    // Per-rank counter for combine_send_offset computation
    int rank_counter[kMaxLLRanks] = {};

    // Iterate over ranks and their tokens for our local experts
    int token_idx = 0;
    for (int r = 0; r < num_ranks; ++r) {
        int cnt = coop_recv_counts_[r];
        if (cnt <= 0) continue;

        int base_offset = r * max_tokens_per_rank;
        for (int t = 0; t < cnt; ++t) {
            source_rank_vec.push_back(static_cast<uint32_t>(r));
            source_offset_vec.push_back(static_cast<uint32_t>(base_offset + t));
            padded_index_vec.push_back(static_cast<uint32_t>(token_idx));
            combine_send_offset_vec.push_back(
                static_cast<uint32_t>(src_group_offset[r] + rank_counter[r]));
            rank_counter[r]++;
            token_idx++;
        }
    }

    int num_efa_tokens = static_cast<int>(source_rank_vec.size());

    // tokens_per_expert: zeros for now (we don't have per-expert breakdown)
    // This is OK because dispatch_recv kernel uses padded_index for placement.
    memset(tokens_per_expert_arr, 0, experts_per_rank * sizeof(uint32_t));

    // Step 3: Upload metadata to GPU via GDRCopy MMIO (Iter 53)
    // Replaces 6 synchronous cudaMemcpy H2D calls (~5-15us each = 30-90us total)
    // with direct MMIO writes (~1-2us total for all metadata).
    auto& cc = coop_config_;
    if (!source_rank_vec.empty()) {
        size_t n = source_rank_vec.size();
        cc.gdr_source_rank->copy(source_rank_vec.data(), n);
        cc.gdr_source_offset->copy(source_offset_vec.data(), n);
        cc.gdr_padded_index->copy(padded_index_vec.data(), n);
        cc.gdr_combine_send_offset->copy(combine_send_offset_vec.data(), n);
    }

    cc.gdr_tokens_per_expert->copy(tokens_per_expert_arr, experts_per_rank);

    // Write num_recv_tokens: [total, efa_only]
    cc.gdr_num_recv_tokens->write(0, static_cast<uint32_t>(coop_total_recv_));
    cc.gdr_num_recv_tokens->write(1, static_cast<uint32_t>(num_efa_tokens));

    // Flush WC buffer: ensure all metadata writes are visible to GPU before
    // the flag is set. gdr_copy_to_mapping() (used by GdrVec::copy above)
    // includes an internal sfence, but the individual write() calls above
    // use volatile stores that may still be in the WC buffer.
    _mm_sfence();
}

// ============================================================================
// Submit Coop Dispatch RDMA (Iter 58)
//
// Non-blocking variant: submits RDMA writes and D2D self-copy, arms GdrCounter,
// but does NOT wait for completions. Returns state for deferred completion.
// This allows overlapping metadata processing with RDMA round-trip.
// ============================================================================
LLPipelineWorker::DispatchRdmaState LLPipelineWorker::submit_coop_dispatch_rdma(
    const int32_t* send_counts, const int32_t* recv_counts,
    int coop_token_stride, int max_tokens_per_rank,
    uint16_t tag, int base_rx) {

    int nr = config_.num_ranks;
    int lr = config_.local_rank;
    int64_t slot_size = static_cast<int64_t>(max_tokens_per_rank) * coop_token_stride;
    uint64_t half_rdma = config_.half_rdma;
    uint8_t* rdma_base = config_.rdma_base;

    // Compute send offsets (contiguous)
    int64_t send_offsets[kMaxLLRanks];
    int64_t send_sizes[kMaxLLRanks];
    int64_t s_off = 0;
    for (int i = 0; i < nr; ++i) {
        send_offsets[i] = s_off;
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * coop_token_stride;
        s_off += send_sizes[i];
    }

    // Self-copy (D2D) — offset by kRouteRegionSize to avoid route data overlap
    if (send_sizes[lr] > 0) {
        uint8_t* src = rdma_base + send_offsets[lr];
        uint8_t* dst = rdma_base + half_rdma + kRouteRegionSize + lr * slot_size;
        cudaMemcpy(dst, src, static_cast<size_t>(send_sizes[lr]), cudaMemcpyDeviceToDevice);
    }

    int num_nics = transport_->num_nics();

    // Post receives
    post_recvs_for_transfer(nr);

    // Process any pending RX CQ entries
    for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
        auto& ep = transport_->endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = poll_rx_cq(ep, drain_entries, 64);
            if (drained > 0) {
                posted_per_nic_[n] -= drained;
                for (int j = 0; j < drained; ++j) {
                    if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(drain_entries[j].data));
                    }
                }
            }
        } while (drained > 0);
    }

    // Count expected incoming writes
    int num_expected_recvs = 0;
    for (int i = 0; i < nr; ++i) {
        if (i == lr) continue;
        int64_t recv_size = static_cast<int64_t>(recv_counts[i]) * coop_token_stride;
        if (recv_size <= 0) continue;
        size_t total_bytes = static_cast<size_t>(recv_size);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);
        if (bytes_per_shard == 0) bytes_per_shard = kShardAlign;
        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;
            num_expected_recvs++;
            shard_offset += shard_len;
        }
    }

    // Arm GdrCounter with the expected RX count
    if (active_gdr_counter_ && num_expected_recvs > 0) {
        active_gdr_counter_->wait(num_expected_recvs);
    } else if (active_gdr_counter_ && num_expected_recvs == 0) {
        active_gdr_counter_->flag()->set(1);
    }

    // Issue RDMA writes, sharding across NICs
    int total_writes = 0;
    int writes_per_nic[4] = {};
    int tx_completed[4] = {};
    int total_tx_completed = 0;

    for (int i = 0; i < nr; ++i) {
        if (send_sizes[i] <= 0 || i == lr) continue;

        size_t total_bytes = static_cast<size_t>(send_sizes[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);

        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;

            auto& ep = transport_->endpoint(n);

            // Flow control
            while (writes_per_nic[n] - tx_completed[n] >= kMaxOutstandingPerNic) {
                struct fi_cq_data_entry drain[64];
                int polled = poll_tx_cq(ep, drain, 64);
                if (polled > 0) { tx_completed[n] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_drain[64];
                int rx_polled = poll_rx_cq(ep, rx_drain, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_drain[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_drain[j].data));
                        }
                    }
                }
            }

            void* src = rdma_base + send_offsets[i] + shard_offset;
            uint64_t dst_offset = half_rdma + kRouteRegionSize + lr * slot_size + shard_offset;
            uint32_t imm = ImmCounterMap::encode(tag, lr, 1);

            int ret = rdma_write_with_imm(
                ep, ep.remote_addrs[i], src, shard_len,
                dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                imm, nullptr);

            struct fi_cq_data_entry tx_entries[64];
            while (ret == -FI_EAGAIN) {
                int polled = poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) { tx_completed[n] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_entries[64];
                int rx_polled = poll_rx_cq(ep, rx_entries, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_entries[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_entries[j].data));
                        }
                    }
                }
                ret = rdma_write_with_imm(
                    ep, ep.remote_addrs[i], src, shard_len,
                    dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                    imm, nullptr);
            }
            if (ret != 0) {
                fprintf(stderr, "[COOP-RDMA r%d] dispatch submit err: peer=%d nic=%d ret=%d len=%zu\n",
                        lr, i, n, ret, shard_len);
            }

            writes_per_nic[n]++;
            total_writes++;
            shard_offset += shard_len;
        }
    }

    // Return state for deferred completion (DO NOT wait here)
    return {total_writes, total_tx_completed, num_expected_recvs, tag, base_rx};
}

// ============================================================================
// Complete Coop Dispatch RDMA (Iter 58)
//
// Waits for all TX+RX completions for previously submitted dispatch RDMA.
// Called after overlapped metadata processing.
// ============================================================================
void LLPipelineWorker::complete_coop_dispatch_rdma(const DispatchRdmaState& state) {
    int num_nics = transport_->num_nics();

    // Wait for all completions
    wait_all_completions(state.total_writes - state.total_tx_completed,
                         state.num_expected_recvs, state.tag, state.base_rx);

    // Drain remaining TX CQ
    for (int n = 0; n < num_nics; ++n) {
        struct fi_cq_data_entry drain[64];
        int drained;
        do { drained = poll_tx_cq(transport_->endpoint(n), drain, 64); } while (drained > 0);
    }
}

// ============================================================================
// Issue Coop Dispatch RDMA (Iter 50)
//
// Slot-based recv layout. Send buffer starts at rdma_base (data region after route region).
// For route exchange, we used offset 0..kRouteRegionSize. For data RDMA, the dispatch
// send kernel writes to rdma_base + 0 (it writes into the send half starting at offset 0).
// The route region is small and the dispatch kernel writes tokens starting from offset 0
// contiguously. Since the route region is only used briefly during route exchange (which
// is done by now), it's safe to reuse offset 0 for token data.
// ============================================================================
void LLPipelineWorker::issue_coop_dispatch_rdma(
    const int32_t* send_counts, const int32_t* recv_counts,
    int coop_token_stride, int max_tokens_per_rank, uint16_t tag, int base_rx) {

    int nr = config_.num_ranks;
    int lr = config_.local_rank;
    int64_t slot_size = static_cast<int64_t>(max_tokens_per_rank) * coop_token_stride;
    uint64_t half_rdma = config_.half_rdma;
    uint8_t* rdma_base = config_.rdma_base;

    // Compute send offsets (contiguous)
    int64_t send_offsets[kMaxLLRanks];
    int64_t send_sizes[kMaxLLRanks];
    int64_t s_off = 0;
    for (int i = 0; i < nr; ++i) {
        send_offsets[i] = s_off;
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * coop_token_stride;
        s_off += send_sizes[i];
    }

    // Self-copy (D2D) — offset by kRouteRegionSize to avoid route data overlap
    if (send_sizes[lr] > 0) {
        uint8_t* src = rdma_base + send_offsets[lr];
        uint8_t* dst = rdma_base + half_rdma + kRouteRegionSize + lr * slot_size;
        cudaMemcpy(dst, src, static_cast<size_t>(send_sizes[lr]), cudaMemcpyDeviceToDevice);
    }

    // Tag was pre-allocated by caller; using snapshot-based counting (no reset)
    int num_nics = transport_->num_nics();

    // Post receives
    post_recvs_for_transfer(nr);

    // Process any pending RX CQ entries (records completions for their respective tags)
    for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
        auto& ep = transport_->endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = poll_rx_cq(ep, drain_entries, 64);
            if (drained > 0) {
                posted_per_nic_[n] -= drained;
                for (int j = 0; j < drained; ++j) {
                    if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(drain_entries[j].data));
                    }
                }
            }
        } while (drained > 0);
    }

    // Count expected incoming writes
    int num_expected_recvs = 0;
    for (int i = 0; i < nr; ++i) {
        if (i == lr) continue;
        int64_t recv_size = static_cast<int64_t>(recv_counts[i]) * coop_token_stride;
        if (recv_size <= 0) continue;
        size_t total_bytes = static_cast<size_t>(recv_size);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);
        if (bytes_per_shard == 0) bytes_per_shard = kShardAlign;
        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;
            num_expected_recvs++;
            shard_offset += shard_len;
        }
    }

    // Iter 57: Arm GdrCounter with the expected RX count BEFORE submitting RDMA writes.
    // GdrCounter::wait(N) does fetch_sub(N), setting counter to -N (or less if inc() already
    // happened from early CQ entries during drain above). When the Nth inc() brings the
    // counter to 0, the flag is auto-set, notifying the GPU immediately — potentially
    // before we finish draining TX completions in wait_all_completions().
    if (active_gdr_counter_ && num_expected_recvs > 0) {
        active_gdr_counter_->wait(num_expected_recvs);
    } else if (active_gdr_counter_ && num_expected_recvs == 0) {
        // No remote data expected (e.g., all recv_counts are 0) — set flag immediately
        active_gdr_counter_->flag()->set(1);
    }

    // Issue RDMA writes, sharding across NICs
    int total_writes = 0;
    int writes_per_nic[4] = {};
    int tx_completed[4] = {};
    int total_tx_completed = 0;

    for (int i = 0; i < nr; ++i) {
        if (send_sizes[i] <= 0 || i == lr) continue;

        size_t total_bytes = static_cast<size_t>(send_sizes[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);

        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;

            auto& ep = transport_->endpoint(n);

            // Flow control
            while (writes_per_nic[n] - tx_completed[n] >= kMaxOutstandingPerNic) {
                struct fi_cq_data_entry drain[64];
                int polled = poll_tx_cq(ep, drain, 64);
                if (polled > 0) { tx_completed[n] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_drain[64];
                int rx_polled = poll_rx_cq(ep, rx_drain, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_drain[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_drain[j].data));
                        }
                    }
                }
            }

            void* src = rdma_base + send_offsets[i] + shard_offset;
            // Slot-based recv: peer i's data goes into slot lr on the remote side
            uint64_t dst_offset = half_rdma + kRouteRegionSize + lr * slot_size + shard_offset;
            uint32_t imm = ImmCounterMap::encode(tag, lr, 1);

            int ret = rdma_write_with_imm(
                ep, ep.remote_addrs[i], src, shard_len,
                dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                imm, nullptr);

            struct fi_cq_data_entry tx_entries[64];
            while (ret == -FI_EAGAIN) {
                int polled = poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) { tx_completed[n] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_entries[64];
                int rx_polled = poll_rx_cq(ep, rx_entries, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_entries[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_entries[j].data));
                        }
                    }
                }
                ret = rdma_write_with_imm(
                    ep, ep.remote_addrs[i], src, shard_len,
                    dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                    imm, nullptr);
            }
            if (ret != 0) {
                fprintf(stderr, "[COOP-RDMA r%d] dispatch write err: peer=%d nic=%d ret=%d len=%zu\n",
                        lr, i, n, ret, shard_len);
            }

            writes_per_nic[n]++;
            total_writes++;
            shard_offset += shard_len;
        }
    }

    // Wait for all completions
    wait_all_completions(total_writes - total_tx_completed, num_expected_recvs, tag, base_rx);

    // Drain remaining TX CQ
    for (int n = 0; n < num_nics; ++n) {
        struct fi_cq_data_entry drain[64];
        int drained;
        do { drained = poll_tx_cq(transport_->endpoint(n), drain, 64); } while (drained > 0);
    }
}

// ============================================================================
// Issue Coop Combine RDMA (Iter 50)
//
// Prefix-sum recv layout (contiguous, no fixed slots).
// Combine: send what we received in dispatch, receive what we sent.
// ============================================================================
void LLPipelineWorker::issue_coop_combine_rdma(
    const int32_t* send_counts, const int32_t* recv_counts,
    int combine_token_dim, uint16_t tag, int base_rx) {

    int nr = config_.num_ranks;
    int lr = config_.local_rank;
    uint64_t half_rdma = config_.half_rdma;
    uint8_t* rdma_base = config_.rdma_base;

    // Compute send offsets (contiguous in send half)
    int64_t send_offsets[kMaxLLRanks];
    int64_t send_sizes[kMaxLLRanks];
    int64_t s_off = 0;
    for (int i = 0; i < nr; ++i) {
        send_offsets[i] = s_off;
        send_sizes[i] = static_cast<int64_t>(send_counts[i]) * combine_token_dim;
        s_off += send_sizes[i];
    }

    // Compute recv offsets: prefix-sum-based (contiguous per-rank)
    int64_t recv_offsets[kMaxLLRanks];
    int64_t recv_sizes[kMaxLLRanks];
    int64_t r_off = 0;
    for (int i = 0; i < nr; ++i) {
        recv_offsets[i] = r_off;
        recv_sizes[i] = static_cast<int64_t>(recv_counts[i]) * combine_token_dim;
        r_off += recv_sizes[i];
    }

    // Self-copy (D2D)
    if (send_sizes[lr] > 0) {
        uint8_t* src = rdma_base + send_offsets[lr];
        uint8_t* dst = rdma_base + half_rdma + kRouteRegionSize + recv_offsets[lr];
        cudaMemcpy(dst, src, static_cast<size_t>(send_sizes[lr]), cudaMemcpyDeviceToDevice);
    }

    // Tag was pre-allocated and reset by caller to avoid race with early CQ completions
    int num_nics = transport_->num_nics();

    post_recvs_for_transfer(nr);

    // Drain leftover RX CQ entries
    for (int n = 0; n < num_nics && n < kMaxNicsPerGpu; ++n) {
        auto& ep = transport_->endpoint(n);
        struct fi_cq_data_entry drain_entries[64];
        int drained;
        do {
            drained = poll_rx_cq(ep, drain_entries, 64);
            if (drained > 0) {
                posted_per_nic_[n] -= drained;
                for (int j = 0; j < drained; ++j) {
                    if (drain_entries[j].flags & FI_REMOTE_CQ_DATA) {
                        record_cq_entry(static_cast<uint32_t>(drain_entries[j].data));
                    }
                }
            }
        } while (drained > 0);
    }

    // Count expected recv writes
    int num_expected_recvs = 0;
    for (int i = 0; i < nr; ++i) {
        if (i == lr || recv_sizes[i] <= 0) continue;
        size_t total_bytes = static_cast<size_t>(recv_sizes[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);
        if (bytes_per_shard == 0) bytes_per_shard = kShardAlign;
        size_t so = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t sl = std::min(bytes_per_shard, total_bytes - so);
            if (sl == 0) break;
            num_expected_recvs++;
            so += sl;
        }
    }

    // Iter 57: Arm GdrCounter for combine (same pattern as dispatch)
    if (active_gdr_counter_ && num_expected_recvs > 0) {
        active_gdr_counter_->wait(num_expected_recvs);
    } else if (active_gdr_counter_ && num_expected_recvs == 0) {
        active_gdr_counter_->flag()->set(1);
    }

    // Issue RDMA writes
    int total_writes = 0;
    int writes_per_nic[4] = {};
    int tx_completed[4] = {};
    int total_tx_completed = 0;

    for (int i = 0; i < nr; ++i) {
        if (send_sizes[i] <= 0 || i == lr) continue;

        size_t total_bytes = static_cast<size_t>(send_sizes[i]);
        size_t bytes_per_shard = ((total_bytes / num_nics) + kShardAlign - 1) & ~(kShardAlign - 1);

        // Combine RDMA: write our data into peer i's recv half at the correct offset.
        // combine_remote_recv_offset_[i] = the token offset on peer i where our data should land.
        // This was computed from the full routing info in process_routing_info().
        int64_t remote_recv_byte_offset = combine_remote_recv_offset_[i] * combine_token_dim;

        size_t shard_offset = 0;
        for (int n = 0; n < num_nics; ++n) {
            size_t shard_len = std::min(bytes_per_shard, total_bytes - shard_offset);
            if (shard_len == 0) break;

            auto& ep = transport_->endpoint(n);

            while (writes_per_nic[n] - tx_completed[n] >= kMaxOutstandingPerNic) {
                struct fi_cq_data_entry drain[64];
                int polled = poll_tx_cq(ep, drain, 64);
                if (polled > 0) { tx_completed[n] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_drain[64];
                int rx_polled = poll_rx_cq(ep, rx_drain, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_drain[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_drain[j].data));
                        }
                    }
                }
            }

            void* src = rdma_base + send_offsets[i] + shard_offset;
            // Combine: write into remote recv at computed offset
            uint64_t dst_offset = half_rdma + kRouteRegionSize + remote_recv_byte_offset + shard_offset;
            uint32_t imm = ImmCounterMap::encode(tag, lr, 1);

            int ret = rdma_write_with_imm(
                ep, ep.remote_addrs[i], src, shard_len,
                dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                imm, nullptr);

            struct fi_cq_data_entry tx_entries[64];
            while (ret == -FI_EAGAIN) {
                int polled = poll_tx_cq(ep, tx_entries, 64);
                if (polled > 0) { tx_completed[n] += polled; total_tx_completed += polled; }
                struct fi_cq_data_entry rx_entries[64];
                int rx_polled = poll_rx_cq(ep, rx_entries, 64);
                if (rx_polled > 0) {
                    if (n < kMaxNicsPerGpu) posted_per_nic_[n] -= rx_polled;
                    for (int j = 0; j < rx_polled; ++j) {
                        if (rx_entries[j].flags & FI_REMOTE_CQ_DATA) {
                            record_cq_entry(static_cast<uint32_t>(rx_entries[j].data));
                        }
                    }
                }
                ret = rdma_write_with_imm(
                    ep, ep.remote_addrs[i], src, shard_len,
                    dst_offset, ep.remote_keys[i], ep.remote_base_addrs[i],
                    imm, nullptr);
            }

            writes_per_nic[n]++;
            total_writes++;
            shard_offset += shard_len;
        }
    }

    // Wait for all completions
    wait_all_completions(total_writes - total_tx_completed, num_expected_recvs, tag, base_rx);

    // Drain remaining TX CQ
    for (int n = 0; n < num_nics; ++n) {
        struct fi_cq_data_entry drain[64];
        int drained;
        do { drained = poll_tx_cq(transport_->endpoint(n), drain, 64); } while (drained > 0);
    }
}

// ============================================================================
// GDR mapping of RDMA buffer route regions (Iter 53)
//
// Maps the route send region [0, kRouteRegionSize) and route recv region
// [half_rdma, half_rdma + kRouteRegionSize) of the RDMA buffer via GDRCopy.
// This enables zero-copy CPU read/write of routing metadata (~1us MMIO)
// instead of synchronous cudaMemcpy (~5-15us per call).
//
// The RDMA buffer is allocated via CUDA VMM with gpuDirectRDMACapable=1,
// which means it is eligible for GDRCopy pinning.
// ============================================================================
void LLPipelineWorker::init_rdma_gdr_mapping() {
    if (rdma_gdr_mapped_) return;

    static constexpr size_t kGdrPageSize = 65536;  // 64KB, same as gdr_signal.cpp

    rdma_gdr_ = get_gdr_handle();
    if (!rdma_gdr_) {
        fprintf(stderr, "[COOP-GDR r%d] WARNING: GDR handle unavailable, falling back to cudaMemcpy\n",
                config_.local_rank);
        return;
    }

    uint64_t half_rdma = config_.half_rdma;
    CUdeviceptr rdma_base = reinterpret_cast<CUdeviceptr>(config_.rdma_base);

    // GDR requires page-aligned addresses. The RDMA buffer from CUDA VMM
    // is always page-aligned (granularity aligned). We map kRouteRegionSize
    // which is 256KB — well within GDR limits.

    // Map send half route region [0, kRouteRegionSize)
    {
        CUdeviceptr aligned_ptr = rdma_base & ~(static_cast<CUdeviceptr>(kGdrPageSize) - 1);
        size_t map_size = kRouteRegionSize;
        // Round up to page size
        size_t extra = static_cast<size_t>(rdma_base - aligned_ptr);
        map_size += extra;
        map_size = (map_size + kGdrPageSize - 1) & ~(kGdrPageSize - 1);

        int ret = gdr_pin_buffer(rdma_gdr_, aligned_ptr, map_size, 0, 0, &rdma_send_mh_);
        if (ret != 0) {
            fprintf(stderr, "[COOP-GDR r%d] WARNING: gdr_pin_buffer(send) failed: ret=%d, "
                    "falling back to cudaMemcpy\n", config_.local_rank, ret);
            return;
        }
        ret = gdr_map(rdma_gdr_, rdma_send_mh_, &rdma_send_cpu_map_, map_size);
        if (ret != 0) {
            fprintf(stderr, "[COOP-GDR r%d] WARNING: gdr_map(send) failed: ret=%d\n",
                    config_.local_rank, ret);
            gdr_unpin_buffer(rdma_gdr_, rdma_send_mh_);
            return;
        }
        // Compute CPU pointer offset (rdma_base may not be at aligned_ptr)
        size_t offset_in_page = static_cast<size_t>(rdma_base - aligned_ptr);
        rdma_send_cpu_ptr_ = reinterpret_cast<volatile uint8_t*>(
            static_cast<uint8_t*>(rdma_send_cpu_map_) + offset_in_page);
    }

    // Map recv half route region [half_rdma, half_rdma + kRouteRegionSize)
    {
        CUdeviceptr recv_start = rdma_base + half_rdma;
        CUdeviceptr aligned_ptr = recv_start & ~(static_cast<CUdeviceptr>(kGdrPageSize) - 1);
        size_t map_size = kRouteRegionSize;
        size_t extra = static_cast<size_t>(recv_start - aligned_ptr);
        map_size += extra;
        map_size = (map_size + kGdrPageSize - 1) & ~(kGdrPageSize - 1);

        int ret = gdr_pin_buffer(rdma_gdr_, aligned_ptr, map_size, 0, 0, &rdma_recv_mh_);
        if (ret != 0) {
            fprintf(stderr, "[COOP-GDR r%d] WARNING: gdr_pin_buffer(recv) failed: ret=%d, "
                    "falling back to cudaMemcpy\n", config_.local_rank, ret);
            // Cleanup send mapping
            size_t send_map_size = (kRouteRegionSize + kGdrPageSize - 1) & ~(kGdrPageSize - 1);
            gdr_unmap(rdma_gdr_, rdma_send_mh_, rdma_send_cpu_map_, send_map_size);
            gdr_unpin_buffer(rdma_gdr_, rdma_send_mh_);
            rdma_send_cpu_ptr_ = nullptr;
            return;
        }
        ret = gdr_map(rdma_gdr_, rdma_recv_mh_, &rdma_recv_cpu_map_, map_size);
        if (ret != 0) {
            fprintf(stderr, "[COOP-GDR r%d] WARNING: gdr_map(recv) failed: ret=%d\n",
                    config_.local_rank, ret);
            gdr_unpin_buffer(rdma_gdr_, rdma_recv_mh_);
            size_t send_map_size = (kRouteRegionSize + kGdrPageSize - 1) & ~(kGdrPageSize - 1);
            gdr_unmap(rdma_gdr_, rdma_send_mh_, rdma_send_cpu_map_, send_map_size);
            gdr_unpin_buffer(rdma_gdr_, rdma_send_mh_);
            rdma_send_cpu_ptr_ = nullptr;
            return;
        }
        size_t offset_in_page = static_cast<size_t>(recv_start - aligned_ptr);
        rdma_recv_cpu_ptr_ = reinterpret_cast<volatile uint8_t*>(
            static_cast<uint8_t*>(rdma_recv_cpu_map_) + offset_in_page);
    }

    rdma_gdr_mapped_ = true;
    fprintf(stderr, "[COOP-GDR r%d] RDMA route regions GDR-mapped: send=%p recv=%p\n",
            config_.local_rank,
            (void*)rdma_send_cpu_ptr_, (void*)rdma_recv_cpu_ptr_);
}

}  // namespace efa
}  // namespace deep_ep
