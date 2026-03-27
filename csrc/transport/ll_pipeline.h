#pragma once

// ============================================================================
// Low-Latency Pipeline: GDRCopy-signaled async RDMA for LL dispatch/combine
//
// Architecture:
// 1. GPU kernel packs tokens and writes send_counts to GDR-mapped memory
// 2. GPU kernel signals "pack_done" via MMIO store (st.mmio.b8)
// 3. Dedicated CPU worker thread detects flag (~1us), reads counts, issues RDMA
// 4. CPU worker polls CQ, signals "recv_done" via GDRCopy when complete
// 5. GPU/Python reads recv_done flag and proceeds to unpack
//
// This eliminates the Python round-trip (.tolist() → offset computation →
// efa_all_to_all C++ call → cudaDeviceSynchronize) replacing it with
// GPU→CPU→network signaling at ~1us granularity.
// ============================================================================

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <thread>
#include <vector>

#include "efa_transport.h"
#include "gdr_signal.h"
#include "imm_counter.h"

namespace deep_ep {
namespace efa {

// Maximum number of ranks for LL mode
static constexpr int kMaxLLRanks = 128;

// Maximum number of experts
static constexpr int kMaxExperts = 512;

// Shard alignment for multi-NIC RDMA writes
static constexpr size_t kShardAlign = 8192;

// Max outstanding writes per NIC before flow control
static constexpr int kMaxOutstandingPerNic = 64;

// Size of routing info region at start of each RDMA half (bytes)
// Must hold num_experts * sizeof(uint32_t) * num_ranks bytes.
// Worst cases: EP64 + 256 experts = 64*256*4 = 64KB, EP32 + 288 experts = 32*288*4 = 36KB.
// Use kMaxLLRanks * kMaxExperts * 4 = 128*512*4 = 256KB for future-proofing.
static constexpr size_t kRouteRegionSize = 262144;

// ============================================================================
// LLPipelineConfig: Pre-computed, stable config for LL transfers
// Set once during init, read by the worker thread on every iteration.
// ============================================================================
struct LLPipelineConfig {
    int num_ranks = 0;
    int local_rank = 0;  // This GPU's global rank
    int device_id = 0;   // CUDA device ID (local GPU index 0-7)
    int packed_bytes_per_token = 0;
    int num_max_dispatch_tokens_per_rank = 0;

    // RDMA buffer layout
    uint64_t half_rdma = 0;       // Half the RDMA buffer size
    uint64_t data_offset = 0;     // Offset within each half for packed data
    uint64_t slot_size = 0;       // Per-rank slot size = num_max_disp * packed_bpt

    // Pointers into RDMA buffer (GPU memory, RDMA-registered)
    uint8_t* rdma_base = nullptr;           // RDMA buffer base address
    uint8_t* send_data_base = nullptr;      // send half + data_offset
    uint8_t* recv_data_base = nullptr;      // recv half + data_offset
    uint64_t recv_data_rdma_offset = 0;     // Byte offset of recv_data_base in RDMA buffer

    bool initialized = false;
};

// ============================================================================
// CoopConfig: Configuration for cooperative kernel worker mode (Iter 53)
//
// Passed once from Buffer::start_coop_dispatch() to the worker thread.
// Contains all the pointers and parameters the worker needs to:
//   1. Read num_routed from GPU (via GDRCopy MMIO — zero-copy)
//   2. RDMA-scatter routing info to all peers
//   3. Compute recv metadata (process_routing_info)
//   4. Write metadata to GPU (via GDRCopy MMIO — zero-copy)
//   5. Issue dispatch/combine RDMA writes
//   6. Signal GPU via GDR flags
//
// Iter 53: All metadata arrays use GdrVec for zero-copy CPU↔GPU transfer.
// This eliminates ~10 synchronous cudaMemcpy calls from the critical path,
// each of which costs 5-15us of CUDA runtime overhead.
// ============================================================================
struct CoopConfig {
    int num_experts = 0;
    int experts_per_rank = 0;
    int num_topk = 0;
    int max_tokens_per_rank = 0;  // num_max_dispatch_tokens_per_rank
    int coop_token_stride = 0;    // dispatch: token_dim + token_scale_dim + 16
    int combine_token_dim = 0;    // combine: round_up(hidden * 2, 16) for bf16

    // GDR-mapped metadata: CPU writes via MMIO (~1us), GPU reads via global load
    // These replace the old gpu_* raw pointers + cudaMemcpy pattern.
    GdrVec<uint32_t>* gdr_num_routed = nullptr;          // [num_experts] — GPU writes, CPU reads
    GdrVec<uint32_t>* gdr_source_rank = nullptr;         // [max_recv_tokens]
    GdrVec<uint32_t>* gdr_source_offset = nullptr;       // [max_recv_tokens]
    GdrVec<uint32_t>* gdr_padded_index = nullptr;        // [max_recv_tokens]
    GdrVec<uint32_t>* gdr_combine_send_offset = nullptr; // [max_recv_tokens]
    GdrVec<uint32_t>* gdr_tokens_per_expert = nullptr;   // [experts_per_rank]
    GdrVec<uint32_t>* gdr_num_recv_tokens = nullptr;     // [2]: [total, efa_only]

    // Raw GPU pointers (aliases of gdr_*->device_ptr(), for kernel launches)
    uint32_t* gpu_num_routed = nullptr;
    uint32_t* gpu_source_rank = nullptr;
    uint32_t* gpu_source_offset = nullptr;
    uint32_t* gpu_padded_index = nullptr;
    uint32_t* gpu_combine_send_offset = nullptr;
    uint32_t* gpu_tokens_per_expert = nullptr;
    uint32_t* gpu_num_recv_tokens = nullptr;

    // GDR flags for cooperative kernel ↔ worker signaling
    GdrFlag* dispatch_route_done = nullptr;   // GPU→CPU: routing counts ready
    GdrFlag* dispatch_pack_done = nullptr;    // GPU→CPU: dispatch send buffer packed
    GdrFlag* tx_ready = nullptr;              // CPU→GPU: send buffer available
    GdrFlag* num_recv_tokens_flag = nullptr;  // CPU→GPU: recv metadata ready
    GdrFlag* dispatch_recv_flag = nullptr;    // CPU→GPU: all dispatch EFA data arrived
    GdrFlag* dispatch_recv_done = nullptr;    // GPU→CPU: recv kernel complete
    GdrFlag* combine_send_done = nullptr;     // GPU→CPU: combine pack complete
    GdrFlag* combine_recv_flag = nullptr;     // CPU→GPU: all combine EFA data arrived
    GdrFlag* combine_recv_done = nullptr;     // GPU→CPU: combine recv kernel complete

    // GDR counters for auto-signaling GPU when RDMA completions reach target
    GdrCounter* dispatch_rdma_counter = nullptr;  // auto-sets dispatch_recv_flag
    GdrCounter* combine_rdma_counter = nullptr;   // auto-sets combine_recv_flag

    bool initialized = false;
};

// ============================================================================
// LLTransferRequest: Describes one async RDMA transfer (dispatch or combine)
//
// The GPU writes send_counts[i] for each peer i into GDR-mapped memory,
// then signals pack_done_flag. The worker reads counts, computes offsets,
// and issues RDMA writes.
// ============================================================================
struct LLTransferRequest {
    // GDR flags for signaling
    GdrFlag* pack_done_flag = nullptr;      // GPU→CPU: "data is packed in send buffer"
    GdrFlag* recv_done_flag = nullptr;      // CPU→GPU: "all RDMA data received"

    // GDR-mapped send counts (GPU writes, CPU reads)
    // GPU writes send_counts[i] = number of tokens to send to rank i
    GdrVec<int32_t>* send_counts = nullptr;

    // Transfer parameters (set per-request by Python before starting GPU work)
    // For dispatch: recv_offsets[i] = local_rank * slot_size (fixed slot)
    // For combine: recv_offsets[i] = sender_rank_slot_offsets (different layout)
    bool use_fixed_slot_recv = true;  // true for dispatch, false for combine

    // Combine-specific: recv_counts are needed (known before combine starts)
    // For dispatch: recv_counts derived from NCCL count exchange (done by worker)
    int32_t recv_counts[kMaxLLRanks] = {};

    // Optional: receive count source (for combine, counts are already known)
    bool recv_counts_known = false;
};

// ============================================================================
// LLPipelineWorker: Spin-loop CPU thread that polls GDR flags and drives RDMA
//
// The worker thread is pinned to a specific CPU core and runs a tight
// spin loop checking GDR flags. When a flag is detected, it immediately
// begins issuing RDMA writes without any lock/condvar overhead.
//
// State machine:
//   IDLE → (pack_done detected) → ISSUING_RDMA → (all CQ done) → SIGNALING → IDLE
// ============================================================================
class LLPipelineWorker {
public:
    LLPipelineWorker() = default;
    ~LLPipelineWorker();

    // Initialize the worker with transport and config.
    // Allocates GDR flags and starts the spin-loop thread.
    void init(EfaTransport* transport, ImmCounterMap* counters,
              const LLPipelineConfig& config);

    // Get pointers to GDR flags for dispatch and combine
    GdrFlag& dispatch_pack_done()  { return dispatch_pack_done_; }
    GdrFlag& dispatch_recv_done()  { return dispatch_recv_done_; }
    GdrFlag& combine_pack_done()   { return combine_pack_done_; }
    GdrFlag& combine_recv_done()   { return combine_recv_done_; }

    // Get GDR-mapped send count vectors
    GdrVec<int32_t>& dispatch_send_counts() { return dispatch_send_counts_; }
    GdrVec<int32_t>& combine_send_counts()  { return combine_send_counts_; }

    // Get GDR-mapped recv count vectors (for dispatch; combine sets directly)
    GdrVec<int32_t>& dispatch_recv_counts() { return dispatch_recv_counts_; }
    GdrVec<int32_t>& combine_recv_counts()  { return combine_recv_counts_; }

    // Set combine recv counts (known before combine GPU work starts)
    void set_combine_recv_counts(const std::vector<int32_t>& counts);

    // Set combine send counts (known before combine GPU work starts)
    void set_combine_send_counts(const std::vector<int32_t>& counts);

    // Enable cooperative kernel mode: DISPATCH command will wait for
    // dispatch_pack_done_ before reading counts and issuing RDMA.
    void set_coop_mode(bool enabled) { coop_mode_ = enabled; }

    // Configure cooperative kernel parameters (called once, before start_coop_dispatch)
    void set_coop_config(const CoopConfig& config) { coop_config_ = config; }

    // Start cooperative dispatch: arms the worker thread to:
    //   1. Wait dispatch_route_done → read num_routed
    //   2. RDMA-scatter routing info to all peers, wait for all peers' routing info
    //   3. Compute recv metadata (process_routing_info)
    //   4. Write metadata to GPU, signal num_recv_tokens_flag
    //   5. Wait dispatch_pack_done → issue dispatch data RDMA
    //   6. Wait completions → signal dispatch_recv_flag
    //   7. Barrier → signal tx_ready
    void start_coop_dispatch();
    void wait_coop_dispatch_done();

    // Start cooperative combine: arms the worker thread to:
    //   1. Wait combine_send_done
    //   2. Issue combine RDMA writes (using stashed recv_counts/send_counts from dispatch)
    //   3. Wait completions → signal combine_recv_flag
    //   4. Wait combine_recv_done → barrier → signal tx_ready
    void start_coop_combine();
    void wait_coop_combine_done();

    // Access stashed per-rank counts (computed by coop dispatch, valid after wait_coop_dispatch_done)
    const int32_t* coop_send_counts() const { return coop_send_counts_; }
    const int32_t* coop_recv_counts() const { return coop_recv_counts_; }
    int coop_total_send() const { return coop_total_send_; }
    int coop_total_recv() const { return coop_total_recv_; }

    // Iter 61: Reset tag counters and current_tag_ back to initial state.
    // Called between benchmark phases (after dist.barrier) to prevent tag wrapping
    // past kMaxTags (1024). Safe to call only when NO coop operations are in-flight.
    void reset_tags();

    // Trigger dispatch: called after GPU kernel is launched.
    // The worker will wait for pack_done flag and then issue RDMA.
    void start_dispatch();

    // Trigger combine: called after GPU kernel is launched.
    void start_combine();

    // Wait for dispatch RDMA to complete (blocking).
    // After this returns, recv_done_flag is set and recv buffer has data.
    void wait_dispatch_done();

    // Wait for combine RDMA to complete.
    void wait_combine_done();

    // Execute an EFA "barrier" (send imm-only writes to all peers, wait for theirs)
    // This replaces dist.barrier() between dispatch and combine.
    void efa_barrier();

    // Post receive buffers proactively (called once at init)
    void post_initial_recvs();

    // Stop the worker thread
    void stop();

    bool is_running() const { return running_.load(std::memory_order_relaxed); }

    // Update config (e.g., when RDMA buffer layout changes)
    void update_config(const LLPipelineConfig& config);

private:
    void worker_loop();

    // Issue RDMA writes for a transfer, sharding across NICs
    void issue_rdma_writes(const int32_t* send_counts, const int32_t* recv_counts,
                           uint8_t* send_base, uint64_t recv_data_rdma_offset,
                           uint64_t slot_size, bool use_fixed_slot);

    // Wait for all TX and RX completions
    void wait_all_completions(int num_expected_tx, int num_expected_rx, uint16_t tag, int base_rx = 0);

    // Post receives for an upcoming transfer
    void post_recvs_for_transfer(int num_expected);

    EfaTransport* transport_ = nullptr;
    ImmCounterMap* counters_ = nullptr;
    LLPipelineConfig config_;

    // GDR flags for dispatch
    GdrFlag dispatch_pack_done_;
    GdrFlag dispatch_recv_done_;
    GdrVec<int32_t> dispatch_send_counts_;
    GdrVec<int32_t> dispatch_recv_counts_;

    // GDR flags for combine
    GdrFlag combine_pack_done_;
    GdrFlag combine_recv_done_;
    GdrVec<int32_t> combine_send_counts_;
    GdrVec<int32_t> combine_recv_counts_;

    // Worker thread
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    // Command queue (simple: one slot, signaled by atomic)
    enum class Command : int {
        NONE = 0,
        DISPATCH = 1,
        COMBINE = 2,
        BARRIER = 3,
        COOP_DISPATCH = 4,
        COOP_COMBINE = 5,
    };
    std::atomic<int> pending_command_{0};
    std::atomic<bool> command_done_{false};

    // Separate done flag for coop commands (so they don't conflict with regular done)
    std::atomic<bool> coop_done_{false};

    // Transfer tag (incremented per transfer)
    // Iter 51: 3 tags per iteration (route_scatter, dispatch, combine) — no barrier tags.
    // With 10-bit encoding (1024 tags), starting at 128, supports (1024-128)/3 = 298 iterations.
    // Use tags 128+ to avoid collision with efa_all_to_all (which uses 0-127)
    uint16_t current_tag_ = 128;

    // Cooperative kernel mode flag
    bool coop_mode_ = false;

    // Cooperative kernel config (set via set_coop_config)
    CoopConfig coop_config_;

    // Pre-allocated tags for the ENTIRE iteration (3 tags: route, dispatch_data, combine_data).
    // Allocated at the END of the previous iteration's combine handler.
    // For iteration 0, they are allocated at the start of the dispatch handler.
    // With absolute counting (base always 0), no baseline tracking needed.
    bool tags_preallocated_ = false;  // true after first combine completes
    uint16_t preallocated_route_scatter_tag_ = 0;
    int preallocated_route_scatter_base_ = 0;
    uint16_t preallocated_dispatch_data_tag_ = 0;
    int preallocated_dispatch_data_base_ = 0;
    uint16_t preallocated_dispatch_barrier_tag_ = 0;
    int preallocated_dispatch_barrier_base_ = 0;
    uint16_t preallocated_combine_data_tag_ = 0;
    int preallocated_combine_data_base_ = 0;
    uint16_t preallocated_combine_barrier_tag_ = 0;
    int preallocated_combine_barrier_base_ = 0;

    // Stashed routing info from coop dispatch (used by coop combine)
    int32_t coop_send_counts_[kMaxLLRanks] = {};  // per-rank send counts (dispatch direction)
    int32_t coop_recv_counts_[kMaxLLRanks] = {};   // per-rank recv counts (dispatch direction)
    int coop_total_recv_ = 0;
    int coop_total_send_ = 0;

    // Iter 60: Route data corruption flag.
    // Set by compute_send_recv_counts() when any per-rank count is negative or
    // unreasonably large. When set, dispatch and combine handlers skip RDMA writes
    // and signal GDR flags with zero-token metadata so cooperative kernels unblock.
    bool coop_data_corrupted_ = false;

    // Combine recv offsets on each remote peer (in token units):
    // combine_remote_recv_offset_[j] = offset (in tokens) where our data should land
    // on peer j's combine recv buffer = sum of dispatch_send_counts_on_j for ranks < our rank.
    int64_t combine_remote_recv_offset_[kMaxLLRanks] = {};

    // GDR vectors for routing info exchange
    // Each rank sends its num_routed row to all peers.
    // route_send_counts_: GDR-mapped, holds our num_routed[num_experts] for peers to read
    // route_recv_: received num_routed from all peers, indexed [rank * num_experts + expert]
    // Allocated in a flat GPU region in the RDMA buffer's route exchange area.

    // Per-expert routing info received from all peers (host-side, populated from RDMA recv)
    uint32_t route_recv_buf_[kMaxLLRanks * kMaxExperts] = {};

    // GDR mapping of RDMA buffer route regions (Iter 53)
    // The RDMA buffer's route send region [0, kRouteRegionSize) and route recv region
    // [half_rdma, half_rdma + kRouteRegionSize) are GDR-mapped so the CPU worker can
    // read/write them directly via MMIO without cudaMemcpy.
    // rdma_route_send_gdr_: maps the send half route region
    // rdma_route_recv_gdr_: maps the recv half route region
    gdr_t rdma_gdr_ = nullptr;
    gdr_mh_t rdma_send_mh_ = {};
    void* rdma_send_cpu_map_ = nullptr;
    volatile uint8_t* rdma_send_cpu_ptr_ = nullptr;  // CPU-accessible MMIO ptr to RDMA send route region
    gdr_mh_t rdma_recv_mh_ = {};
    void* rdma_recv_cpu_map_ = nullptr;
    volatile uint8_t* rdma_recv_cpu_ptr_ = nullptr;  // CPU-accessible MMIO ptr to RDMA recv route region
    bool rdma_gdr_mapped_ = false;

    // Initialize GDR mapping of RDMA buffer route regions
    void init_rdma_gdr_mapping();

    // Coop RDMA handler methods
    void handle_coop_dispatch();
    void handle_coop_combine();

    // RDMA scatter routing info to all peers — async version (Iter 56, no TX wait)
    void rdma_scatter_route_info_async(const uint32_t* num_routed, int num_experts, uint16_t tag);

    // Wait for all peers' routing info via RDMA
    void rdma_wait_route_info(int num_experts, uint16_t tag, int base_rx);

    // Compute recv metadata from routing info (C++ port of Python metadata computation)
    void process_routing_info(const uint32_t* all_num_routed, int num_experts,
                              int experts_per_rank, int num_ranks, int local_rank,
                              int max_tokens_per_rank);

    // Issue coop dispatch RDMA writes (slot-based recv layout)
    // tag must be pre-allocated by caller; base_rx = snapshot of counter before sends
    void issue_coop_dispatch_rdma(const int32_t* send_counts, const int32_t* recv_counts,
                                   int coop_token_stride, int max_tokens_per_rank,
                                   uint16_t tag, int base_rx);

    // State returned by submit_coop_dispatch_rdma for deferred completion waiting
    struct DispatchRdmaState {
        int total_writes = 0;
        int total_tx_completed = 0;
        int num_expected_recvs = 0;
        uint16_t tag = 0;
        int base_rx = 0;
    };

    // Submit coop dispatch RDMA writes (non-blocking: submits writes, arms GdrCounter,
    // but does NOT call wait_all_completions). Returns state for deferred waiting.
    DispatchRdmaState submit_coop_dispatch_rdma(
        const int32_t* send_counts, const int32_t* recv_counts,
        int coop_token_stride, int max_tokens_per_rank,
        uint16_t tag, int base_rx);

    // Complete the dispatch RDMA phase: wait for all TX+RX completions.
    void complete_coop_dispatch_rdma(const DispatchRdmaState& state);

    // Compute send/recv counts from routing info (fast, no metadata upload).
    // Extracted from process_routing_info for early computation.
    void compute_send_recv_counts(const uint32_t* all_num_routed, int num_experts,
                                   int experts_per_rank, int num_ranks, int local_rank);

    // Compute combine_remote_recv_offset and upload recv metadata to GPU.
    // Must be called AFTER compute_send_recv_counts() populated coop_recv_counts_.
    void upload_recv_metadata(const uint32_t* all_num_routed, int num_experts,
                               int experts_per_rank, int num_ranks, int local_rank,
                               int max_tokens_per_rank);

    // Issue coop combine RDMA writes (prefix-sum recv layout)
    // tag must be pre-allocated by caller; base_rx = snapshot of counter before sends
    void issue_coop_combine_rdma(const int32_t* send_counts, const int32_t* recv_counts,
                                  int combine_token_dim, uint16_t tag, int base_rx);

    // Persistent recv pool tracking (mirrors efa_all_to_all's g_posted_per_nic)
    // posted_per_nic_[n] tracks how many fi_recv() calls are outstanding on NIC n.
    // Decremented when RX CQ entries are polled, incremented when we post recvs.
    static constexpr int kMaxNicsPerGpu = 4;
    int posted_per_nic_[kMaxNicsPerGpu] = {};

    // Iter 56: Outstanding route scatter TX completions (drained by overlapped wait)
    int route_scatter_outstanding_tx_ = 0;

    // Iter 57: Active GdrCounter for auto-signaling GPU when RDMA completions arrive.
    // When non-null, every RX CQ entry whose tag matches active_counter_tag_ triggers
    // active_gdr_counter_->inc(). When the counter reaches the target (set via wait()),
    // the associated GdrFlag is auto-set, notifying the GPU.
    GdrCounter* active_gdr_counter_ = nullptr;
    uint16_t active_counter_tag_ = 0;

    // Helper: record a CQ entry in counters_ AND increment active GdrCounter if tag matches.
    // Called at EVERY site where we process RX CQ entries with immediate data.
    inline void record_cq_entry(uint32_t imm_data) {
        counters_->record(imm_data);
        if (active_gdr_counter_ != nullptr) {
            uint16_t tag = ImmCounterMap::decode_tag(imm_data);
            if (tag == active_counter_tag_) {
                active_gdr_counter_->inc();
            }
        }
    }

    // Ensure enough recvs are posted on all NICs for an upcoming transfer
    void ensure_recv_pool(int target_per_nic);

    // Profiling
    struct Stats {
        double total_rdma_us = 0;
        double total_cq_wait_us = 0;
        double total_flag_wait_us = 0;
        int count = 0;
    };
    Stats dispatch_stats_;
    Stats combine_stats_;
};

}  // namespace efa
}  // namespace deep_ep
