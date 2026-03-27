#include "imm_counter.h"

namespace deep_ep {
namespace efa {

void ImmCounter::init(int num_ranks) {
    num_ranks_ = num_ranks;
    counts_ = std::make_unique<std::atomic<int>[]>(num_ranks);
    for (int i = 0; i < num_ranks; ++i) counts_[i].store(0);
}

void ImmCounter::record(int src_rank, uint16_t count) {
    counts_[src_rank].fetch_add(static_cast<int>(count),
                                 std::memory_order_relaxed);
}

int ImmCounter::get_count(int src_rank) const {
    return counts_[src_rank].load(std::memory_order_relaxed);
}

int ImmCounter::get_total_count() const {
    int total = 0;
    for (int i = 0; i < num_ranks_; ++i) {
        total += counts_[i].load(std::memory_order_relaxed);
    }
    return total;
}

void ImmCounter::reset() {
    for (int i = 0; i < num_ranks_; ++i) counts_[i].store(0, std::memory_order_relaxed);
}

bool ImmCounter::is_complete(const int* expected_per_rank, int num_ranks) const {
    for (int i = 0; i < num_ranks; ++i) {
        if (counts_[i].load(std::memory_order_relaxed) < expected_per_rank[i])
            return false;
    }
    return true;
}

// ============================================================================
// ImmCounterMap (Iter 51: 10-bit tags, 6-bit rank for EP64)
// ============================================================================

void ImmCounterMap::init(int num_ranks) {
    num_ranks_ = num_ranks;
    int total = kMaxTags * num_ranks;
    counts_ = std::make_unique<std::atomic<int>[]>(total);
    for (int i = 0; i < total; ++i) counts_[i].store(0);
    // Iter 44: Initialize token count tracking
    token_counts_ = std::make_unique<std::atomic<int>[]>(total);
    for (int i = 0; i < total; ++i) token_counts_[i].store(-1);  // -1 = not set
    // Iter 44 FIX: Initialize CQ count at which token_count was set
    token_count_at_cq_ = std::make_unique<std::atomic<int>[]>(total);
    for (int i = 0; i < total; ++i) token_count_at_cq_[i].store(-1);  // -1 = not set
}

void ImmCounterMap::record(uint32_t imm_data) {
    uint16_t tag = decode_tag(imm_data);  // Already 10-bit (0-1023) from decode
    uint8_t rank = decode_rank(imm_data);
    if (rank >= num_ranks_) return;  // Safety check (tag already < kMaxTags from decode)
    int idx = static_cast<int>(tag) * num_ranks_ + rank;
    // Iter 44: Always increment CQ entry count by 1 (regardless of count field value)
    int new_count = counts_[idx].fetch_add(1, std::memory_order_relaxed) + 1;
    // Iter 44: If bit 15 is set, record the token count AND the CQ count at which it was set
    if (has_token_count(imm_data)) {
        uint16_t tc = decode_token_count(imm_data);
        token_counts_[idx].store(static_cast<int>(tc), std::memory_order_relaxed);
        token_count_at_cq_[idx].store(new_count, std::memory_order_relaxed);
    }
}

int ImmCounterMap::get_count(uint16_t tag, int rank) const {
    tag %= kMaxTags;  // Modular tag wrapping for long-running benchmarks
    if (rank >= num_ranks_) return 0;
    return counts_[static_cast<int>(tag) * num_ranks_ + rank].load(
        std::memory_order_relaxed);
}

int ImmCounterMap::get_total(uint16_t tag) const {
    tag %= kMaxTags;  // Modular tag wrapping
    int total = 0;
    int base = static_cast<int>(tag) * num_ranks_;
    for (int i = 0; i < num_ranks_; ++i) {
        total += counts_[base + i].load(std::memory_order_relaxed);
    }
    return total;
}

int ImmCounterMap::get_fresh_token_count(uint16_t tag, int rank, int cq_baseline) const {
    tag %= kMaxTags;  // Modular tag wrapping
    if (rank >= num_ranks_) return -1;
    int idx = static_cast<int>(tag) * num_ranks_ + rank;
    int tc = token_counts_[idx].load(std::memory_order_relaxed);
    if (tc < 0) return -1;  // Never set
    int at_cq = token_count_at_cq_[idx].load(std::memory_order_relaxed);
    if (at_cq > cq_baseline) {
        // Token count was set AFTER the baseline → it's from the current epoch
        return tc;
    }
    return -1;  // Stale (from a previous epoch)
}

void ImmCounterMap::get_all_fresh_token_counts(uint16_t tag,
                                                const std::vector<int>& cq_baselines,
                                                std::vector<int32_t>& out) const {
    out.resize(num_ranks_);
    tag %= kMaxTags;  // Modular tag wrapping
    int base = static_cast<int>(tag) * num_ranks_;
    for (int i = 0; i < num_ranks_; ++i) {
        int tc = token_counts_[base + i].load(std::memory_order_relaxed);
        int at_cq = token_count_at_cq_[base + i].load(std::memory_order_relaxed);
        if (tc >= 0 && at_cq > cq_baselines[i]) {
            out[i] = tc;
        } else {
            out[i] = 0;  // Not set or stale → 0 tokens
        }
    }
}

void ImmCounterMap::reset_tag(uint16_t tag) {
    tag %= kMaxTags;  // Modular tag wrapping
    int base = static_cast<int>(tag) * num_ranks_;
    for (int i = 0; i < num_ranks_; ++i) {
        counts_[base + i].store(0, std::memory_order_relaxed);
        token_counts_[base + i].store(-1, std::memory_order_relaxed);
        token_count_at_cq_[base + i].store(-1, std::memory_order_relaxed);
    }
}

void ImmCounterMap::reset_all() {
    int total = kMaxTags * num_ranks_;
    for (int i = 0; i < total; ++i) {
        counts_[i].store(0, std::memory_order_relaxed);
        token_counts_[i].store(-1, std::memory_order_relaxed);
        token_count_at_cq_[i].store(-1, std::memory_order_relaxed);
    }
}

}  // namespace efa
}  // namespace deep_ep
