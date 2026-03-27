#pragma once

#include <cstdint>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace deep_ep {
namespace efa {

// ============================================================================
// ImmCounter: Tracks RDMA write completion via immediate data
// 
// When a remote rank performs an RDMA write with immediate data, the local
// RX CQ receives the immediate data value. ImmCounter uses this to count
// how many writes from each source rank have completed.
//
// The immediate data encodes: [src_rank (16 bits) | counter_value (16 bits)]
// This allows tracking per-rank completion without remote atomics.
// ============================================================================
class ImmCounter {
public:
    ImmCounter() = default;

    // Initialize with number of ranks
    void init(int num_ranks);

    // Encode immediate data: packs rank and count into 32 bits
    static uint32_t encode(int src_rank, uint16_t count) {
        return (static_cast<uint32_t>(src_rank) << 16) |
               static_cast<uint32_t>(count);
    }

    // Decode source rank from immediate data
    static int decode_rank(uint32_t imm_data) {
        return static_cast<int>(imm_data >> 16);
    }

    // Decode count from immediate data
    static uint16_t decode_count(uint32_t imm_data) {
        return static_cast<uint16_t>(imm_data & 0xFFFF);
    }

    // Record a completion from a source rank with a count value
    void record(int src_rank, uint16_t count);

    // Get the total number of tokens received from a specific rank
    int get_count(int src_rank) const;

    // Get the total number of tokens received from all ranks
    int get_total_count() const;

    // Reset all counters
    void reset();

    // Check if all expected tokens have been received
    bool is_complete(const int* expected_per_rank, int num_ranks) const;

private:
    // Use unique_ptr to array since std::atomic is non-copyable/non-movable
    std::unique_ptr<std::atomic<int>[]> counts_;
    int num_ranks_ = 0;
};

// ============================================================================
// ImmCounterMap: Maps (tag, src_rank) -> count for multi-operation tracking
// Used when multiple dispatch/combine operations can be in-flight.
// The tag is embedded in higher bits of the immediate data.
//
// Iter 51: Changed to 10-bit tags + 6-bit rank to support EP64 (64 ranks).
// Encoding: [tag (10 bits) | src_rank (6 bits) | count (16 bits)]
// This supports up to 64 ranks and 1024/5 = 204 iterations before wrapping.
//
// Iter 44: Dual tracking — counts_ tracks CQ entries (count=1 always),
// token_counts_ tracks per-rank token counts encoded in special imm data.
// 
// Iter 44 FIX: Token counts are stored alongside the CQ count at which they
// were set (token_count_at_cq_). This allows distinguishing fresh early arrivals
// from stale values left over from a previous epoch (tag wrap-around) or from
// intermediate non-dispatch calls.
// ============================================================================
class ImmCounterMap {
public:
    ImmCounterMap() = default;

    void init(int num_ranks);

    // Iter 51: Full encode: [tag (10 bits) | src_rank (6 bits) | count (16 bits)]
    // 10-bit tags (1024 values) = 298 iterations at 3 tags/iter (starting from 128).
    // 6-bit rank (64 values) supports up to EP64.
    // For CQ entry tracking, count=1 always.
    static uint32_t encode(uint16_t tag, uint8_t src_rank, uint16_t count) {
        return (static_cast<uint32_t>(tag & 0x3FF) << 22) |
               (static_cast<uint32_t>(src_rank & 0x3F) << 16) |
               static_cast<uint32_t>(count);
    }

    // Iter 44/51: Encode with token_count flag in bit 15 of count field.
    // When bit 15 is set, bits [14:0] contain the actual token count (max 32767).
    // When bit 15 is clear, the count is a normal CQ entry count (always 1).
    static uint32_t encode_with_tokens(uint16_t tag, uint8_t src_rank, uint16_t token_count) {
        uint16_t encoded = (1u << 15) | (token_count & 0x7FFF);
        return (static_cast<uint32_t>(tag & 0x3FF) << 22) |
               (static_cast<uint32_t>(src_rank & 0x3F) << 16) |
               static_cast<uint32_t>(encoded);
    }

    static uint16_t decode_tag(uint32_t imm_data) {
        return static_cast<uint16_t>((imm_data >> 22) & 0x3FF);
    }

    static uint8_t decode_rank(uint32_t imm_data) {
        return static_cast<uint8_t>((imm_data >> 16) & 0x3F);
    }

    static uint16_t decode_count(uint32_t imm_data) {
        return static_cast<uint16_t>(imm_data & 0xFFFF);
    }

    // Iter 44: Check if imm data carries a token count (bit 15 set)
    static bool has_token_count(uint32_t imm_data) {
        return (imm_data & 0x8000) != 0;
    }

    // Iter 44: Extract token count from imm data (bits [14:0])
    static uint16_t decode_token_count(uint32_t imm_data) {
        return static_cast<uint16_t>(imm_data & 0x7FFF);
    }

    // Record a completion — always increments CQ entry count by 1.
    // Iter 44: Also records token count if bit 15 is set in count field,
    // along with the CQ count at which it was set (for freshness detection).
    void record(uint32_t imm_data);

    // Get CQ entry count for a (tag, rank) pair
    int get_count(uint16_t tag, int rank) const;

    // Get total CQ entry count for a tag across all ranks
    int get_total(uint16_t tag) const;

    // Iter 44 FIX: Get token count for a (tag, rank) pair, but only if it's "fresh"
    // (i.e., was set at or after cq_baseline for this rank). Returns -1 if stale or not set.
    int get_fresh_token_count(uint16_t tag, int rank, int cq_baseline) const;

    // Iter 44: Fill recv_token_counts vector for a tag (all ranks).
    // Uses cq_baselines to filter stale values.
    void get_all_fresh_token_counts(uint16_t tag, const std::vector<int>& cq_baselines,
                                     std::vector<int32_t>& out) const;

    // Reset counters for a specific tag
    void reset_tag(uint16_t tag);

    // Reset all counters
    void reset_all();

private:
    static constexpr int kMaxTags = 1024;  // Iter 51: 10-bit tags (6-bit rank for EP64)
    // Flat array: [tag * num_ranks + rank] -> CQ entry count
    std::unique_ptr<std::atomic<int>[]> counts_;
    // Iter 44: Flat array: [tag * num_ranks + rank] -> token count (from first shard)
    std::unique_ptr<std::atomic<int>[]> token_counts_;
    // Iter 44 FIX: Flat array: [tag * num_ranks + rank] -> CQ count at which token_count was set
    // Used to detect stale token_counts from previous epoch/call.
    std::unique_ptr<std::atomic<int>[]> token_count_at_cq_;
    int num_ranks_ = 0;
};

}  // namespace efa
}  // namespace deep_ep
