// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once

/**
 * Hierarchical Rank Mapping for AG Overlap Communication
 *
 * Provides mapping from logical chunk position to target rank, following the
 * "congruence group first, intra-node redistribution second" traversal order.
 *
 * Traversal order (BWD, left-to-right):
 *   [local, congruence_partners..., same_node_rank_1, its_congruence_partners..., ...]
 *
 * FWD is the physical reverse of BWD (right-to-left traversal).
 *
 * When num_nodes == 1 (single node), degenerates to the current circular shift order.
 *
 * Performance note: gpus_per_node and num_nodes are always powers of 2 (typical values: 4 or 8).
 * All mod/div/mul by these values are replaced with bit operations (&, >>, <<).
 */

namespace flashmask::hier {

/**
 * Integer log2 for power-of-2 values. Used to convert div/mod/mul to bit ops.
 * Precondition: x must be a positive power of 2.
 */
__device__ __host__ inline int ilog2(unsigned x) {
#ifdef __CUDA_ARCH__
    return __ffs(x) - 1;   // __ffs returns 1-indexed position of lowest set bit
#else
    return __builtin_ctz(x);  // counts trailing zeros = log2 for power of 2
#endif
}

struct HierRankInfo {
    int target_rank;    // The rank whose KV data is at this position
    int src_pe;         // The PE to fetch from (Phase 1: target_rank; Phase 2: same-node rank holding the data)
    int logical_pos;    // Logical chunk position (0 = local, 1 = first remote, ...)
    bool is_phase1;     // True = cross-node congruence group fetch (IB RDMA)
    bool is_local;      // True = local chunk (no fetch needed)
};

/**
 * Map a logical chunk position to rank info in the hierarchical traversal order.
 *
 * Rank layout: rank = slot + node * gpus_per_node
 *   slot = rank & (gpus_per_node - 1)   (intra-node index)
 *   node = rank >> log2(gpus_per_node)   (node index)
 */
__device__ __host__ inline HierRankInfo hier_map_chunk(
    int logical_pos,
    int my_pe,
    int total_n_pes,
    int gpus_per_node
) {
    const int log2_gpn  = ilog2(gpus_per_node);
    const int gpn_mask  = gpus_per_node - 1;
    const int num_nodes = total_n_pes >> log2_gpn;
    const int log2_nn   = ilog2(num_nodes);
    const int nn_mask   = num_nodes - 1;
    const int my_slot   = my_pe & gpn_mask;
    const int my_node   = my_pe >> log2_gpn;

    HierRankInfo info;
    info.logical_pos = logical_pos;

    // Slot 0: local rank's congruence group (positions 0..num_nodes-1)
    //   Position 0: local chunk
    //   Position 1..num_nodes-1: congruence partners (Phase 1, cross-node)
    // Slot s (s >= 1): same-node rank's congruence group (Phase 2, intra-node)
    //   Positions s*num_nodes .. (s+1)*num_nodes-1
    if (logical_pos < num_nodes) {
        info.target_rank = my_slot | (((my_node + logical_pos) & nn_mask) << log2_gpn);
        info.is_local  = (logical_pos == 0);
        info.is_phase1 = (logical_pos > 0);
        info.src_pe    = info.target_rank;
        return info;
    }

    const int adj  = logical_pos - num_nodes;
    const int slot = (adj >> log2_nn) + 1;    // slot index (1-based)
    const int sub  = adj & nn_mask;            // sub index within the slot
    const int base = (my_slot + slot) & gpn_mask;

    info.target_rank = base | (((my_node + sub) & nn_mask) << log2_gpn);
    info.is_local  = false;
    info.is_phase1 = false;
    info.src_pe    = base | (my_node << log2_gpn);  // base slot on our node
    return info;
}

/**
 * Compute the target rank for a given logical chunk position.
 * Same result as hier_map_chunk(...).target_rank but without computing the full struct.
 */
__device__ __host__ inline int hier_target_rank(
    int logical_pos,
    int my_pe,
    int total_n_pes,
    int gpus_per_node
) {
    const int log2_gpn  = ilog2(gpus_per_node);
    const int gpn_mask  = gpus_per_node - 1;
    const int num_nodes = total_n_pes >> log2_gpn;
    const int log2_nn   = ilog2(num_nodes);
    const int nn_mask   = num_nodes - 1;
    const int my_slot   = my_pe & gpn_mask;
    const int my_node   = my_pe >> log2_gpn;

    if (logical_pos < num_nodes)
        return my_slot | (((my_node + logical_pos) & nn_mask) << log2_gpn);

    const int adj  = logical_pos - num_nodes;
    const int slot = (adj >> log2_nn) + 1;
    const int sub  = adj & nn_mask;
    return ((my_slot + slot) & gpn_mask) | (((my_node + sub) & nn_mask) << log2_gpn);
}

/**
 * Compute the chunk position of target_rank's data in src_pe's SR buffer.
 * Used to calculate src_addr for Phase 2 (intra-node) fetches.
 *
 * Precondition: target_rank and src_pe are congruent (same slot, different nodes).
 * @return Chunk position (0-based) in src_pe's hierarchical order
 */
__device__ __host__ inline int hier_position_in_src_pe(
    int target_rank,
    int src_pe,
    int total_n_pes,
    int gpus_per_node
) {
    const int log2_gpn  = ilog2(gpus_per_node);
    const int num_nodes = total_n_pes >> log2_gpn;
    const int nn_mask   = num_nodes - 1;
    const int src_node  = src_pe >> log2_gpn;
    const int tgt_node  = target_rank >> log2_gpn;
    return (tgt_node - src_node + num_nodes) & nn_mask;
}

/**
 * Forward seqlen_id for a given logical chunk position (the SR buffer is filled right to left).
 */
__device__ __host__ inline int hier_seqlen_id(
    int logical_pos,
    int row_within_chunk,
    int row_per_block,
    int nranks,
    int S_chunk
) {
    return (nranks - 1 - logical_pos) * S_chunk + row_within_chunk * row_per_block;
}

/**
 * Forward: seqlen offset of the source chunk in src_pe's SR buffer (filled right to left).
 * Phase 1: src_pe's local chunk, at position nranks-1.
 * Phase 2: position of target_rank's data in src_pe's SR buffer.
 */
__device__ __host__ inline int hier_src_chunk_offset(
    const HierRankInfo& info,
    int nranks,
    int S_chunk,
    int total_n_pes,
    int gpus_per_node
) {
    if (info.is_phase1 || info.is_local) {
        return (nranks - 1) * S_chunk;
    }
    int src_chunk_pos = hier_position_in_src_pe(info.target_rank, info.src_pe, total_n_pes, gpus_per_node);
    return (nranks - 1 - src_chunk_pos) * S_chunk;
}

/** Location of a chunk in the backward split-AG SR buffer: [segment][batch][chunk][row]. */
struct HierBwdChunk {
    int segment;        // segment region, each num_batch * num_chunks * S_chunk rows
    int chunk;          // chunk within the segment's batch row block
};

/**
 * Backward: where the source chunk sits in src_pe's split-AG SR buffer.
 * Phase 1 / local: target_rank's local KV, which lives at segment 0, chunk 0.
 * Phase 2: src_pe fetched target_rank at its logical position p (< num_nodes) while
 *   running segment p / num_chunks, into that segment's region at chunk p % num_chunks.
 */
__device__ __host__ inline HierBwdChunk hier_bwd_src_chunk(
    const HierRankInfo& info,
    int num_chunks,
    int total_n_pes,
    int gpus_per_node
) {
    if (info.is_phase1 || info.is_local) {
        return {0, 0};
    }
    const int src_chunk_pos = hier_position_in_src_pe(info.target_rank, info.src_pe, total_n_pes, gpus_per_node);
    return {src_chunk_pos / num_chunks, src_chunk_pos % num_chunks};
}

/**
 * Inverse of hier_map_chunk: given that MY PE is the consumer at logical_pos,
 * which PE is the sender (producer) of that chunk?
 *
 * Derivation:
 *   For pos < num_nodes (congruence group):
 *     target = X_slot | ((X_node + pos) & nn_mask) << log2_gpn
 *     X_slot = my_slot,  X_node = (my_node - pos) & nn_mask
 *   For pos >= num_nodes (intra-node redistribution):
 *     adj = pos - num_nodes; slot = (adj >> log2_nn) + 1; sub = adj & nn_mask
 *     target = (X_slot + slot) & gpn_mask  |  ((X_node + sub) & nn_mask) << log2_gpn
 *     X_slot = (my_slot - slot) & gpn_mask
 *     X_node = (my_node - sub) & nn_mask
 */
__device__ __host__ inline int hier_sender_at_pos(
    int logical_pos,
    int my_pe,
    int total_n_pes,
    int gpus_per_node
) {
    const int log2_gpn  = ilog2(gpus_per_node);
    const int gpn_mask  = gpus_per_node - 1;
    const int num_nodes = total_n_pes >> log2_gpn;
    const int log2_nn   = ilog2(num_nodes);
    const int nn_mask   = num_nodes - 1;
    const int my_slot   = my_pe & gpn_mask;
    const int my_node   = my_pe >> log2_gpn;

    if (logical_pos < num_nodes) {
        const int sender_node = (my_node - logical_pos + num_nodes) & nn_mask;
        return my_slot | (sender_node << log2_gpn);
    }

    const int adj         = logical_pos - num_nodes;
    const int slot        = (adj >> log2_nn) + 1;
    const int sub         = adj & nn_mask;
    const int sender_slot = (my_slot - slot + gpus_per_node) & gpn_mask;
    const int sender_node = (my_node - sub + num_nodes) & nn_mask;
    return sender_slot | (sender_node << log2_gpn);
}

// ---------------------------------------------------------------------------
// Non-hot-path helpers (called once on host, no bit-op optimization needed)
// ---------------------------------------------------------------------------

__device__ __host__ inline bool hier_is_effective(int total_n_pes, int gpus_per_node) {
    return total_n_pes > gpus_per_node;
}

/** Number of consumers that will read our local KV: (num_nodes-1) + (gpus_per_node-1) */
__device__ __host__ inline int hier_local_refcount(int total_n_pes, int gpus_per_node) {
    return total_n_pes / gpus_per_node - 1 + gpus_per_node - 1;
}

} // namespace flashmask::hier
