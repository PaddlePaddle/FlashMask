// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cmath>
#include "debug_logger.cuh"

namespace flashmask {

// Chunks per backward segment (bwd runs nranks / chunks segments). One segment
// (chunks == nranks) saves per-segment drain tails for head_dim >= 256 with small B * H_kv,
// but its fp32 dK/dV accumulators cover all ranks' rows, so each is capped at 1 GiB.
// nranks <= 8: split AG / RS / reduce kernels exist only for chunks in {1, 2, 4, 8}.
inline int get_num_chunk_per_segment(int b_kv, int local_seqlen_k, int h_kv, int d_kv, int nranks) {
    const bool small_kv = b_kv * h_kv <= 4 && (h_kv == 1 || local_seqlen_k >= 32768);
    const long long accum_bytes = 4LL * b_kv * h_kv * local_seqlen_k * nranks * d_kv;
    if (d_kv >= 256 && nranks <= 8 && small_kv && accum_bytes <= (1LL << 30)) return nranks;
    // 32K+ seqlen does not need chunk grouping
    if (local_seqlen_k >= 32768) return 1;
    // logarithm heuristic. Note(heqianyue): using 2 chunks per segement is actually faster for CP16
    int power = int(std::floor(std::log2(nranks) * 0.5));
    int num_chunks = std::pow(2, power);
    return num_chunks;
}

// Power-of-two bucket of S_local. Buffers and the chunks per segment follow the largest bucket
// seen, so a varying S_local reallocates or re-lays out a window only when it outgrows it.
inline int seqlen_bucket(int s_local) {
    int bucket = 1;
    while (bucket < s_local) bucket <<= 1;
    return bucket;
}

}   // namespace flashmask
