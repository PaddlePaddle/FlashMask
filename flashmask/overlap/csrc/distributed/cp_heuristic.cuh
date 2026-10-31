// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 The FlashMask Authors. All rights reserved.

#pragma once
#include <cmath>
#include "debug_logger.cuh"

namespace flashmask {

// Chunks per backward segment; the bwd runs nranks / chunks segments. Each extra segment
// costs a bwd-launch drain tail (a head_dim >= 256 bwd CTA fills an SM), a per-segment dK/dV
// postprocess and one more AG/RS enqueue. One segment (chunks == nranks) instead leaves the
// dK/dV RS of every remote chunk, B * S_local * H_kv * D elements each, behind the bwd
// launch. That RS stays small next to the saved tails for MQA, and for B * H_kv <= 4 once
// S_local >= 32K, where the default below splits into nranks segments. One segment also sizes
// the bwd fp32 dK/dV accumulators for all S_local * nranks rows (the default sizes them for
// one segment's rows), so it is capped at 1 GiB per accumulator. nranks <= 8: the split AG /
// RS / reduce kernels are instantiated for chunks in {1, 2, 4, 8}.
inline int get_num_chunk_per_segment(int b_kv, int local_seqlen_k, int h_kv, int d_kv, int nranks) {
    const bool small_kv = b_kv * h_kv <= 4 && (h_kv == 1 || local_seqlen_k >= 32768);
    const long long accum_bytes = 4LL * b_kv * h_kv * local_seqlen_k * nranks * d_kv;
    if (d_kv >= 256 && nranks <= 8 && small_kv && accum_bytes <= (1LL << 30)) return nranks;
    // 32K+ seqlen does not need chunk grouping
    if (local_seqlen_k >= 32768) return 1;
    // logarithm heuristic. Note(heqianyue): using 2 chunks per segement is actually faster for CP16
    int power = int(std::floor(std::log2(nranks) * 0.5));
    int num_chunks = std::pow(2, power);
    DEBUG_PRINT("BWD RS-overlap chunk per segment: %d\n", num_chunks);
    return num_chunks;
}

}   // namespace flashmask
