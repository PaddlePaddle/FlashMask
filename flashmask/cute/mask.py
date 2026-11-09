# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

from typing import Optional, Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from . import utils
from . import layout_utils


@cute.jit
def mask_r2p(X: cute.Tensor, col_limit: Int32, arch: int = 90, rank1: bool = False) -> None:
    # Bit manipulation, compiles down to the R2P instruction
    # For sm100: we know that tScS_t2r[i][1] == i, for the particular tmem copy atom we're using.
    # For sm90: instead of comparing limit to 0, 1, 8, 9, 16, 17, ...,
    # we compare a transformed version of limit to 0, 1, 2, 3, 4, 5, ...
    if const_expr(arch == 90):
        col_limit_transformed = col_limit // 8 * 2 + min(col_limit % 8, 2)
    else:
        col_limit_transformed = col_limit
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # Don't need to clamp to 32 since the shr.u32 instruction does that already
        col_limit_right_s = max(col_limit_transformed - s * 24, 0)
        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
        mask = (1 << col_limit_right_s) - 1
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            in_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
                # This is the equivalent of:
                # X[s * 24 + i] = X[s * 24 + i] if col_limit_right_s <= i else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def flashmask_count_le(s: cute.Tensor, base: Int32, n: cutlass.Constexpr[int], row: Int32) -> Int32:
    # Number of c in [0, n) with s[base + c] <= row, for s non-decreasing over c (n a power
    # of 2). Branchless lower bound: the steps sum to n - 1, the last probe adds the final 1.
    pos = Int32(0)
    for k in cutlass.range_constexpr(n.bit_length() - 1):
        step = const_expr(n >> (k + 1))
        pos = pos + step if s[base + pos + step - 1] <= row else pos
    pos = pos + 1 if s[base + pos] <= row else pos
    return pos


@cute.jit
def flashmask_warp_not_monotone(s: cute.Tensor, base: Int32, n: cutlass.Constexpr[int]) -> Int32:
    # Nonzero if s[base + c] decreases anywhere over c in [0, n). Each lane checks the pairs
    # (c, c + 1) for c = lane, lane + 32, ...; the last pair compares s[n - 1] with itself.
    lane = cute.arch.lane_idx()
    bad = Int32(0)
    for k in cutlass.range_constexpr(n // 32):
        c = lane + k * 32
        nxt = cutlass.min(c + 1, n - 1)
        bad = bad + 1 if s[base + c] > s[base + nxt] else bad
    return bad


@cute.jit
def mask_r2p_ranges(
    X: cute.Tensor,
    lo1: Int32,
    hi1: Int32,
    lo2: Int32,
    hi2: Int32,
    two_ranges: cutlass.Constexpr[bool],
    lo1_open: cutlass.Constexpr[bool],
    hi2_open: cutlass.Constexpr[bool],
) -> None:
    # Same register-index layout as mask_r2p (tScS_t2r[i][1] == i): set X[c] = -inf for c in
    # [lo1, hi1) or [lo2, hi2). An empty range (lo >= hi) masks nothing. lo1_open means
    # lo1 = 0 and hi2_open means hi2 = ncol; those bounds are not built (the values passed
    # for them are ignored).
    ncol = const_expr(cute.size(X.shape))
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        # Shifts >= 32 give 0 (shl clamps), so (1 << n) - 1 is all ones past the chunk.
        mask = (1 << max(hi1 - s * 24, 0)) - 1
        if const_expr(not lo1_open):
            mask = mask & (((1 << max(lo1 - s * 24, 0)) - 1) ^ -1)
        if const_expr(two_ranges):
            mask2 = ((1 << max(lo2 - s * 24, 0)) - 1) ^ -1
            if const_expr(not hi2_open):
                mask2 = mask2 & ((1 << max(hi2 - s * 24, 0)) - 1)
            mask = mask | mask2
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            masked = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            X[c] = -Float32.inf if masked else X[c]


@cute.jit
def mask_r2p_transposed(
    X: cute.Tensor, row_limit_top: Int32, num_rep: int, row_group_stride=None
) -> None:
    # Bit manipulation, compiles down to the R2P instruction
    # Element c of the fragment sits at relative row
    #     (c // num_rep) * row_group_stride + c % num_rep
    # For sm100 that is e.g. 0, 1, ..., 15, 32, ..., 47, 64, ... (num_rep 16,
    # stride 32) or 0, 1, ..., 31, 64, ..., 95 (num_rep 32, stride 64).
    # The stride is NOT always num_rep * num_wg: with a folded accumulator
    # (2CTA d256/dv256, tile_n=64) the t2r stages advance the row by num_rep, so
    # the fragment rows are 0..63 contiguous. Pass the stride measured off the
    # coordinate tensor rather than assuming it.
    # We compare a transformed version of limit to 0, 1, 2, 3, 4, 5, ...
    if row_group_stride is None:
        row_group_stride = num_rep * 2  # two warp groups
    row_limit_top_transformed = row_limit_top // row_group_stride * num_rep + min(
        row_limit_top % row_group_stride, num_rep
    )
    ncol = cute.size(X.shape)
    # Ideally we'd move by 32 instead of 24, but mask >> i isn't correct for i == 31
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, 24)):
        row_limit_top_s = max(row_limit_top_transformed - s * 24, 0)
        # 0 -> 0b00...00, 1 -> 0b00...01, ..., 31 -> 0b01...11, 32 -> 0b11...11
        mask = (1 << row_limit_top_s) - 1
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(24, ncol - s * 24)):
            out_bound = cutlass.Boolean(mask & (1 << i))
            c = s * 24 + i
            X[c] = -Float32.inf if out_bound else X[c]


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_q: Int32
    seqlen_k: Int32
    window_size_left: Optional[Int32] = None
    window_size_right: Optional[Int32] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
    swap_AB: cutlass.Constexpr[bool] = False

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
    ) -> None:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_S_mn = layout_utils.make_acc_tensor_mn_view(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS), transpose=self.swap_AB)
        # We use t0ScS as these indices are known at compile time. We then must subtract the
        # column limit by the thread column offset.
        t0ScS_mn = layout_utils.make_acc_tensor_mn_view(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                # The compiler now choses not to use R2P
                r2p = const_expr(False and not self.swap_AB)
                if const_expr(not r2p):
                    # traverse column index.
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
                else:
                    mask_r2p(acc_S_mn, seqlenk_col_limit, arch=90)

        else:  # Causal or local
            if const_expr(not self.swap_AB):
                # If PackGQA, we split the work of compute divmod among threads in the same row
                threads_per_row = thr_mma.tv_layout_C.shape[0][0]
                mma_m_idx = None
                if const_expr(self.qhead_per_kvhead_packgqa != 1):
                    assert not self.swap_AB, "swap_AB with PackGQA not supported yet"
                    assert cute.arch.WARP_SIZE % threads_per_row == 0, (
                        "threads_per_row must divide WARP_SIZE"
                    )
                    assert cute.size(acc_S_mn.shape[0]) <= threads_per_row
                    tidx = thr_mma.thr_idx
                    mma_m_idx = (
                        m_block * self.tile_m + tScS_mn[tidx % threads_per_row, 0][0]
                    ) // self.qhead_per_kvhead_packgqa
                causal_row_offset = (
                    1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q - thr_col_offset
                )
                if const_expr(mask_causal):
                    # R2P (mask_r2p) assumes a specific per-thread column layout
                    # (0,1,8,9,16,17,...) via col_limit//8*2+min(col_limit%8,2). That
                    # holds for some MMA configs but NOT e.g. the head_dim=256 bwd
                    # (swap_AB=False, 64x64 tile), where it mis-masked near the causal
                    # diagonal -> residual dQ/dK/dV error. Use the layout-agnostic
                    # direct loop (with the real column indices) instead. (The
                    # seqlen-only path above already disables R2P for the same reason.)
                    r2p = const_expr(False)
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        # get the column index limit based on current row. Only consider the row index, so the column index sets to 0.
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        col_limit_right = row_idx + causal_row_offset
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        if const_expr(not r2p):
                            # traverse column index.
                            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                                acc_S_mn[r, c] = (
                                    -Float32.inf
                                    if t0ScS_mn[0, c][1] >= col_limit_right
                                    else acc_S_mn[r, c]
                                )
                        else:
                            mask_r2p(acc_S_mn[r, None], col_limit_right, arch=90, rank1=True)
                else:  # Local
                    local_row_offset_right = (
                        causal_row_offset + self.window_size_right
                        if const_expr(self.window_size_right is not None)
                        else None
                    )
                    local_row_offset_left = (
                        causal_row_offset - 1 - self.window_size_left
                        if const_expr(self.window_size_left is not None)
                        else None
                    )
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        if const_expr(self.qhead_per_kvhead_packgqa == 1):
                            row_idx = tScS_mn[r, 0][0] + m_block * self.tile_m
                        else:
                            row_idx = utils.shuffle_sync(
                                mma_m_idx, r % threads_per_row, width=threads_per_row
                            )
                        if const_expr(self.window_size_right is not None):
                            col_limit_right = row_idx + local_row_offset_right
                        else:
                            col_limit_right = self.tile_n
                        if const_expr(mask_seqlen):
                            col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                        col_limit_left = (
                            row_idx + local_row_offset_left
                            if const_expr(self.window_size_left is not None)
                            else 0
                        )
                        # traverse column index.
                        for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                            col_idx = t0ScS_mn[0, c][1]
                            # only consider the column index, so the row index sets to 0.
                            if col_idx >= col_limit_right or col_idx < col_limit_left:
                                acc_S_mn[r, c] = -Float32.inf
            else:  # swap_AB
                assert self.qhead_per_kvhead_packgqa == 1
                thr_row_offset = tScS_mn[0][ROW]
                causal_row_offset = (
                    seqlenk_col_limit - self.seqlen_q + m_block * self.tile_m + thr_row_offset
                )
                if const_expr(mask_causal):
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit and mask_seqlen
                            else col0 - causal_row_offset
                        )
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if t0ScS_mn[r, 0][ROW] < row_limit_top
                                else acc_S_mn[r, c]
                            )
                else:
                    for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                        col0 = t0ScS_mn[0, c][COL]
                        # If col0 is beyond the column limit, we want to mask out the entire
                        # column, by setting row limit to be self.tile_m.
                        row_limit_top = (
                            self.tile_m
                            if col0 >= seqlenk_col_limit
                            else col0 - causal_row_offset - self.window_size_right
                        )
                        # TODO: do we need col_limit_sink?
                        row_limit_bot = col0 - causal_row_offset + self.window_size_left
                        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                            row_idx = t0ScS_mn[r, 0][ROW]
                            acc_S_mn[r, c] = (
                                -Float32.inf
                                if row_idx < row_limit_top or row_idx > row_limit_bot
                                else acc_S_mn[r, c]
                            )

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        mask_causal: cutlass.Constexpr[bool],
        enable_flashmask: cutlass.Constexpr[bool],
        mask_local: cutlass.Constexpr[bool] = False,
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        s_startend_row_indices: Optional[cute.Tensor] = None,
        has_lt_end: cutlass.Constexpr[bool] = False,
        has_ut_start: cutlass.Constexpr[bool] = False,
        has_ut_end: cutlass.Constexpr[bool] = False,
        mbar_ptr: Optional[cute.Pointer] = None,
        mbar_load_startend_row_indices_empty_offset: Int32 = None,
        mbar_load_startend_row_indices_full_offset: Int32 = None,
        kv_stage:Int32 = None,
        stage: Int32 = None,
        load_startend_row_indices_consumer_state: Optional[cutlass.pipeline.PipelineState] = None,
        n_block_idx: Int32 = None,
        encode_n_block: Int32 = None,
        generate_block_buffer_usable_block_count: Int32 = None,
        use_r2p: cutlass.Constexpr[bool] = True,
        causal_active: Optional[cutlass.Boolean] = None,
    ) -> Optional[cutlass.pipeline.PipelineState]:
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        # To handle edge cases of completely masked out rows where n_block_max = 0,
        # we treat negative n_blocks as 0th n_block
        # TODO: find more transparent solution
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        # mask_r2p masks by REGISTER INDEX: it relies on tScS_t2r[i][1] == i (see its comment).
        # That holds only when a thread owns a whole row of the accumulator starting at column
        # 0. With a folded accumulator (m_block_size == 64 per CTA, N split across the TMEM
        # lane halves) thread t + 64 holds columns tile_n/2 .. tile_n-1, so its register index i
        # is tile_n/2 short of the real column and NOTHING gets masked -- the out-of-range keys
        # then feed softmax (observed: row_sum exactly 2x the true value). Callers on such a
        # config pass use_r2p=False to take the coordinate-based loop instead.
        r2p = const_expr(use_r2p)
        if const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if const_expr(causal_active is not None):
                    # The caller gates the seqlen_k pass to the blocks that can need it.
                    if causal_active:
                        mask_r2p(acc_S, seqlenk_col_limit, arch=100, rank1=True)
                elif const_expr(not r2p):
                    for i in cutlass.range(cute.size(tScS_t2r.shape), unroll_full=True):
                        # if tScS_t2r[i][1] >= seqlenk_col_limit:
                        #     acc_S[i] = -Float32.inf
                        # For some reason the 2 lines above generate really bad SASS
                        acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= seqlenk_col_limit else acc_S[i]
                else:
                    mask_r2p(acc_S, seqlenk_col_limit, arch=100, rank1=True)

        else:  # Causal or local
            causal_row_offset = 1 + self.seqlen_k - n_block * self.tile_n - self.seqlen_q
            row_idx = tScS_t2r[0][0] + m_block * self.tile_m
            if const_expr(self.qhead_per_kvhead_packgqa != 1):
                row_idx = row_idx // self.qhead_per_kvhead_packgqa
            if const_expr(mask_causal):
                col_limit_right = row_idx + causal_row_offset
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                ncol = const_expr(cute.size(tScS_t2r.shape))
                if const_expr(causal_active is None):
                    if const_expr(not r2p):
                        for i in cutlass.range(ncol, unroll_full=True):
                            acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                    else:
                        mask_r2p(acc_S, col_limit_right, arch=100, rank1=True)
                elif causal_active:
                    # fm_one_softmax: the caller runs one softmax_step instance for every block and only
                    # the causal diagonal / seqlen_k edge blocks need this pass.
                    if const_expr(not r2p):
                        for i in cutlass.range(ncol, unroll_full=True):
                            acc_S[i] = -Float32.inf if tScS_t2r[i][1] >= col_limit_right else acc_S[i]
                    else:
                        mask_r2p(acc_S, col_limit_right, arch=100, rank1=True)
            else:
                local_row_offset_right = (
                    causal_row_offset + self.window_size_right
                    if const_expr(self.window_size_right is not None)
                    else None
                )
                local_row_offset_left = (
                    causal_row_offset - 1 - self.window_size_left
                    if const_expr(self.window_size_left is not None)
                    else None
                )
                if const_expr(self.window_size_right is not None):
                    col_limit_right = row_idx + local_row_offset_right
                else:
                    col_limit_right = self.tile_n
                if const_expr(mask_seqlen):
                    col_limit_right = cutlass.min(col_limit_right, seqlenk_col_limit)
                col_limit_left = (
                    row_idx + local_row_offset_left
                    if const_expr(self.window_size_left is not None)
                    else 0
                )
                for i in cutlass.range_constexpr(cute.size(tScS_t2r.shape)):
                    col_idx = tScS_t2r[i][1]
                    acc_S[i] = (
                        -Float32.inf
                        if col_idx >= col_limit_right or col_idx < col_limit_left
                        else acc_S[i]
                    )

        if const_expr(enable_flashmask):

            # compute mbar_ptr in softmax_loop
            cute.arch.mbarrier_wait(
                mbar_ptr + mbar_load_startend_row_indices_full_offset + stage * kv_stage + load_startend_row_indices_consumer_state.index,
                load_startend_row_indices_consumer_state.phase)

            if n_block_idx < generate_block_buffer_usable_block_count and encode_n_block >= 0:
                # range_constexpr, not range(..., unroll_full=True): the latter emits a
                # real loop with a dynamic induction variable, and a dynamically indexed
                # store into acc_S keeps its alloca out of registers -- the whole S
                # fragment then lives in local memory and every access in softmax becomes
                # an ld.local/st.local (measured: ~16KB of local traffic per KV tile).
                nelem = const_expr(cute.size(tScS_t2r.shape))
                # Compare in global row space: row + m_block * tile_m against the raw
                # bounds, instead of rebasing every loaded LTS/LTE/UTS/UTE by
                # m_block * tile_m. Same integer inequality rearranged
                # (row >= lts - off <=> row + off >= lts; Int32, no overflow for
                # seqlen < 2^31), so the mask is bit-identical for every layout
                # (any head dim, q_stage, 2-CTA, folded). Each element keeps its own row
                # coordinate; where a thread owns one full S row (M=128 non-folded) those
                # are all equal and the add folds to one per thread.
                row_offset = m_block * self.tile_m
                s_base = load_startend_row_indices_consumer_state.index * 4 * self.tile_n
                # Where a thread owns one whole S row r (the R2P layout) and every bound is
                # non-decreasing over the block's columns, {c: LTS[c] <= r} is a prefix
                # [0, a) and {c: LTE[c] > r} a suffix [b, N), so the masked columns are the
                # range [b, a) (UTS/UTE likewise). Binary-search the range ends in smem and
                # mask with R2P instead of comparing every element; bit-identical result.
                # Blocks whose bounds are not monotone take the per-element loop.
                # Only with 2+ bounds: LTS alone is already one compare + select per element;
                # there the monotone check only adds latency (B300 A/B: SW -2.6%, Share
                # Question -5.6%, Causal Doc -1.9%), while GSW (4 bounds) gained 11%.
                tn = const_expr(self.tile_n)
                multi_bound = const_expr(any((has_ut_start, has_lt_end, has_ut_end)))
                use_ranges = const_expr(
                    all((
                        r2p,
                        nelem == tn,
                        tn % 32 == 0,
                        tn & (tn - 1) == 0,
                        multi_bound,
                    ))
                )
                if const_expr(use_ranges):
                    s = s_startend_row_indices
                    bad = flashmask_warp_not_monotone(s, s_base, tn)
                    if const_expr(has_ut_start):
                        bad = bad + flashmask_warp_not_monotone(s, s_base + tn, tn)
                        bad = bad + flashmask_warp_not_monotone(s, s_base + tn * 2, tn)
                        bad = bad + flashmask_warp_not_monotone(s, s_base + tn * 3, tn)
                    elif const_expr(has_lt_end):
                        bad = bad + flashmask_warp_not_monotone(s, s_base + tn, tn)
                    elif const_expr(has_ut_end):
                        bad = bad + flashmask_warp_not_monotone(s, s_base + tn * 3, tn)
                    if cute.arch.vote_any_sync(bad != 0):
                        self._apply_flashmask_elementwise(
                            acc_S, tScS_t2r, s, s_base, row_offset,
                            has_lt_end, has_ut_start, has_ut_end,
                        )
                    else:
                        row_g = tScS_t2r[0][0] + row_offset
                        a = flashmask_count_le(s, s_base, tn, row_g)
                        if const_expr(has_ut_start):
                            b = flashmask_count_le(s, s_base + tn, tn, row_g)
                            a2 = flashmask_count_le(s, s_base + tn * 2, tn, row_g)
                            b2 = flashmask_count_le(s, s_base + tn * 3, tn, row_g)
                            mask_r2p_ranges(acc_S, b, a, b2, a2, True, False, False)
                        elif const_expr(has_lt_end):
                            b = flashmask_count_le(s, s_base + tn, tn, row_g)
                            mask_r2p_ranges(acc_S, b, a, b, a, False, False, False)
                        elif const_expr(has_ut_end):
                            b2 = flashmask_count_le(s, s_base + tn * 3, tn, row_g)
                            mask_r2p_ranges(acc_S, a, a, b2, b2, True, True, True)
                        else:
                            mask_r2p_ranges(acc_S, a, a, a, a, False, True, False)
                else:
                    self._apply_flashmask_elementwise(
                        acc_S, tScS_t2r, s_startend_row_indices, s_base, row_offset,
                        has_lt_end, has_ut_start, has_ut_end,
                    )

            cute.arch.mbarrier_arrive(
                mbar_ptr + mbar_load_startend_row_indices_empty_offset + stage * kv_stage + load_startend_row_indices_consumer_state.index)
            load_startend_row_indices_consumer_state.advance()
        return load_startend_row_indices_consumer_state

    @cute.jit
    def _apply_flashmask_elementwise(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        s_startend_row_indices: cute.Tensor,
        s_base: Int32,
        row_offset: Int32,
        has_lt_end: cutlass.Constexpr[bool],
        has_ut_start: cutlass.Constexpr[bool],
        has_ut_end: cutlass.Constexpr[bool],
    ) -> None:
        nelem = const_expr(cute.size(tScS_t2r.shape))
        if const_expr(has_ut_start):
            for i in cutlass.range_constexpr(nelem):
                row_g = tScS_t2r[i][0] + row_offset
                col = tScS_t2r[i][1]
                lts = s_startend_row_indices[s_base + col]
                lte = s_startend_row_indices[s_base + col + self.tile_n]
                uts = s_startend_row_indices[s_base + col + self.tile_n * 2]
                ute = s_startend_row_indices[s_base + col + self.tile_n * 3]
                if (row_g >= lts and row_g < lte) or (row_g >= uts and row_g < ute):
                    acc_S[i] = -cutlass.Float32.inf
        elif const_expr(has_lt_end):
            for i in cutlass.range_constexpr(nelem):
                row_g = tScS_t2r[i][0] + row_offset
                col = tScS_t2r[i][1]
                lts = s_startend_row_indices[s_base + col]
                lte = s_startend_row_indices[s_base + col + self.tile_n]
                if row_g >= lts and row_g < lte:
                    acc_S[i] = -cutlass.Float32.inf
        elif const_expr(has_ut_end):
            for i in cutlass.range_constexpr(nelem):
                row_g = tScS_t2r[i][0] + row_offset
                col = tScS_t2r[i][1]
                lts = s_startend_row_indices[s_base + col]
                ute = s_startend_row_indices[s_base + col + self.tile_n * 3]
                if row_g >= lts or row_g < ute:
                    acc_S[i] = -cutlass.Float32.inf
        else:
            for i in cutlass.range_constexpr(nelem):
                row_g = tScS_t2r[i][0] + row_offset
                lts = s_startend_row_indices[s_base + tScS_t2r[i][1]]
                if row_g >= lts:
                    acc_S[i] = -cutlass.Float32.inf

    @cute.jit
    def apply_flashmask_sm90(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        thr_mma: cute.TiledMma,
        s_startend_row_indices: cute.Tensor,
        has_lt_end: cutlass.Constexpr[bool] = False,
        has_ut_start: cutlass.Constexpr[bool] = False,
        has_ut_end: cutlass.Constexpr[bool] = False,
    ) -> None:
        """SM90 flashmask (startend_row_indices) application on S.

        s_startend_row_indices is the current pipeline stage's flat smem buffer
        holding up to 4 vectors of tile_n Int32 at offsets
        [0, tile_n, 2*tile_n, 3*tile_n] = [LTS, LTE, UTS, UTE].

        Mirrors apply_mask_sm100's flashmask branch but operates on the SM90
        (row, col) mn-view of the accumulator. The masked rows for a given kv
        column are described relative to the query tile via `- m_block*tile_m`.
        """
        acc_S_mn = layout_utils.make_acc_tensor_mn_view(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.make_acc_tensor_mn_view(
            thr_mma.partition_C(cS), transpose=self.swap_AB
        )
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        nrow = const_expr(cute.size(tScS_mn.shape[0]))
        ncol = const_expr(cute.size(tScS_mn.shape[1]))
        tile_n = const_expr(self.tile_n)
        # In the mn view a thread's column coordinate depends only on c and its
        # row coordinate only on r, so hoist the (up to 4) smem index loads and
        # the m_block*tile_m rebase out of the row loop: one set of loads per
        # column instead of per element.
        m_offset = m_block * self.tile_m
        for c in cutlass.range(ncol, unroll_full=True):
            col = tScS_mn[0, c][COL]
            lts = s_startend_row_indices[col] - m_offset
            lte = Int32(0)
            uts = Int32(0)
            ute = Int32(0)
            if const_expr(has_ut_start):
                lte = s_startend_row_indices[tile_n + col] - m_offset
                uts = s_startend_row_indices[2 * tile_n + col] - m_offset
                ute = s_startend_row_indices[3 * tile_n + col] - m_offset
            elif const_expr(has_lt_end):
                lte = s_startend_row_indices[tile_n + col] - m_offset
            elif const_expr(has_ut_end):
                ute = s_startend_row_indices[3 * tile_n + col] - m_offset
            for r in cutlass.range(nrow, unroll_full=True):
                row = tScS_mn[r, c][ROW]
                if const_expr(has_ut_start):
                    if (row >= lts and row < lte) or (row >= uts and row < ute):
                        acc_S_mn[r, c] = -Float32.inf
                elif const_expr(has_lt_end):
                    if row >= lts and row < lte:
                        acc_S_mn[r, c] = -Float32.inf
                elif const_expr(has_ut_end):
                    if row >= lts or row < ute:
                        acc_S_mn[r, c] = -Float32.inf
                else:
                    if row >= lts:
                        acc_S_mn[r, c] = -Float32.inf

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        mask_local: cutlass.Constexpr,
        sStartEndRowIndices: cute.Tensor,
        # Python bool (compile-time branch) or cutlass.Boolean (runtime branch around
        # the unrolled element loop below, used by the 2CTA bwd flat loop). Left
        # unannotated on purpose: a `bool` annotation makes the DSL treat a dynamic
        # value as constexpr.
        partially_masked,
        per_cta_tile_n: cutlass.Constexpr[int] = 0,
        # True when startend_row_indices.shape[-1] == 4, i.e. sStartEndRowIndices holds
        # 4 columns [LTS, LTE, UTS, UTE] instead of 2.
        has_ut_start: cutlass.Constexpr[bool] = False,
    ) -> None:
        """
        Backward pass: mask S = K @ Q.T where n_block tiles seqlen_k and m_block tiles seqlen_q.
        """
        assert not (mask_causal and mask_local), "mask_causal and mask_local cannot be both True"
        ROW = 0 if const_expr(not self.swap_AB) else 1
        COL = 1 if const_expr(not self.swap_AB) else 0
        # assert t0ScS_t2r[0][COL] == 0, "col0 == 0" # tmp comment for 2-cta bwd
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(not mask_causal and not mask_local):
            if const_expr(mask_seqlen):
                if seqlenk_col_limit <= 0:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = -cutlass.Float32.inf
            # FlashMask
            if partially_masked:
                # In 2CTA mode, COL coordinates span cta_group_size * tile_n (e.g. 256),
                # but sStartEndRowIndices has per-CTA tile_n entries (e.g. 128).
                # Convert global COL to per-CTA local coordinate.
                _fm_tile_n = per_cta_tile_n if const_expr(per_cta_tile_n > 0) else self.tile_n
                if const_expr(has_ut_start):
                    # 4 vectors: mask out the union of the lower band [LTS, LTE) and
                    # the upper band [UTS, UTE). Same semantics as apply_flashmask_sm90.
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        col_local = tScS_t2r[i][COL] % _fm_tile_n
                        m_offset = m_block * self.tile_m
                        lts = sStartEndRowIndices[col_local, 0] - m_offset
                        lte = sStartEndRowIndices[col_local, 1] - m_offset
                        uts = sStartEndRowIndices[col_local, 2] - m_offset
                        ute = sStartEndRowIndices[col_local, 3] - m_offset
                        row = tScS_t2r[i][ROW]
                        acc_S[i] = (
                            -cutlass.Float32.inf
                            if (row >= lts and row < lte) or (row >= uts and row < ute)
                            else acc_S[i]
                        )
                else:
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        col_local = tScS_t2r[i][COL] % _fm_tile_n
                        lts = sStartEndRowIndices[col_local, 0] - m_block * self.tile_m
                        ute = sStartEndRowIndices[col_local, 1] - m_block * self.tile_m
                        acc_S[i] = (
                            -cutlass.Float32.inf if tScS_t2r[i][ROW] >= lts else acc_S[i]
                        )
                        acc_S[i] = (
                            -cutlass.Float32.inf if tScS_t2r[i][ROW] < ute else acc_S[i]
                        )

        else:  # Causal or local
            thr_row_offset = tScS_t2r[0][ROW]
            seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
            causal_offset = seqlenq_row_limit - seqlenk_col_limit
            if const_expr(mask_causal):
                row_limit_top = causal_offset
                if const_expr(mask_seqlen):
                    # If col is beyond the column limit, we want to mask out the entire
                    # column, by setting row limit to be self.tile_m.
                    if seqlenk_col_limit <= 0:
                        row_limit_top = self.tile_m

                r2p = True
                if const_expr(not r2p):
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        acc_S[i] = (
                            -cutlass.Float32.inf if t0ScS_t2r[i][ROW] < row_limit_top else acc_S[i]
                        )
                else:
                    num_rep = cute.size(tScS_t2r, mode=[0])  # 16 or 32
                    # Row step between the fragment's t2r stages, straight from the
                    # coordinate tensor (see mask_r2p_transposed): num_rep * 2 for the
                    # unfolded accumulators, num_rep when the accumulator is folded.
                    if const_expr(cute.size(tScS_t2r) > num_rep):
                        row_group_stride = t0ScS_t2r[num_rep][ROW] - t0ScS_t2r[0][ROW]
                    else:
                        row_group_stride = num_rep
                    mask_r2p_transposed(acc_S, row_limit_top, num_rep, row_group_stride)

                if partially_masked:
                    # FlashMask
                    # In 2CTA mode, COL coordinates span cta_group_size * tile_n (e.g. 256),
                    # but sStartEndRowIndices has per-CTA tile_n entries (e.g. 128).
                    # Convert global COL to per-CTA local coordinate.
                    _fm_tile_n = per_cta_tile_n if const_expr(per_cta_tile_n > 0) else self.tile_n
                    for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                        col_local = tScS_t2r[i][COL] % _fm_tile_n
                        lts = sStartEndRowIndices[col_local, 0] - m_block * self.tile_m
                        lte = sStartEndRowIndices[col_local, 1] - m_block * self.tile_m
                        acc_S[i] = (
                            -cutlass.Float32.inf if tScS_t2r[i][ROW] >= lts and tScS_t2r[i][ROW] < lte else acc_S[i]
                        )
            else:
                assert False, "Local masking isn't supported yet"
