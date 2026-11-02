# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""SM100 backward for head dims larger than 256 (supported and measured: 512/512).

Why a separate file from `flash_bwd_sm100.py`: that kernel hand-unrolls the
low|high halves of its split axes in ~8 places in the load warp, and hardcodes
`tile_m == tile_n == 128`. Two halves is not enough once head_dim exceeds 512
(576/2 = 288 exceeds the UMMA N limit of 256, so that shape needs 3 chunks), so
every one of those sites would have to become a loop, which puts the
already-working d256/d192 configs at risk. Here the chunk count and `cta_group`
are parameters from the start, so the axis splits into as many chunks as the
shape needs.

TMEM resource model (verified on B200, encoded in tmem_plan() / sm100_utils.tmem_cols):
an accumulator costs N * (rows_per_cta / 128) columns, so cta_group=2 with M=128 halves
the column cost. The one irregular case is cta_group=1 with M=64, which takes the
16-datapath lane interleave and still spans N columns.

Phase 1 (the only path that runs today): cta_group=1, no swapAB, all three of dV/dK/dQ
are produced per m-iteration and drained by the reduce warps into the fp32 gmem
accumulators. Correct but accum-traffic bound -- the drain is roughly 40% of the kernel,
split about evenly between its dQ and dKV halves (measurements in __init__, where the
drain cost model and the tile-size dependence are derived).

Operand reuse: every tile is read by two gemms in different orientations -- K by S and
dQ, Q by S and dK, dO by dP and dV -- and this used to re-TMA the tile for the second
one. The two orientations are the same bytes (derivation in _setup_smem_layout), so the
second gemm now reads whatever the first left in SMEM and _reuse_schedule decides per
access whether a fetch is needed at all, cutting the per-m-tile fetch count by about a
fifth at 512/512. The saving is TMA issue and mbarrier round trips on the load warp, not
HBM bytes: an n-block's K is re-read by all 64 query heads, so L2 was already serving
those fetches.

Phase 2 (later): cta_group=2 + swapAB on dV/dK so both stay resident in TMEM for
the whole m loop, dropping the accum traffic from once per m-iteration to once
per n-block. The cta_group=2 arithmetic is already threaded through (column rules,
n_block_pair skip derivation, MMA M constraints) but the path is NOT enabled and has
never run: `_launch` passes no `cluster=` to .launch() and does not round the grid to the
pair size, the kernel body still slices every MMA at `get_slice(0)` instead of the CTA's
own v coordinate, and gK / gV are tiled by their own n index rather than the PAIR's.

It was costed out and deliberately not taken. The short version: the TMEM column rules
leave only two legal (tile_m, tile_n) pairs that save anything, each halves ONE of the
two drain terms and leaves the other flat, and the better of the two is worth well under
a third of the drain. That variant is additionally blocked on the dQ gemm's A operand --
at cta_group=2 its K spans the pair's whole key range while the accumulator's M folds, so
each CTA needs dS for its own m half over BOTH CTAs' keys, and it only computed dS for
all m over its own keys. flash_bwd_sm100.py solves this with separate_sdS_buffers /
smem_dS_dq_block_view, but that machinery asserts tile_n <= mma_k (64), which the config
in question violates, so there is no working precedent in this repo to copy. A cluster of
(2,1) is also entirely consumed by the MMA's own v mode, so Q / dO^T are NOT multicast
(see flash_bwd_sm100.py:1005) and there is no operand-traffic win to add on top. See the
git history of this docstring for the per-config numbers.
"""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.utils.layout import LayoutEnum
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05

from flashmask.cute import blackwell_helpers as sm100_utils
from flashmask.cute.blackwell_helpers import SM100_SMEM_CAPACITY_BYTES
from flashmask.cute import barrier, copy_utils, layout_utils, utils
from flashmask.cute.flash_bwd_sm100 import _overlap_gate_bwd
from flashmask.cute.tile_scheduler import SingleTileScheduler, TileSchedulerArguments


SM100_TMEM_CAPACITY_COLUMNS = 512
# Slack for barrier storage, LSE / dPsum, the flashmask row indices and the
# per-buffer alignment padding that cute.struct adds on top of the tile bytes.
# The real struct runs a few KB over the solver's tile bytes, so this is ~2x headroom.
SM100_SMEM_RESERVE_BYTES = 6 * 1024
# UMMA N limit (cutlass/cute/nvgpu/tcgen05/mma.py).
UMMA_MAX_N = 256
# All accumulators that take part in a "T2R then packed-bf16 R2T" round trip must
# have M = 128 so the 32-datapath atoms apply: at M=64 only the 16-datapath atoms
# exist and their per-thread element order does not survive the bf16 packing
# (measured on B200: exact for a constant A operand, ~7% off for A = m or A = n).
UMMA_REQUIRED_M = 128
# Slots used to reduce flashmask bounds once per key pair. 0-1 give the m skip range's
# lower-tail band (max of the starts, min of the ends), 2-3 its upper-tail band for the
# 4-bound form; 4-7 give the unmasked-block test (min / max of each bound column).
FLASHMASK_META_SLOTS = 8


MAX_OUT_SLOTS = 2


def tmem_plan(tile_m: int, tile_n: int, d_chunk: int, dv_chunk: int, cta_group: int):
    """Return the TMEM column plan for one configuration.

    Args:
        tile_m: Query tile height.
        tile_n: Key tile width.
        d_chunk: GEMM chunk width along head_dim.
        dv_chunk: GEMM chunk width along head_dim_v.
        cta_group: Number of CTAs participating in each MMA.

    Returns:
        Dictionary with the S / dP offsets, the output scratch slot width, the number
        of output slots and the total per-CTA column count.

    The single source of truth for the layout: ``solve_config`` uses it to decide
    feasibility and ``__init__`` uses it to place the regions. They used to carry two
    hand-written copies of the formula that disagreed on both terms that matter -- the
    solver charged the dQ accumulator ``tile_m`` columns where the layout charges it
    ``d_chunk``, and it did not account for the output slot count at all, so it could
    hand back a config the constructor then rejected.
    """
    s_cols = sm100_utils.tmem_cols(tile_m, cta_group * tile_n, cta_group)
    dp_cols = sm100_utils.tmem_cols(tile_m, cta_group * tile_n, cta_group)
    dv_cols = sm100_utils.tmem_cols(dv_chunk, cta_group * tile_n, cta_group)
    dk_cols = sm100_utils.tmem_cols(d_chunk, cta_group * tile_n, cta_group)
    dq_cols = sm100_utils.tmem_cols(d_chunk, tile_m, cta_group)
    out_offset = s_cols + dp_cols
    slot_cols = max(dv_cols, dk_cols, dq_cols)
    num_out_slots = min(
        MAX_OUT_SLOTS, (SM100_TMEM_CAPACITY_COLUMNS - out_offset) // slot_cols
    )
    return dict(
        s_cols=s_cols,
        out_offset=out_offset,
        slot_cols=slot_cols,
        num_out_slots=num_out_slots,
        total=out_offset + max(num_out_slots, 1) * slot_cols,
    )



def accum_slice_candidates(hdim: int, chunk: int, max_slice: int = 192) -> list:
    """Return the legal accumulator slice widths for an axis, narrowest first.

    Args:
        hdim: Padded head dimension represented by the accumulator.
        chunk: GEMM chunk width; a slice must contain whole chunks.
        max_slice: Preferred maximum width accepted by postprocessing.

    Returns:
        Ascending list of legal slice widths, empty if the chunk admits none.

    The fp32 accumulators are blocked as [slice][row block][...] so that each slice
    is byte-identical to a `head_dim = slice` accumulator and can be handed to the
    shared FlashAttentionBackwardPostprocess as-is. That kernel stages a whole
    tile_m x head_dim fp32 tile in SMEM and holds it in registers, hence the cap.

    A slice must be a whole number of gemm chunks (so a chunk never straddles two
    slices) and a multiple of 64 (the postprocess rounds head_dim up to 64 and would
    otherwise read past the slice). A chunk wider than the cap raises it to the chunk
    width -- the existing d256 path already runs that postprocess at head_dim 256.
    """
    max_slice = max(max_slice, chunk)
    step = chunk * 64 // math.gcd(chunk, 64)
    return [w for w in range(step, min(hdim, max_slice) + 1, step) if hdim % w == 0]


def accum_slice_width(hdim: int, chunk: int, max_slice: int = 192) -> int:
    """Return the widest legal accumulator slice width for an axis.

    Args:
        hdim: Padded head dimension represented by the accumulator.
        chunk: GEMM chunk width; a slice must contain whole chunks.
        max_slice: Preferred maximum width accepted by postprocessing.

    Returns:
        Largest legal slice width not exceeding the effective maximum.
    """
    widths = accum_slice_candidates(hdim, chunk, max_slice)
    assert widths, (
        f"no legal accumulator slice for hdim={hdim}, chunk={chunk} "
        f"(nothing that is a whole number of chunks, a multiple of 64, divides "
        f"{hdim} and is <= {max(max_slice, chunk)})"
    )
    return widths[-1]


def _chunk_candidates(hdim: int) -> list:
    """Legal chunk widths for a headdim axis, widest first.

    <= UMMA_MAX_N because the chunk is an MMA N; a multiple of 32 because the
    reduce path tiles the fp32 workspace by dQ_reduce_ncol = 32; and a divisor of
    the (64-padded) headdim so the axis splits evenly.
    """
    return [
        c
        for c in range(min(UMMA_MAX_N, hdim), 0, -32)
        if hdim % c == 0
    ]


def _reuse_schedule(num_chunks: int, num_stages: int, carry: bool):
    """Decide, per SMEM access, whether an operand chunk still has to be fetched.

    Every tile is read by two gemms -- S and dQ read K, S and dK read Q, dP and dV read
    dO -- and the second one only wants the transposed view, which is the same bytes
    (see _setup_smem_layout), so an access whose buffer already holds the chunk it wants
    needs no TMA. Pass 0 runs ASCENDING and pass 1 DESCENDING so that the chunk pass 0
    leaves behind is the first one pass 1 asks for; with both ascending the hit count is
    zero at any stage count, so the order is a precondition, not a tweak.

    `carry` is whether buffer contents survive an m iteration: true for K (depends only
    on the n block), false for Q / dO (the bytes survive but belong to the last m tile).

    Returns (pass0, pass1, prologue): per-access records in issue order, plus the
    (chunk, stage) pairs that must already be in SMEM when the m loop starts. A record's
    `after` is the access that last used the same buffer, whose *_in_empty barrier a
    fetch must wait on; `after_prev_iter` marks that arrival as belonging to the previous
    m iteration, which the caller skips on the first one (a parity-1 wait on a fresh
    mbarrier never falls through).
    """
    assert 1 <= num_stages <= num_chunks
    stage_of = tuple(c % num_stages for c in range(num_chunks))
    orders = (tuple(range(num_chunks)), tuple(reversed(range(num_chunks))))
    accesses = tuple((p, c) for p in (0, 1) for c in orders[p])
    n = len(accesses)
    records = []
    for i, (_, chunk) in enumerate(accesses):
        stage = stage_of[chunk]
        # Last access to the same buffer, walking backwards through the cyclic
        # sequence. prev == i means this is the buffer's only access in the cycle.
        prev = next(
            (i - k) % n
            for k in range(1, n + 1)
            if stage_of[accesses[(i - k) % n][1]] == stage
        )
        from_prev_iter = prev >= i
        hit = accesses[prev][1] == chunk and (carry or not from_prev_iter)
        records.append(
            dict(
                chunk=chunk,
                stage=stage,
                load=not hit,
                after=accesses[prev],
                after_prev_iter=from_prev_iter,
            )
        )
    passes = tuple(
        tuple(r for r, (p, _) in zip(records, accesses) if p == which)
        for which in (0, 1)
    )
    # A buffer whose first access in the cycle is a hit needs its content in place
    # before the m loop starts, and that content is whatever the cycle's LAST access to
    # that buffer left there.
    first_of_stage, last_chunk_of_stage = {}, {}
    for i, (_, chunk) in enumerate(accesses):
        first_of_stage.setdefault(stage_of[chunk], i)
        last_chunk_of_stage[stage_of[chunk]] = chunk
    prologue = tuple(
        (last_chunk_of_stage[s], s)
        for s in range(num_stages)
        if not records[first_of_stage[s]]["load"]
    )
    return passes[0], passes[1], prologue


def solve_config(
    head_dim: int,
    head_dim_v: int,
    *,
    cta_group: int = 1,
    dtype_bytes: int = 2,
    # tile_m == tile_n == 128 only: the output drain maps one drain thread to one
    # accumulator row and asserts it (a 64-row m tile would need a second mapping).
    # 64 cannot buy occupancy either: sK / sV / sdS all scale with tile_n, which the UMMA
    # M=128 rule pins, so the best case is ~163KB, still 1 CTA/SM.
    tile_m_choices: tuple = (128,),
    smem_budget: int = SM100_SMEM_CAPACITY_BYTES - SM100_SMEM_RESERVE_BYTES,
    reduce_ncol: int = 32,
):
    """Pick a feasible tile and chunk configuration for a head shape.

    Args:
        head_dim: Query/key head dimension before padding.
        head_dim_v: Value head dimension before padding.
        cta_group: Number of CTAs participating in each MMA.
        dtype_bytes: Bytes per input element used by the SMEM model.
        tile_m_choices: Candidate query tile heights.
        smem_budget: Maximum modeled shared-memory bytes per CTA.
        reduce_ncol: Column width of each output T2R / reduce slice.

    Returns:
        Dictionary containing the selected tile, chunk, and resource values.

    Feasibility is the verified resource model:

      TMEM (columns, per CTA)  -- see tmem_plan(), which is what actually decides
        S/P and dP/dS live across the whole m iteration; the three outputs time-share
        a scratch region of num_out_slots x the widest output chunk.

      SMEM (bytes)
        sQ  tile_m * d_chunk      sK  tile_n * d_chunk * d_chunks_resident
        sV  tile_n * dv_chunk     sdO tile_m * dv_chunk
        sdS tile_n * tile_m

      Accumulator slicing -- a chunk width that admits no legal slice (see
      accum_slice_candidates) is rejected here rather than left to blow up in the
      constructor.

    Ranking is MEASURED, not modelled. `flush/work` (accumulator traffic per unit of
    work) used to be the primary key; the kernel turned out to sit far below both HBM
    bandwidth and MMA peak, i.e. bound by latency rather than traffic, so
    that model does not describe the bottleneck. What the 512/512 sweep
    (b=16 s=4096 h=16) actually showed:

      dv_chunk == tile_n is a sweet spot, not "wider is better": narrowing dv below
        tile_n costs progressively more, and dv 256 (which forces d_chunk 32) is the
        worst config measured. dv_chunk is the dV gemm's N; away from tile_n it either
        adds MMA rounds or starves the output scratch.
      At equal SMEM, a wider d_chunk wins: d128 beat d64.

    Hence the key: |dv_chunk - tile_n|, then d_chunk, then flush/work, then the LOWER
    K residency.

    That last key looks backwards -- more resident K chunks ought to help the S and dQ
    gemms -- but it is what keeps the ranking on measured ground: the sweep that produced
    the numbers above ran at d_chunks_resident=2, and nothing has measured what the
    leftover SMEM is worth as extra residency. That leftover is real (the accumulator
    staging buffer this model used to charge for is gone), so residency is the obvious
    next knob to sweep; until then the solver does not spend it on a guess.
    """
    pad = lambda x: int(math.ceil(x / 64) * 64)
    d, dv = pad(head_dim), pad(head_dim_v)
    # Rule 2: the K-side gemms have M = cta_group * tile_n, and that must be 128.
    tile_n = UMMA_REQUIRED_M // cta_group
    solutions = []
    for tile_m in tile_m_choices:
        for d_chunk in _chunk_candidates(d):
            for dv_chunk in _chunk_candidates(dv):
                num_d_chunks = d // d_chunk
                if not accum_slice_candidates(d, d_chunk):
                    continue
                if not accum_slice_candidates(dv, dv_chunk):
                    continue
                for resident in range(min(2, num_d_chunks), num_d_chunks + 1):
                    plan = tmem_plan(tile_m, tile_n, d_chunk, dv_chunk, cta_group)
                    if plan["num_out_slots"] < 1:
                        continue
                    if plan["total"] > SM100_TMEM_CAPACITY_COLUMNS:
                        continue
                    smem = dtype_bytes * (
                        tile_m * d_chunk                      # sQ (aliases sQt)
                        + tile_n * d_chunk * resident         # sK (aliases sKt)
                        + tile_n * dv_chunk                   # sV
                        + tile_m * dv_chunk                   # sdO (aliases sdOt)
                        + tile_n * tile_m                     # sdS
                    )
                    if smem > smem_budget:
                        continue
                    flush_per_work = (
                        tile_n * (d + dv) + tile_m * d
                    ) / (tile_m * tile_n)
                    solutions.append(
                        dict(
                            tile_m=tile_m,
                            tile_n=tile_n,
                            d_chunk=d_chunk,
                            dv_chunk=dv_chunk,
                            d_chunks_resident=resident,
                            dQ_reduce_ncol=reduce_ncol,
                            cta_group=cta_group,
                            tmem_columns=plan["total"],
                            smem_bytes=smem,
                            flush_per_work=flush_per_work,
                            # Ranking key: measured, not modelled. Prefer dv_chunk ==
                            # tile_n, then the wider d_chunk.
                            _tie=(
                                abs(dv_chunk - tile_n),
                                -d_chunk,
                                flush_per_work,
                                resident,
                            ),
                        )
                    )
    if not solutions:
        raise ValueError(
            f"no feasible config for head_dim={head_dim}, head_dim_v={head_dim_v}, "
            f"cta_group={cta_group}"
        )
    solutions.sort(key=lambda s: s["_tie"])
    return solutions[0]


CONFIG_KEYS = (
    "tile_m", "tile_n", "d_chunk", "dv_chunk", "d_chunks_resident", "dQ_reduce_ncol",
)

# Measured pins, keyed by (head_dim, head_dim_v, cta_group). A pin's optional "kv_shared"
# block holds the overrides that mode needs, layered on top of the same shape's base pin.
# These win over the solver's ranking, which is a *model*: the measured kernel sits far
# below both HBM bandwidth and MMA peak, so it does not currently describe the bottleneck.
#
# 512/512 (B30Z / sm103, causal document mask): measured faster than the config the solver
# ranked first at the time. What the pin buys is a halved slice count in the drain, which is
# why dQ_reduce_ncol has to come with the wider d_chunk -- d_chunk is the tile the drain
# slices, so a 64-column slice is only legal once the chunk is at least 64 wide.
# 576/512 (same host and mask): the solver picks d_chunk 96, which is the worst width this
# axis can take -- gcd(96, dv_chunk 128) is 32, so the whole drain drops to 32-column slices
# (roughly double the count) and the axis splits into 6 chunks. 576 = 2**6 * 9, so of the
# legal widths (32/96/192/576) only 192 divides 64: it is the only way to get the 64-column
# slice back, and it measured substantially faster despite doing slightly more math.
# The price, and the thing this pin is measuring: a 192-wide output chunk leaves
# (512-256)/192 = 1 TMEM out slot instead of 2, so the output gemms no longer overlap the
# drain, and 2x192 K stages do not fit in SMEM so K is single-buffered.
#
# 576/512 kv_shared adds a 4th key element (kv_shared) because the mode changes what is
# feasible, not just what is fast: the merge needs dv_chunk == d_chunk == 192, and the
# drain's two staging slots then have to fit next to sQ/sK/sdO/sdS, which ncol 64 does not
# and ncol 32 does. The dv axis is padded 512 -> 576 for the tiling; see tile_hdimv in
# __init__.
_MEASURED_CONFIG = {
    (512, 512, 1): {
        "d_chunk": 128,
        "d_chunks_resident": 2,
        "dQ_reduce_ncol": 64,
        "kv_shared": {
            # Same widths the generic pin already uses; dv_chunk is spelled out because
            # the merge asserts d_chunk == dv_chunk and leaving it to the solver's
            # tie-break makes that equality a coincidence rather than a constraint.
            "dv_chunk": 128,
        },
    },
    (576, 512, 1): {
        "d_chunk": 192,
        "d_chunks_resident": 1,
        "dQ_reduce_ncol": 64,
        "kv_shared": {
            "dv_chunk": 192,
            "dQ_reduce_ncol": 32,
        },
    },
}


# --------------------------------------------------------------- drain helpers
# Module-level on purpose: the DSL rejects locally-defined closures that
# capture variables once the kernel enters dynamic control flow (the warp-role
# branches), so everything these need is passed in explicitly. `self` is the
# kernel object (compile-time configuration only).
@cute.jit
def _bigd_m_block(it, m_lo, n1, seg2_base, n2, seg3_base):
    """The m block of iteration `it`: the walk over [m_lo, m_hi) minus the two
    fully masked bands, as three consecutive segments (see the skip range).

    Every warp role -- load, mma, compute, drain -- calls this one function, so they
    all visit the same blocks in the same order; a second copy of the expression that
    drifts from this one would desynchronise the barrier phases."""
    m_block = seg3_base + (it - n1 - n2)
    m_block = seg2_base + (it - n1) if it < n1 + n2 else m_block
    m_block = m_lo + it if it < n1 else m_block
    return m_block


@cute.jit
def _bigd_make_out_drain(self, wg_tidx, tmem_ptr):
    """T2R machinery for draining the output scratch, built once per draining
    warpgroup (the copy is partitioned over that warpgroup's 128 threads).
    Used by the drain warps always, and by the compute warps under drain_split."""
    ncol = cutlass.const_expr(self.dQ_reduce_ncol)
    mma_out_slice = sm100_utils_basic.make_trivial_tiled_mma(
        self.q_dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        self.acc_dtype,
        self.cta_group,
        (self.mma_tiler_pdo[0], ncol),
    )
    thr_mma_out_slice = mma_out_slice.get_slice(0)
    out_slice_layout = thr_mma_out_slice.make_fragment_C(
        thr_mma_out_slice.partition_shape_C((self.mma_tiler_pdo[0], ncol))
    ).layout
    tmem_load_atom_out = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(ncol // 4)), Float32
    )
    thr_t2r_out = tcgen05.make_tmem_copy(
        tmem_load_atom_out,
        cute.make_tensor(tmem_ptr + self.tmem_out_offset, out_slice_layout),
    ).get_slice(wg_tidx)
    c_out_slice = thr_mma_out_slice.partition_C(
        cute.make_identity_tensor((self.mma_tiler_pdo[0], ncol))
    )
    shape_out_slice = thr_t2r_out.partition_D(c_out_slice).shape
    return thr_t2r_out, out_slice_layout, shape_out_slice

@cute.jit
def _bigd_drain_wg_iteration(
    self,
    phase,
    m_iter,
    wg_tidx,
    thr_t2r_out,
    out_slice_layout,
    shape_out_slice,
    slice_lo,
    slot_idx,
    barrier_id,
    release_in_bars,
    mdKaccum_cur,
    mdQaccum_cur,
    seqlen_k,
    num_m_block,
    n_block,
    tmem_ptr,
    sOutAccum,
    mbar_out_full,
    mbar_out_empty,
    mbar_dKin_empty,
    mbar_dVin_empty,
    mbar_dQin_empty,
):
    """One m iteration of the SPLIT output drain, for one warpgroup.

    kv_shared only. This warpgroup takes slices slice_lo, slice_lo + 2, ...
    of every output chunk; the other warpgroup takes the rest. Chunk order
    and the out_full / out_empty handshakes match the mma warp's issue order
    exactly -- only the slice set differs between the two callers, so the
    byte layout of the fp32 accumulators is unchanged and the shared
    postprocess keeps working. Each warpgroup stages into its OWN sOutAccum
    slot and tracks its OWN bulk-group stream; the single-slot pipeline per
    warpgroup is T2R(s) -> wait own previous reduce -> fill -> reduce(s),
    with the wait sitting after the T2R so the previous reduce's latency
    hides under it.

    release_in_bars: exactly one caller (the drain warps) releases the SMEM
    operands (dKin / dVin / dQin empty) -- their arrive counts are sized for
    one warpgroup.
    """
    ncol = cutlass.const_expr(self.dQ_reduce_ncol)
    flen_slice = cutlass.const_expr(cute.size(shape_out_slice))
    num_wg_threads = cutlass.const_expr(cute.arch.WARP_SIZE * 4)
    out_reduce_bytes = cutlass.const_expr(
        self.tile_m * ncol * Float32.width // 8
    )
    num_n_block_out = cute.ceil_div(seqlen_k, self.tile_n)
    out_base_ptr = tmem_ptr + self.tmem_out_offset
    outputs = (
        (self.num_d_chunks, self.d_chunk, mdKaccum_cur,
         self.accum_slice_d, num_n_block_out, n_block, self.out_base_dK,
         mbar_dKin_empty, mbar_dVin_empty),
        (self.num_d_chunks, self.d_chunk, mdQaccum_cur,
         self.accum_slice_d, num_m_block, m_iter, self.out_base_dQ,
         mbar_dQin_empty, None),
    )
    for oi in cutlass.range_constexpr(len(outputs)):
        (nchunks, chunk_w, maccum, hd_slice, num_blocks,
         block_idx, base, in_bar, in_bar2) = outputs[oi]
        chunks_per_slice = cutlass.const_expr(hd_slice // chunk_w)
        slice_stride = num_blocks * (self.tile_m * hd_slice)
        block_base = block_idx * (self.tile_m * hd_slice)
        my_slices = cutlass.const_expr(
            tuple(range(slice_lo, chunk_w // ncol, 2))
        )
        for pos_in_seg in cutlass.range_constexpr(nchunks):
            c = cutlass.const_expr(nchunks - 1 - pos_in_seg)
            out_c = cutlass.const_expr(base + c)
            chunk_base = cutlass.const_expr(
                (c % chunks_per_slice) * (self.tile_m * chunk_w)
            )
            slice_idx = cutlass.const_expr(c // chunks_per_slice)
            cute.arch.mbarrier_wait(mbar_out_full + out_c, phase)
            if cutlass.const_expr(release_in_bars):
                cute.arch.mbarrier_arrive(in_bar + c)
                if cutlass.const_expr(in_bar2 is not None):
                    cute.arch.mbarrier_arrive(in_bar2 + c)
            elem_base = slice_idx * slice_stride + block_base + chunk_base
            slot_base = cutlass.const_expr(
                (self.out_pos(out_c) % self.num_out_slots)
                * self.tmem_out_slot_cols
            )
            for si in cutlass.range_constexpr(len(my_slices)):
                s = cutlass.const_expr(my_slices[si])
                frag = cute.make_fragment(shape_out_slice, Float32)
                tmem_src = cute.make_tensor(
                    out_base_ptr + slot_base + s * ncol, out_slice_layout
                )
                cute.copy(
                    thr_t2r_out, thr_t2r_out.partition_S(tmem_src), frag
                )
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(si == len(my_slices) - 1):
                    # This warpgroup's share of the slot is in registers;
                    # the other warpgroup arrives for its own share, and the
                    # mma warp reuses the slot at the combined count.
                    cute.arch.mbarrier_arrive(mbar_out_empty + out_c)
                # ONE staging slot per warpgroup: the previous bulk reduce
                # must have READ it before this fill. Waiting here, after
                # the T2R, hides that reduce's latency under the T2R
                # instead of exposing it right after its commit.
                if wg_tidx < cute.arch.WARP_SIZE:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(
                    barrier_id=barrier_id,
                    number_of_threads=num_wg_threads,
                )
                # r2s of this warpgroup's slice into its staging slot.
                sslot = sOutAccum[None, slot_idx].iterator + wg_tidx * 4
                cute.autovec_copy(
                    cute.make_tensor(
                        frag.iterator,
                        cute.make_layout((4, flen_slice // 4), stride=(1, 4)),
                    ),
                    cute.make_tensor(
                        sslot,
                        cute.make_layout(
                            (4, flen_slice // 4), stride=(1, num_wg_threads * 4)
                        ),
                    ),
                )
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(
                    barrier_id=barrier_id,
                    number_of_threads=num_wg_threads,
                )
                if wg_tidx < cute.arch.WARP_SIZE:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sOutAccum[None, slot_idx].iterator,
                            maccum.iterator
                            + (elem_base + s * (self.tile_m * ncol)),
                            out_reduce_bytes,
                        )
                    cute.arch.cp_async_bulk_commit_group()


class FlashAttentionBackwardSm100BigD:
    """SM100 backward kernel specialized for head dimensions larger than 256."""

    @classmethod
    def from_shape(
        cls,
        head_dim: int,
        head_dim_v: int,
        cta_group: int = 1,
        kv_shared: bool = False,
        **kwargs,
    ):
        """Build a kernel with the selected configuration for a head shape.

        Args:
            head_dim: Query/key head dimension.
            head_dim_v: Value head dimension.
            cta_group: Number of CTAs participating in each MMA.
            kv_shared: Whether k and v alias, which changes the pin the shape gets.
            **kwargs: Explicit constructor overrides for selected configuration values.

        Returns:
            Configured ``FlashAttentionBackwardSm100BigD`` instance.
        """
        cfg = solve_config(head_dim, head_dim_v, cta_group=cta_group)
        # A shape's pin may carry a "kv_shared" block of overrides, layered on top when
        # k and v alias: the merge constrains the chunk widths and the SMEM budget
        # differently (see _MEASURED_CONFIG). An empty pin is the "solver only" case.
        pin = dict(_MEASURED_CONFIG.get((head_dim, head_dim_v, cta_group), {}))
        kv_shared_pin = pin.pop("kv_shared", {})
        if kv_shared:
            pin.update(kv_shared_pin)
        assert pin.keys() <= set(CONFIG_KEYS), (
            f"pin for {(head_dim, head_dim_v, cta_group)} (kv_shared={kv_shared}) sets "
            f"keys the constructor does not take: "
            f"{sorted(pin.keys() - set(CONFIG_KEYS))}"
        )
        cfg = {**cfg, **pin}
        # Precedence: explicit kwargs beat the pin, and the pin beats the solver.
        for key in CONFIG_KEYS:
            kwargs.setdefault(key, cfg[key])
        return cls(
            head_dim, head_dim_v, cta_group=cta_group, kv_shared=kv_shared, **kwargs
        )

    def __init__(
        self,
        head_dim: int,
        head_dim_v: int,
        is_causal: bool = False,
        qhead_per_kvhead: int = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        d_chunk: int = 96,
        dv_chunk: int = 128,
        d_chunks_resident: int = 3,
        dQ_reduce_ncol: int = 32,
        cta_group: int = 1,
        swap_dKV: bool = False,
        deterministic: bool = False,
        fm_bound_num: int = 0,
        kv_shared: bool = False,
    ):
        # head_dim is padded to a multiple of 64 to match head_dim_rounded in the
        # interface. Out-of-range columns need no predication here: the TMA zero-fills
        # them and the per-element mask below kills their contribution.
        hdim_multiple_of = 64
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # kv_shared merges dV chunk c into dK chunk c, so the dv axis has to be tiled
        # exactly like the d axis. When head_dim_v < head_dim (576/512) that means
        # padding dv up to head_dim and letting the dO TMA zero-fill columns
        # [head_dim_v, head_dim): dP += K[:, 512:576] @ 0 = 0 and dV[:, 512:576] =
        # P^T @ 0 = 0, so the merged chunk is dkv[:, 512:576] = dK[:, 512:576], which is
        # exactly right. No V buffer is involved -- under kv_shared the dP gemm reads sK.
        if kv_shared:
            self.tile_hdimv = self.tile_hdim

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.d_chunk = d_chunk
        self.dv_chunk = dv_chunk
        self.d_chunks_resident = d_chunks_resident
        # Width of one output T2R / reduce slice. Narrower slices mean more, smaller
        # vector atomics; 32 is the default and 64 is what 512/512 measured best at. Any
        # multiple of 4 keeps the byte layout the postprocess expects (its run is ordered
        # [4-column group][row][4 columns], so a slice of any multiple of 4 columns is
        # still a contiguous range).
        assert dQ_reduce_ncol % 4 == 0 and d_chunk % dQ_reduce_ncol == 0
        assert dv_chunk % dQ_reduce_ncol == 0
        self.dQ_reduce_ncol_cfg = dQ_reduce_ncol
        self.cta_group_size = cta_group
        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        # Number of columns of startend_row_indices (0 = no flashmask). It has to be a
        # constructor arg: the tensor arrives with a fully dynamic layout, so the kernel
        # cannot read it off the shape at trace time.
        #
        # The legal widths depend on is_causal, and the dependency is load bearing in
        # two places, so an out-of-set width is a silently wrong answer rather than a
        # missed optimization:
        #   - the per-element mask derives `has_end` as (causal and 2) or (not causal
        #     and 4); a non-causal width of 1 takes the has_end=False path, which reads
        #     fm_row[1] -- past the end of a single-column tensor.
        #   - the m-block skip only reduces an `end` bound when the width is >= 2, so a
        #     non-causal width of 1 leaves min_end at INT32_MAX, which pushes m_lo past
        #     m_hi and makes num_iters 0: the whole n block computes nothing and dK / dV
        #     / dQ stay at their zero-initialised accumulator values.
        legal_bound_num = (0, 1, 2) if is_causal else (0, 2, 4)
        assert fm_bound_num in legal_bound_num, (
            f"startend_row_indices width {fm_bound_num} is not valid for "
            f"is_causal={is_causal} (legal widths: {legal_bound_num})"
        )
        self.fm_bound_num = fm_bound_num

        # Deterministic (bitwise reproducible) reduction. dQ / dK / dV are cross-CTA
        # fp32 gmem reduces, so the arrival order decides the result. Under this flag a
        # global int32 semaphore pins that order: a CTA spins until the counter equals
        # its own ordinal, reduces, then increments. See the drain warp below.
        self.deterministic = deterministic
        # cta_group=2 is NOT ENABLED and has never run: _launch builds multicast TMA
        # atoms from cluster_shape_mnk but passes no `cluster=` to .launch(), and the grid
        # is not rounded to a multiple of the pair size. The column rules in tmem_plan(),
        # the n_block_pair skip derivation and the tile_m constraint below are all in
        # place for it, but turning it on means fixing the launch and the grid first.
        assert cta_group in (1, 2)

        assert not swap_dKV, "swapAB (phase 2) is not implemented yet"
        # UMMA N limit. This is what rules out the d=576 -> 2x288 split that the
        # d=256 config uses.
        assert d_chunk <= 256 and dv_chunk <= 256, "chunk width exceeds the UMMA N limit"
        # The dQ / dK reduce paths tile the flat fp32 workspace by dQ_reduce_ncol
        # (32) and dK_reduce_ncol, so the chunk width must be a multiple of 32.
        assert d_chunk % 32 == 0 and dv_chunk % 32 == 0
        assert self.tile_hdim % d_chunk == 0, (
            f"head_dim {self.tile_hdim} must be a whole number of {d_chunk}-wide chunks"
        )
        assert self.tile_hdimv % dv_chunk == 0, (
            f"head_dim_v {self.tile_hdimv} must be a whole number of {dv_chunk}-wide chunks"
        )
        self.num_d_chunks = self.tile_hdim // d_chunk
        self.num_dv_chunks = self.tile_hdimv // dv_chunk
        assert 1 <= d_chunks_resident <= self.num_d_chunks
        # head_dim slicing of the fp32 accumulators, so the shared postprocess can be
        # called once per slice. Public: interface.py reads these to drive the
        # postprocess loop.
        self.accum_slice_d = accum_slice_width(self.tile_hdim, d_chunk)
        self.accum_slice_dv = accum_slice_width(self.tile_hdimv, dv_chunk)

        # KV shared: k and v are the SAME tensor (v = k[..., :head_dim_v]), so dV and dK
        # are two halves of one gradient -- dKV[:, :head_dim_v] = dK + dV, and the
        # framework's two grads for the aliased input add up to exactly that. Instead of
        # producing them separately, the dV gemm runs with zero_init=False on top of the
        # dK gemm's accumulator, so a chunk leaves TMEM as dKV and is flushed ONCE.
        #
        # Two things that costs nothing and one it buys: the flush drops from
        # tile_n*(d+dv) + tile_m*d to tile_n*d + tile_m*d per m tile, and num_dv_chunks
        # output chunks disappear with their tcgen05.commit and out barrier pair. The
        # output chunk COUNT is the axis this kernel measured as dominant: raising it by
        # narrowing d_chunk and adding K residency cost noticeably more time.
        # The math is unchanged (dV is still P^T@dO), only the accumulator is shared.
        #
        # The merge needs dV chunk c to cover exactly dK chunk c's columns, hence equal
        # widths. 576/512 has no width that divides both axes, so the dv axis is padded
        # to 576 instead (see tile_hdimv above) and runs at 192 like the d axis.
        self.kv_shared = kv_shared
        if kv_shared:
            assert d_chunk == dv_chunk, (
                "kv_shared merges dV chunk c into dK chunk c, so the two chunk widths "
                f"must match (got d_chunk={d_chunk}, dv_chunk={dv_chunk})"
            )
            assert head_dim_v <= head_dim, (
                "kv_shared needs every dV chunk to have a dK chunk to land in, i.e. "
                f"head_dim_v <= head_dim (got {head_dim} / {head_dim_v})"
            )

        # Output chunk numbering, shared by the mma warp (issue) and the drain warps
        # (wait): the dV, dK and dQ segments in that order. With kv_shared the dV
        # segment folds onto dK's, so there are 2 * num_d_chunks chunks, not 3.
        self.out_base_dV = 0
        self.out_base_dK = 0 if kv_shared else self.num_dv_chunks
        self.out_base_dQ = self.out_base_dK + self.num_d_chunks
        self.num_out_chunks = self.out_base_dQ + self.num_d_chunks

        # ---------------------------------------------------------------- MMA tilers
        cg = self.cta_group_size
        # S^T = K @ Q^T          (contraction over d, accumulated over d chunks)
        self.mma_tiler_kq = (cg * tile_n, tile_m, d_chunk)
        # dP^T = V @ dO^T        (contraction over dv, accumulated over dv chunks)
        self.mma_tiler_vdo = (cg * tile_n, tile_m, dv_chunk)
        # dV_c = P^T @ dO_c      (output split over dv)
        self.mma_tiler_pdo = (cg * tile_n, dv_chunk, tile_m)
        # dK_c = dS^T @ Q_c      (output split over d)
        self.mma_tiler_dsq = (cg * tile_n, d_chunk, tile_m)
        # dQ_c = dS @ K_c        (output split over d)
        self.mma_tiler_dsk = (tile_m, d_chunk, tile_n * cg)
        if cg == 2:
            # A cta_group=2 MMA needs M in {128, 256}; dQ's M is tile_m.
            assert tile_m >= 128, "cta_group=2 requires tile_m >= 128 for the dQ gemm"

        self.acc_dtype = Float32
        self.cluster_shape_mn = (cg, 1)
        self.cta_group = tcgen05.CtaGroup.TWO if cg == 2 else tcgen05.CtaGroup.ONE

        # ---------------------------------------------------------------- TMEM layout
        # S/P and dP/dS live across the whole m-iteration (P is the A operand of
        # the dV gemm, dS of the dK/dQ gemms). The three outputs are produced and
        # immediately drained, so they time-share one scratch region whose width
        # is the widest of them.
        #
        # Output scratch slots. All three outputs share the same columns -- they never
        # overlap in time -- but with a single slot the mma warp cannot issue chunk
        # c+1's gemm until the drain warps have emptied chunk c, so the ~20 output
        # gemms per m tile are fully serialised against their own drain.
        #
        # A second slot breaks that: chunk c+1 goes to the other slot, so its gemm
        # overlaps chunk c's T2R + reduce. This is what DSA does for its dKV (two
        # slots, with dKV2/3 aliased onto them) -- note it does NOT keep dKV resident
        # either, it double buffers.
        #
        # True residency for dK/dV is a different thing and does not fit here: at
        # tile_n=128 one resident dV is tile_n * head_dim_v * 4B = 256KB, which is
        # the entire 512-column TMEM (512 cols x 128 lanes x 4B). Both dK and dV
        # resident needs 512KB, i.e. cta_group=2 (two CTAs' TMEM, and each
        # accumulator split in half along N). No column trick substitutes for that.
        #
        # The arithmetic lives in tmem_plan() so that solve_config's feasibility test
        # and this layout cannot drift apart.
        plan = tmem_plan(tile_m, tile_n, d_chunk, dv_chunk, cg)
        self.tmem_S_cols = plan["s_cols"]

        self.tmem_S_offset = 0
        # P and dS are bf16, so they need only half the columns of the f32
        # accumulator they overlay, and they sit in its UPPER half: S (resp. dP)
        # must be fully read into registers before P (resp. dS) clobbers it.
        self.tmem_s_to_p_offset = tile_m // 2
        self.tmem_P_offset = self.tmem_S_offset + self.tmem_s_to_p_offset
        self.tmem_dP_offset = self.tmem_S_offset + self.tmem_S_cols
        self.tmem_dS_offset = self.tmem_dP_offset + self.tmem_s_to_p_offset
        self.tmem_out_offset = plan["out_offset"]
        self.tmem_out_slot_cols = plan["slot_cols"]
        self.num_out_slots = plan["num_out_slots"]
        assert self.num_out_slots >= 1, (
            f"no room for an output slot: out_offset={self.tmem_out_offset}, "
            f"slot={self.tmem_out_slot_cols}"
        )
        # Chunk c uses slot c % num_out_slots, starting at tmem_out_offset plus
        # (c % num_out_slots) * tmem_out_slot_cols.
        self.tmem_total = plan["total"]
        assert self.tmem_total <= SM100_TMEM_CAPACITY_COLUMNS, (
            f"TMEM overflow: {self.tmem_total} > {SM100_TMEM_CAPACITY_COLUMNS} columns"
        )
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # ------------------------------------------------------------ warps / barriers
        # The output drain (T2R of dV/dK/dQ -> SMEM staging -> cp.reduce into the
        # fp32 gmem accumulators) runs on its OWN warpgroup, not on the compute
        # warps. It used to sit at the tail of the compute warps' m iteration, which
        # serialised ~48 16KB slices against the next iteration's softmax / dS math
        # and kept three f32 output fragments (dV 128 + dK 64 + dQ 64 registers per
        # thread) live inside the compute warps' budget. DSA's sm100 bwd splits the
        # same way (4 compute + 8 reduce warps).
        self.drain_warp_ids = (0, 1, 2, 3)
        # The compute warpgroup is 4 warps, and the CTA is 12 warps. The TMEM->register
        # copies are partitioned over 128 threads (tcgen05.make_tmem_copy sizes its
        # thread layout from the accumulator and for these shapes that is 4 warps), so
        # 4 warps is also the natural width.
        #
        # Why 12 warps and not 16: this is a register-ceiling decision, not an occupancy
        # one. ptxas allocates 65536/threads_per_cta registers per thread, so at 512
        # threads it can only give every warp 128 -- setmaxnreg redistributes the
        # physical pool at runtime but cannot change what the code was compiled against.
        # The compute branch's live set is 4 x W f32 (S/P/dP/dS) plus 2 x W/2 packed =
        # 160 f32 before addressing, so at 16 warps it spilled unconditionally, with most
        # of the warp cycles between issues waiting on an L1TEX scoreboard. Every register
        # split tried at 16 warps (200/136, 152/136, 168/128, and 144 x 2 with the drain
        # doubled) measured flat, which is the signature of a constraint and not a knob.
        # At 384 threads ptxas allocates 162 and the local traffic is exactly zero.
        #
        # A second compute warpgroup (8 warps, interleaving the softmax chunks with a
        # rendezvous where the packed P / dS columns of one chunk overlap another's
        # reads) was worth a few percent, and it cannot coexist with this: 8 compute warps
        # + 4 drain + mma / load is 16 warps, i.e. back to 128 registers and the spill.
        # The two effects cancel to the same wall clock; this side of the trade is kept
        # because it also removes all local traffic.
        self.compute_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        # Warps 10-11 idle. They exist to complete the mma / load warpgroup: setmaxnreg
        # is warpgroup aligned.
        self.num_warps = 12
        self.threads_per_cta = cute.arch.WARP_SIZE * self.num_warps

        # setmaxnreg budget. The launch is 384 threads, so every warp starts at
        # 65536/384 = 168 registers (rounded to a multiple of 8); a warpgroup asking for
        # less must *decrease* and for more must *increase*, and the per-thread values
        # summed over the warpgroups must stay within 65536/128 = 512.
        #   drain 176 (warps 0-3)   compute 240 (4-7)
        #   load / mma 88 (8-11, warps 10-11 idle at 24)      sum 504
        # The drain keeps the odd one out. Its inner step is "T2R one dQ_reduce_ncol
        # slice into a 64 f32 fragment, then read that fragment out with 16
        # red.global.add.v4.f32", and the next slice's T2R has to wait for those
        # reductions to have read the registers. ncu calls that wait a scoreboard
        # dependency on an L1TEX operation, and it is most of the drain's whole stall
        # budget -- the single largest one in the kernel at 576/512, and still dominant
        # at 512/512 with the dKV merge.
        #
        # Do NOT try to hide that by giving the drain more live state. Measured on
        # 512/512 kv_shared: keeping every slice of a chunk in flight (2 x 64 f32) with a
        # 208/216 split pushed launch__registers_per_thread to the 65536/384 ceiling and
        # started spilling, and the duration came out flat -- the spill traffic lands on
        # L1TEX, the exact pipe the stall is on, and cancels the deeper pipeline.
        # setmaxnreg redistributes the physical pool at runtime but cannot change what
        # ptxas compiled against, so at 384 threads no register quota here buys the drain
        # a second live fragment. The lever for that stall is to issue FEWER T2R/atomic
        # pairs (dQ resident + TMA store, the way dsa_bwd_sm100.py does it), not to
        # pipeline the ones we have.
        #
        # Compute gets 240 instead of the 128-136 it was pinned to at 16 warps, which
        # is the whole point of dropping to 12 (see the warp ids above): its live set is
        # 160 f32 plus addressing, so 240 is the first quota that can hold it without
        # spilling. Measured history at 16 warps, all flat: 152/136, 168/128, 144 x 2
        # with drain doubled.
        self.num_regs_drain = 176
        self.num_regs_compute = 240
        self.num_regs_load = 88
        self.num_regs_mma = 88
        self.num_regs_empty = 24
        assert (
            self.num_regs_drain
            + self.num_regs_compute
            + max(self.num_regs_load, self.num_regs_mma, self.num_regs_empty)
            <= 512
        )
        # setmaxnreg takes values in [24, 256] in multiples of 8.
        for _q in (
            self.num_regs_drain,
            self.num_regs_compute,
            self.num_regs_load,
            self.num_regs_mma,
            self.num_regs_empty,
        ):
            assert 24 <= _q <= 256 and _q % 8 == 0, (
                f"setmaxnreg value {_q} is not in [24, 256] and a multiple of 8"
            )

        self.buffer_align_bytes = 1024
        # Width, in m columns, of one S -> P -> dP -> dS round trip. The live register
        # set of that round trip is 5 * softmax_chunk_m f32 per thread (S, P, dP, dS
        # and the two packed R2T buffers, which are half-width), so this is what keeps
        # the compute warps out of local memory: at the full tile_m = 128 that set is far
        # more than the per-thread register quota and ncu showed heavy local traffic with
        # most of the stall budget on L1TEX. 32 brings it inside the quota.
        #
        # Must divide tile_m and be a multiple of 32: the packed bf16 P / dS region is
        # addressed in W // 32 * 16 f32-equivalent columns, and the R2T store rep is
        # W // 8.
        self.softmax_chunk_m = 32 if tile_m % 32 == 0 else tile_m
        assert self.tile_m % self.softmax_chunk_m == 0
        assert self.softmax_chunk_m % 32 == 0
        self.num_softmax_chunks = self.tile_m // self.softmax_chunk_m

    def _setup_attributes(self):
        # Q and dO hold one chunk at a time, K holds `d_chunks_resident` for the whole
        # n-block. Whether a chunk is actually re-fetched for the gemm that reads it a
        # second time is decided by the reuse schedules below, not by the stage count.
        self.Q_stage = 1
        self.dO_stage = 1
        self.K_smem_stages = self.d_chunks_resident

        # Output reduce granularity: the T2R slice width, which also fixes the byte
        # layout of the fp32 accumulators and therefore has to stay something
        # FlashAttentionBackwardPostprocess can read back.
        self.dQ_reduce_ncol = self.dQ_reduce_ncol_cfg
        assert (self.d_chunk // self.cta_group_size) % self.dQ_reduce_ncol == 0

        # Staging slots for the output drain's bulk reduce-add (kv_shared only; the
        # split path keeps the register atomics). Two slots so that filling slot s+1
        # overlaps slot s's cp.reduce.async.bulk, which is the whole point -- at ONE
        # slot the fill has to wait for the previous reduce to have read it and the
        # drain fully serialises (that is what the earlier staged version did, see the
        # drain's comment). The SMEM for them is what dropping sV under kv_shared
        # freed: 2 * tile_m * dQ_reduce_ncol * 4B, which is more than sV's own bytes but
        # fits in what the budget already had spare.
        #
        # Two slots is the measured optimum for the depth. The drain is what the profile
        # blames -- most of the cycles between issued instructions are a scoreboard
        # dependency on an L1TEX op. The trade is T2R+reduce pair count against pipeline
        # depth and both sides fit SMEM, so both directions were swept on 576/512
        # kv_shared: ncol 64 with 1 slot (half the pairs) and ncol 32 with 3 slots (a
        # deeper pipeline) both came out slower than this default.
        #
        # So this axis is done: neither fewer L1TEX ops nor a deeper pipeline helps,
        # which means the stall is not the drain's own shape -- it is latency the CTA has
        # too few warps to hide (cudnn's DSA bwd carries the same L1TEX throughput at 20
        # warps against our 12 and a much lower warp-cycles-per-issue). Do not re-sweep
        # ncol / the slot count; change the warp count or remove the dQ half of the drain
        # instead.
        #
        # Throwaway probes that decomposed the drain into its gmem and on-chip parts put
        # the whole drain at roughly 40% of the kernel, split about evenly between its dQ
        # and dKV halves, and showed the DRAM side to be nearly free: both halves are
        # almost entirely on-chip (T2R + st.shared), dq_accum's round trip is worth very
        # little and dk_accum stays in L2. The two halves also cost the same per byte, so
        # the drain's cost is proportional to
        #   density * seqlen^2 * head_dim * 4 * num_head * (1/tile_m + 1/tile_n)
        # -- the dKV term depends only on tile_m and the dQ term only on tile_n.
        #
        # Two consequences. A bf16 accumulator would buy almost nothing, so halving the
        # accumulator's bytes is pointless. And making dQ resident by flipping to a
        # Q-outer loop does NOT help: it needs tile_m 64 for the folded TMEM
        # accumulator, which doubles the dKV term and gives exactly the bytes back.
        # What does help is dividing a term by something other than a tile size:
        # packing H q heads into one CTA divides the dKV term by H (they share one
        # dK / dV accumulator when num_head_kv == 1).
        # deterministic never enters the staged drain (see the register-atomic branch in
        # the drain warp), so the staging slots would be reserved and never touched.
        self.out_stage = 2 if (self.kv_shared and not self.deterministic) else 0
        self.out_stage_elems = self.tile_m * self.dQ_reduce_ncol
        # Two-warpgroup drain split (kv_shared only). With ONE TMEM out slot (the
        # 576/512 config) the chunk chain "output gemm -> T2R -> next gemm" is
        # serial: the mma warp cannot issue chunk c-1 until out_empty(c), i.e. until
        # every slice of chunk c has left TMEM -- and the compute warps' softmax of
        # the NEXT m tile cannot start either, because the mma warp only reaches its
        # S gemms after the dQ gemms, which sit behind the same slot gate. So the
        # whole 2 * num_d_chunks * (d_chunk / ncol) slice chain is on the critical
        # path once per m iteration (which is exactly what the drain-skip probes
        # measured: removing half the slices bought a large chunk of the drain's cost).
        # Splitting each
        # chunk's slices by PARITY between the drain and compute warpgroups halves
        # that serial T2R chain: both warpgroups read different columns of the same
        # TMEM slot concurrently, and the compute warps are already done with their
        # softmax by the time the output gemms run. Each warpgroup gets its OWN
        # staging slot (drain: 0, compute: 1) and its own named barrier (4 / 5), so
        # the two bulk-reduce streams share no state.
        # `deterministic` forces this off: the dQ semaphore has a single releaser per
        # CTA, so exactly one warpgroup must own the whole dQ drain.
        self.drain_split = (
            self.kv_shared
            and self.out_stage >= 2
            and (self.d_chunk // self.dQ_reduce_ncol) % 2 == 0
            and not self.deterministic
        )

        # SMEM operand reuse. K carries across m iterations, Q / dO do not. V has a
        # single consumer (the dP gemm), so there is no second pass to recycle it from
        # and it keeps its own hand-written streaming.
        self.sched_K = _reuse_schedule(
            self.num_d_chunks, self.K_smem_stages, carry=True
        )
        self.sched_Q = _reuse_schedule(self.num_d_chunks, self.Q_stage, carry=False)
        self.sched_dO = _reuse_schedule(self.num_dv_chunks, self.dO_stage, carry=False)
        # Only K can need a prologue: a Q / dO buffer that still holds the right chunk
        # holds it for the *previous* m tile, so their first access is never a hit.
        assert not self.sched_Q[2] and not self.sched_dO[2]

        # dV / dK / dQ are issued in ONE order shared by the mma warp and the drain
        # warps, and each segment runs DESCENDING so that the chunk the S / dP pass left
        # in SMEM is the first one the output pass asks for. The order has to be shared:
        # the drain gates the mma warp through num_out_slots scratch slots, so a drain
        # that waits in a different order than the mma warp issues deadlocks as soon as
        # the two diverge by more than the slot count.
        self.out_issue = (
            (
                ()
                if self.kv_shared
                else tuple(reversed(range(self.num_dv_chunks)))
            )
            + tuple(
                self.out_base_dK + c for c in reversed(range(self.num_d_chunks))
            )
            + tuple(
                self.out_base_dQ + c for c in reversed(range(self.num_d_chunks))
            )
        )

        # mbarrier slot offsets inside the single mbar_ptr MemRange: the pipeline slots
        # occupy [0, mbar_count() - 1) and the one scalar barrier sits at the end.
        self.mbar_tmem_dealloc_offset = self.mbar_count() - 1

    def out_pos(self, out_c):
        """Position of an output chunk in the shared dV / dK / dQ issue order."""
        return self.out_issue.index(out_c)

    def _get_tiled_mma(self, ab_dtype):
        cg = self.cta_group
        mk = tcgen05.OperandMajorMode.K
        mn = tcgen05.OperandMajorMode.MN
        tmem_src = tcgen05.OperandSource.TMEM
        mma = lambda tiler, b_major, a_src=None: sm100_utils_basic.make_trivial_tiled_mma(
            ab_dtype, mk, b_major, self.acc_dtype, cg, tiler[:2], *( (a_src,) if a_src else () )
        )
        tiled_mma_S = mma(self.mma_tiler_kq, mk)
        tiled_mma_dP = mma(self.mma_tiler_vdo, mk)
        # A operands of the dV / dK gemms come from TMEM (P and dS), which is the
        # path flash_fwd_sm100 / flash_bwd_sm100 use. It requires the K-side
        # accumulators to be M=128 so that the 32-datapath T2R/R2T atoms apply:
        # with M=64 the 16-datapath atoms are forced, and their per-thread element
        # order does not survive the bf16 packing (measured: exact for a constant A
        # operand, ~7% off for A = m or A = n, i.e. a local permutation). Hence
        # the K-side MMA M mode is fixed at 128; tile_m independently remains 128
        # for the accumulator staging and dQ mapping used by this kernel.
        tiled_mma_dV = mma(self.mma_tiler_pdo, mn, tmem_src)
        tiled_mma_dK = mma(self.mma_tiler_dsq, mn, tmem_src)
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            ab_dtype, mn, mn, self.acc_dtype, cg, self.mma_tiler_dsk[:2]
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self, ab_dtype):
        mma_S, mma_dP, mma_dK, mma_dV, mma_dQ = self._get_tiled_mma(ab_dtype)
        la = sm100_utils_basic.make_smem_layout_a
        lb = sm100_utils_basic.make_smem_layout_b

        # The transposed views below (sQt / sKt / sdO) are the SAME BYTES as their
        # partners (sQ / sK / sdOt), which is what lets the gemm reading a tile second
        # reuse what the first left in SMEM (see _reuse_schedule). From
        # cutlass.utils.blackwell_helpers: an operand layout is
        # tile_to_mma_shape(atom, shape, order) with the atom chosen by the MAJOR mode
        # size and order (1,2,3) for K-major, (2,1,3) for MN-major. K_SWnn and MN_SWnn
        # are each other's transpose and the order flip lays out the non-major axis
        # first, so a K-major (X, Y) tile and an MN-major (Y, X) tile sharing major axis
        # Y are the same map. The kernel already relies on this for dS (sdSt is the
        # A-K-major (n, m) view and sdS the A-MN-major (m, n) view of one buffer), so the
        # dQ path is its standing test.
        #
        # This holds at the major-mode sizes this config runs at, not universally: the
        # d_chunk=64 pair really is transposed, sK ((128,16),1,4,8):((64,1),0,16,8192)
        # against sKt ((64,16),1,8,8):((1,64),0,1024,8192), because a 64-element major
        # mode selects a different swizzle atom on the two sides. Hence the assertion at
        # the end of this function.

        # S^T = K @ Q^T : all d chunks of K stay resident for the n-block, Q is
        # streamed one chunk at a time.
        self.sK_layout = la(mma_S, self.mma_tiler_kq, ab_dtype, self.K_smem_stages)
        self.sQ_layout = lb(mma_S, self.mma_tiler_kq, ab_dtype, self.Q_stage)
        # dP^T = V @ dO^T
        self.sV_layout = cute.slice_(
            la(mma_dP, self.mma_tiler_vdo, ab_dtype, 1), (None, None, None, 0)
        )
        self.sdOt_layout = lb(mma_dP, self.mma_tiler_vdo, ab_dtype, self.dO_stage)
        # dV_c = P^T @ dO_c   (P comes from TMEM)
        self.tP_layout = cute.slice_(
            la(mma_dV, self.mma_tiler_pdo, ab_dtype, 1), (None, None, None, 0)
        )
        self.sdO_layout = lb(mma_dV, self.mma_tiler_pdo, ab_dtype, self.dO_stage)
        # dK_c = dS^T @ Q_c   (dS comes from TMEM)
        self.tdS_layout = cute.slice_(
            la(mma_dK, self.mma_tiler_dsq, ab_dtype, 1), (None, None, None, 0)
        )
        # The SMEM staging of dS must come from a SMEM-sourced mma: make_smem_layout_a
        # on the TMEM-sourced tiled_mma_dK describes the TMEM operand (two bf16 per
        # 32-bit lane), so its element count is 2x and partition_D handed each thread
        # 256 elements against the 128 the T2R produced.
        mma_dK_smem = sm100_utils_basic.make_trivial_tiled_mma(
            ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_dsq[:2],
        )
        self.sdSt_layout = cute.slice_(
            la(mma_dK_smem, self.mma_tiler_dsq, ab_dtype, 1), (None, None, None, 0)
        )
        self.sQt_layout = lb(mma_dK, self.mma_tiler_dsq, ab_dtype, self.Q_stage)
        # dQ_c = dS @ K_c
        self.sdS_layout = cute.slice_(
            la(mma_dQ, self.mma_tiler_dsk, ab_dtype, 1), (None, None, None, 0)
        )
        self.sKt_layout = lb(mma_dQ, self.mma_tiler_dsk, ab_dtype, self.K_smem_stages)

        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64))
        )

        # A transposed view that is not byte-identical to its partner would make the
        # second gemm read a transposed tile -- which compiles and returns a wrong
        # answer -- so check the extents agree at the sizes this config runs at.
        for k_major, mn_major in (
            (self.sQ_layout, self.sQt_layout),
            (self.sK_layout, self.sKt_layout),
            (self.sdOt_layout, self.sdO_layout),
        ):
            assert cute.cosize(k_major) == cute.cosize(mn_major), (
                "an operand and its transposed view must occupy the same bytes"
            )

    def mbar_count(self):
        """Number of Int64 mbarrier slots in SharedStorage.

        Exactly the layout the kernel builds (see the mbar_* offsets there), because a
        fixed fudge factor silently capped the config space: with 8 d chunks and 8 dv
        chunks the slots overflowed the MemRange and the config became unusable.

          Sin  full/empty    2 * num_d_chunks       S full          1
          dPin full/empty    2 * num_dv_chunks      dP full         1
          dVin full/empty    2 * num_dv_chunks      PdS full        1
          dKin full/empty    2 * num_d_chunks       dSsmem full     1
          dQin full/empty    2 * num_d_chunks       K prologue      1
          out  full/empty    2 * num_out_chunks   (dV folds onto dK when kv_shared)
                                                    stats full      1
                                                    stats empty     1
                                                    tmem dealloc    1

        tmem dealloc is last, so it is the slot mbar_tmem_dealloc_offset points at.
        """
        per_chunk = (
            2 * self.num_d_chunks       # Sin
            + 2 * self.num_dv_chunks    # dPin
            + 2 * self.num_dv_chunks    # dVin
            + 2 * self.num_d_chunks     # dKin
            + 2 * self.num_d_chunks     # dQin
            + 2 * self.num_out_chunks   # out
        )
        # S full, dP full, PdS full, dSsmem full, K prologue, stats full / empty,
        # tmem dealloc.
        scalars = 8
        return per_chunk + scalars

    def make_shared_storage(self, q_dtype, do_dtype, ds_dtype):
        """Build the shared-memory storage type for the selected configuration.

        Args:
            q_dtype: Element type used by Q, K, and V staging buffers.
            do_dtype: Element type used by dO staging buffers.
            ds_dtype: Element type used by dS staging buffers.

        Returns:
            Cute struct type describing the kernel's shared-memory allocation.

        Aliasing, mirroring the 1CTA layout of flash_bwd_sm100.py:
          - sQ also backs sQt (transposed view)
          - sdO also backs sdOt
          - sK also backs sKt

        The outputs have no SMEM buffer of their own: dV / dK / dQ leave TMEM through
        vector atomics straight out of the drain warps' registers.
        """
        sQ_alloc_bytes = max(
            cute.size_in_bytes(q_dtype, self.sQ_layout),
            cute.size_in_bytes(q_dtype, self.sQt_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(do_dtype, self.sdO_layout),
            cute.size_in_bytes(do_dtype, self.sdOt_layout),
        )
        # Under kv_shared V IS K: same rows, head_dim_v == head_dim and dv_chunk ==
        # d_chunk, so mma_tiler_vdo == mma_tiler_kq and sV_layout is exactly one stage
        # of sK_layout. The dP gemm therefore reads the sK stage the S gemm just read
        # (the two are fused, see the mma warp) and V needs neither storage nor a TMA.
        # The buffer is not wasted: it becomes the drain's bulk reduce-add staging, so
        # the field is sized for whichever of the two the mode needs (they are never
        # both live) and typed as bytes like sQ. Under kv_shared + deterministic neither
        # is needed -- the drain uses per-thread atomics, so out_stage is 0 -- and the
        # field collapses to zero bytes.
        sV_bytes = cute.size_in_bytes(q_dtype, self.sV_layout)
        sV_alloc_bytes = max(
            0 if self.kv_shared else sV_bytes,
            self.out_stage * self.out_stage_elems * Float32.width // 8,
        )
        mbar_count = self.mbar_count()
        align = self.buffer_align_bytes

        @cute.struct
        class SharedStorage:
            """Shared-memory buffers and barriers used by one BigD CTA."""

            mbar_ptr: cute.struct.MemRange[cutlass.Int64, mbar_count]
            tmem_holding_buf: cutlass.Int32
            sFM_max_min_ptr: cute.struct.MemRange[
                cutlass.Int32, FLASHMASK_META_SLOTS
            ]

            sQ: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes], align]
            sK: cute.struct.Align[
                cute.struct.MemRange[q_dtype, cute.cosize(self.sK_layout)], align
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sV_alloc_bytes], align
            ]
            sdO: cute.struct.Align[cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes], align]
            sdS: cute.struct.Align[
                cute.struct.MemRange[ds_dtype, cute.cosize(self.sdSt_layout)], 128
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sLSE_layout)], 128
            ]
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.sdPsum_layout)], 128
            ]

        return SharedStorage

    # `__call__` matches FlashAttentionBackwardSm100's signature so interface.py can pick
    # between the two by shape alone.

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        window_size_left=None,
        window_size_right=None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        blocksparse_tensors=None,
        flashmask_info=None,
        overlap_k_addr=None,
        overlap_v_addr=None,
        overlap_work_done_addr=None,
        overlap_segment_idx=None,
        overlap_dk_addr=None,
        overlap_dv_addr=None,
        overlap_b=None,
        overlap_s=None,
        overlap_h=None,
        overlap_d=None,
        overlap_comm_rpb: cutlass.Constexpr = None,
        overlap_bhsd_layout: cutlass.Constexpr = False,
        # Always keep stream as the last parameter.
        stream=None,
    ):
        # None of the features below are implemented; fail loudly rather than silently
        # wrong.

        assert window_size_left is None and window_size_right is None, (
            "local attention is not supported by the big-headdim bwd yet"
        )
        assert blocksparse_tensors is None, (
            "block sparsity is not supported by the big-headdim bwd"
        )
        assert (flashmask_info is None) == (self.fm_bound_num == 0), (
            "fm_bound_num must match whether flashmask_info was passed"
        )
        if cutlass.const_expr(flashmask_info is not None):
            assert flashmask_info.is_causal == self.is_causal, (
                "flashmask_info.is_causal disagrees with the kernel's is_causal"
            )
        if cutlass.const_expr(self.deterministic):
            assert mdQ_semaphore is not None, (
                "deterministic needs a dQ semaphore"
            )
            assert self.qhead_per_kvhead == 1 or mdK_semaphore is not None, (
                "deterministic with qhead_per_kvhead > 1 needs a dK semaphore"
            )
            # mdV_semaphore is accepted but unused: the single dK lock is held across the
            # whole m loop, which already serialises this CTA's dV reduces too.
        else:
            assert mdQ_semaphore is None and mdK_semaphore is None and mdV_semaphore is None, (
                "semaphores were passed but the kernel was not built with deterministic=True"
            )
        assert overlap_dk_addr is None and overlap_dv_addr is None, (
            "dK / dV always leave this kernel through the fp32 accumulators, so the "
            "overlap send buffers are filled by the postprocess, not here"
        )
        # FM-4 overlap: K and V live in the split-AG SRBuffer, which has no dlpack
        # capsule, so rebuild the views from the raw segment addresses. The dims must
        # stay runtime Int32 -- a static layout specializes the TMA descriptor
        # differently and reads the wrong bytes.
        self.overlap_bhsd_layout = cutlass.const_expr(overlap_bhsd_layout)
        if cutlass.const_expr(overlap_k_addr is not None):
            from_addr = (
                utils.make_bhsd_storage_bshd_from_addr
                if cutlass.const_expr(overlap_bhsd_layout)
                else utils.make_contiguous_bshd_from_addr
            )
            mK = from_addr(
                overlap_k_addr,
                overlap_b,
                overlap_s,
                overlap_h,
                overlap_d,
                mQ.element_type,
                align=16,
            )
            mV = from_addr(
                overlap_v_addr,
                overlap_b,
                overlap_s,
                overlap_h,
                overlap_d,
                mQ.element_type,
                align=16,
            )
        # mdK / mdV are the fp32 accumulators here (need_kv_accum is forced on for
        # this kernel: dK and dV always leave TMEM through gmem).
        self._launch(
            mQ,
            mK,
            mV,
            mdO,
            mLSE,
            mdPsum,
            mdQaccum,
            mdK,
            mdV,
            softmax_scale * math.log2(math.e),
            stream,
            mFM=(
                None
                if cutlass.const_expr(flashmask_info is None)
                else flashmask_info.startend_row_indices
            ),
            mdQ_semaphore=mdQ_semaphore,
            mdK_semaphore=mdK_semaphore,
            mdV_semaphore=mdV_semaphore,
            overlap_work_done_addr=overlap_work_done_addr,
            overlap_segment_idx=overlap_segment_idx,
            overlap_comm_rpb=overlap_comm_rpb,
        )

    def _launch(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        softmax_scale_log2: Float32,
        stream,
        mFM: Optional[cute.Tensor] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        overlap_work_done_addr: Optional[cutlass.Int64] = None,
        overlap_segment_idx: Optional[Int32] = None,
        overlap_comm_rpb: cutlass.Constexpr = None,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.ds_dtype = mQ.element_type

        self._setup_attributes()
        self._setup_smem_layout(self.q_dtype)
        (
            tiled_mma_S,
            tiled_mma_dP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
        ) = self._get_tiled_mma(self.q_dtype)
        self.cluster_shape_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.shared_storage = self.make_shared_storage(
            self.q_dtype, self.do_dtype, self.ds_dtype
        )
        # solve_config's SMEM model is an estimate (it does not know about the barriers,
        # LSE / dPsum or cute.struct's alignment padding) and _MEASURED_CONFIG can pin a
        # config the estimate never approved, so check the real struct against the
        # hardware cap here rather than finding out as a launch failure.
        smem_bytes = self.shared_storage.size_in_bytes()
        assert smem_bytes <= SM100_SMEM_CAPACITY_BYTES, (
            f"shared storage is {smem_bytes} B, over the {SM100_SMEM_CAPACITY_BYTES} B "
            f"SM100 cap (tile_m={self.tile_m}, tile_n={self.tile_n}, "
            f"d_chunk={self.d_chunk}, dv_chunk={self.dv_chunk}, "
            f"d_chunks_resident={self.d_chunks_resident})"
        )

        layout_transpose = [1, 3, 2, 0]  # (b, s, h, d) -> (s, d, h, b)
        mQ, mK, mV, mdO = [
            layout_utils.select(t, mode=layout_transpose) for t in (mQ, mK, mV, mdO)
        ]
        # No (d, s, h, b) views here on purpose. The dV / dK / dQ gemms take dO / Q / K
        # as MN-major operands, i.e. their SMEM tile is the transpose of the (s, d) tile
        # the S / dP gemms use -- but only the LOGICAL shape is transposed, the bytes are
        # identical (see _setup_smem_layout). So one TMA atom per tensor fills the buffer
        # for both gemms and the second one reads it through the transposed view.

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            tiled_mma_S,
            self.cluster_shape_mnk.shape,
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            tiled_mma_S,
            self.cluster_shape_mnk.shape,
        )
        # dP^T = V @ dO^T: V is the A operand, dO^T the B operand, both chunked
        # over head_dim_v.
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            tiled_mma_dP,
            self.cluster_shape_mnk.shape,
        )
        tma_atom_dOt, tma_tensor_dOt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mdO,
            cute.select(self.sdOt_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            tiled_mma_dP,
            self.cluster_shape_mnk.shape,
        )
        # dV_c = P^T @ dO_c, dK_c = dS^T @ Q_c and dQ_c = dS @ K_c read dO / Q / K
        # through their MN-major views, which are byte-identical to the K-major ones
        # above, so they need no atoms and no reload of their own.
        self.tma_copy_bytes = {
            name: self.cta_group_size
            * cute.size_in_bytes(dtype, cute.select(layout, mode=[0, 1, 2]))
            for name, dtype, layout in [
                ("Q", self.q_dtype, self.sQ_layout),
                ("K", self.k_dtype, self.sK_layout),
                ("V", self.v_dtype, self.sV_layout),
                ("dOt", self.do_dtype, self.sdOt_layout),
            ]
        }

        num_n_block = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
        grid_dim = (num_n_block, cute.size(mQ.shape[2]), cute.size(mK.shape[3]))
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dOt,
            mLSE,
            mdPsum,
            mFM,
            mdQaccum,
            mdKaccum,
            mdVaccum,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dOt,
            tiled_mma_S,
            tiled_mma_dP,
            tiled_mma_dV,
            tiled_mma_dK,
            tiled_mma_dQ,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sdOt_layout,
            self.sdO_layout,
            self.sQt_layout,
            self.tP_layout,
            self.tdS_layout,
            self.sdS_layout,
            self.sdSt_layout,
            self.sKt_layout,
            softmax_scale_log2,
            overlap_work_done_addr,
            overlap_segment_idx,
            overlap_comm_rpb,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mFM: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        mdKaccum: cute.Tensor,
        mdVaccum: cute.Tensor,
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dOt: cute.CopyAtom,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        softmax_scale_log2: Float32,
        overlap_work_done_addr: Optional[cutlass.Int64],
        overlap_segment_idx: Optional[Int32],
        overlap_comm_rpb: cutlass.Constexpr,
    ):
        """Run the tiled BigD backward dataflow for one scheduled KV block.

        Tensor arguments provide global inputs and fp32 gradient accumulators; TMA
        atoms, MMA descriptors, and layouts define the compile-time transfer and
        compute mappings. The kernel writes dQ, dK, and dV through the accumulator
        tensors and has no Python return value.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        n_block, head_idx, batch_idx = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar_ptr = storage.mbar_ptr.data_ptr()
        # Every barrier below is used exactly ONCE per m iteration, which is what lets a
        # single phase bit per warp track all of them (see the m loops: iteration j waits
        # on parity j & 1).
        # K and Q for one d chunk arrive on the same barrier (same producer, same
        # consuming gemm), so one full/empty pair per d chunk.
        mbar_Sin_full = mbar_ptr + 0
        mbar_Sin_empty = mbar_Sin_full + self.num_d_chunks
        mbar_S_full = mbar_Sin_empty + self.num_d_chunks
        # V and dO^T for one dv chunk arrive on the same barrier (same producer,
        # consumed together by the dP gemm), so one full/empty pair per dv chunk.
        mbar_dPin_full = mbar_S_full + 1
        mbar_dPin_empty = mbar_dPin_full + self.num_dv_chunks
        mbar_dP_full = mbar_dPin_empty + self.num_dv_chunks
        # P/dS written back to TMEM by the compute warps (bf16, upper half of the
        # S/dP regions), then the three output gemms. Each edge gets its own
        # barrier so that the one-arrival-per-iteration rule above holds.
        mbar_PdS_full = mbar_dP_full + 1
        mbar_dVin_full = mbar_PdS_full + 1
        mbar_dVin_empty = mbar_dVin_full + self.num_dv_chunks
        mbar_dKin_full = mbar_dVin_empty + self.num_dv_chunks
        mbar_dKin_empty = mbar_dKin_full + self.num_d_chunks
        # dS also has to reach SMEM: the dQ gemm needs it in the (m, n) orientation,
        # which TMEM cannot provide (its A operand is the (n, m) accumulator). One
        # buffer, two views -- write through sdSt (n, m), read through sdS (m, n).
        mbar_dSsmem_full = mbar_dKin_empty + self.num_d_chunks
        mbar_dQin_full = mbar_dSsmem_full + 1
        mbar_dQin_empty = mbar_dQin_full + self.num_d_chunks
        mbar_out_full = mbar_dQin_empty + self.num_d_chunks
        num_out_chunks = cutlass.const_expr(self.num_out_chunks)
        mbar_out_empty = mbar_out_full + num_out_chunks
        # K only depends on the n block, so the chunks the steady-state reuse schedule
        # expects to already be resident when the m loop starts are fetched once, here.
        mbar_Kpre_full = mbar_out_empty + num_out_chunks
        # LSE and dPsum for one m tile, staged in SMEM by the load warp. The compute
        # warps used to read them straight from gmem, per element: S^T is (n, m), so the
        # m coordinate is the per-ELEMENT one and every thread walked all tile_m values
        # of both tensors once per m tile -- ~2 * tile_m dependent scalar loads per
        # thread, with all 32 lanes of a warp asking for the identical value. Two bulk
        # copies per m tile replace them.
        mbar_stats_full = mbar_Kpre_full + 1
        mbar_stats_empty = mbar_stats_full + 1
        mbar_tmem_dealloc = mbar_ptr + self.mbar_tmem_dealloc_offset
        # The pipeline slots above have to end exactly where tmem_dealloc starts, or
        # mbar_count() and this layout have drifted apart and some barrier is aliasing
        # another.
        pipeline_slots = (
            2 * self.num_d_chunks       # Sin full / empty
            + 1                         # S full
            + 2 * self.num_dv_chunks    # dPin full / empty
            + 1                         # dP full
            + 1                         # PdS full
            + 2 * self.num_dv_chunks    # dVin full / empty
            + 2 * self.num_d_chunks     # dKin full / empty
            + 1                         # dSsmem full
            + 2 * self.num_d_chunks     # dQin full / empty
            + 2 * num_out_chunks        # out full / empty
            + 1                         # K prologue full
            + 2                         # stats full / empty
        )
        assert pipeline_slots == self.mbar_tmem_dealloc_offset, (
            f"kernel uses {pipeline_slots} pipeline barrier slots but mbar_count() "
            f"reserved {self.mbar_tmem_dealloc_offset}"
        )

        if warp_idx == 1:
            # *_in_empty and out_empty are arrived by the DRAIN warps (they own the
            # T2R and the SMEM operand release); PdS_full / dSsmem_full stay with the
            # compute warps; tmem_dealloc needs both, since compute reads S / dP out
            # of TMEM and the drain reads the output scratch.
            num_drain_threads = cutlass.const_expr(
                cute.arch.WARP_SIZE * len(self.drain_warp_ids)
            )
            num_compute_threads_init = cutlass.const_expr(
                cute.arch.WARP_SIZE * len(self.compute_warp_ids)
            )
            for c in cutlass.range_constexpr(self.num_d_chunks):
                cute.arch.mbarrier_init(mbar_Sin_full + c, 1)
                cute.arch.mbarrier_init(mbar_Sin_empty + c, 1)
                cute.arch.mbarrier_init(mbar_dKin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dKin_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(mbar_S_full, 1)
            for c in cutlass.range_constexpr(self.num_dv_chunks):
                cute.arch.mbarrier_init(mbar_dPin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dPin_empty + c, 1)
                cute.arch.mbarrier_init(mbar_dVin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dVin_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(mbar_dP_full, 1)
            for c in cutlass.range_constexpr(self.num_d_chunks):
                cute.arch.mbarrier_init(mbar_dQin_full + c, 1)
                cute.arch.mbarrier_init(mbar_dQin_empty + c, num_drain_threads)
            cute.arch.mbarrier_init(mbar_dSsmem_full, num_compute_threads_init)
            # every compute thread arrives on PdS_full; every drain thread on out_empty
            cute.arch.mbarrier_init(mbar_PdS_full, num_compute_threads_init)
            # Under the split drain BOTH warpgroups read each output chunk (the
            # drain warps take the even slices, the compute warps the odd ones), so
            # out_empty completes only when every thread of both has arrived.
            num_out_empty_arrivals = cutlass.const_expr(
                num_drain_threads
                + (num_compute_threads_init if self.drain_split else 0)
            )
            for c in cutlass.range_constexpr(num_out_chunks):
                cute.arch.mbarrier_init(mbar_out_full + c, 1)
                cute.arch.mbarrier_init(mbar_out_empty + c, num_out_empty_arrivals)
            cute.arch.mbarrier_init(mbar_Kpre_full, 1)
            # stats_full: one arrival from the load warp's elected lane, plus the bytes
            # of both bulk copies. stats_empty: every compute thread, once it is done
            # reading the tile -- the buffers are single, so the next m tile's fetch has
            # to wait for that.
            cute.arch.mbarrier_init(mbar_stats_full, 1)
            cute.arch.mbarrier_init(mbar_stats_empty, num_compute_threads_init)
            cute.arch.mbarrier_init(
                mbar_tmem_dealloc, num_compute_threads_init + num_drain_threads
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if cutlass.const_expr(self.kv_shared):
            # sV has no storage of its own (make_shared_storage): the fused S + dP pass
            # below reads V out of the sK stage the S gemm just used, indexed by that
            # stage instead of by sV's sliced-off stage mode.
            sV = sK
        else:
            sV = storage.sV.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=self.q_dtype
            )
        sdOt = storage.sdO.get_tensor(
            sdOt_layout.outer, swizzle=sdOt_layout.inner, dtype=self.do_dtype
        )
        # Second views of the same buffer, used by the gemm that reads the tile after
        # the S / dP gemm is done with it: dV wants dO as (dv_chunk, tile_m) and dK
        # wants Q^T. Same bytes, so no reload (see _setup_smem_layout).
        sdO = storage.sdO.get_tensor(
            sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype
        )
        sQt = storage.sQ.get_tensor(
            sQt_layout.outer, swizzle=sQt_layout.inner, dtype=self.q_dtype
        )

        # TMEM: S^T at column 0, dP^T right after it. We always allocate all 512
        # columns starting at 0, so a fake pointer at 0 addresses them (fwd trick).
        thr_mma_S = tiled_mma_S.get_slice(0)
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        tStS_frag = thr_mma_S.make_fragment_C(
            thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])
        )
        tdPtdP_frag = thr_mma_dP.make_fragment_C(
            thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        )
        tmem_ptr = cute.make_ptr(
            Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16
        )
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS_frag.layout)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP_frag.layout)
        # The bf16 P / dS views of the upper halves of these two regions are built per
        # softmax chunk in the compute branch (tStP_c / tStdS_c); the MMA reaches them
        # through tA_addr, so there is no full-tile view here.
        # dS in SMEM: sdSt is the natural (n, m) orientation the compute warps
        # produce, sdS the (m, n) view the dQ gemm needs as its A operand. Same
        # bytes -- this is the dual-view trick flash_bwd_sm100 uses for sdSt / sdS.
        sdSt = storage.sdS.get_tensor(
            sdSt_layout.outer, swizzle=sdSt_layout.inner, dtype=self.ds_dtype
        )
        sdS = cute.make_tensor(
            cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer
        )
        sKt = storage.sK.get_tensor(
            sKt_layout.outer, swizzle=sKt_layout.inner, dtype=self.k_dtype
        )
        # Flashmask metadata reduced before deriving the m-block skip ranges.
        #
        # Built HERE, at the kernel's top level, and not inside the warp regions: the
        # DSL's AST pass treats `obj.method(...)` as a write to `obj` and threads that
        # object through every enclosing region, and a SharedStorage instance cannot be
        # flattened into a region argument ("unable to convert SharedStorage to
        # Numeric"). Every storage.*.get_tensor() call therefore has to stay out here.
        # The layout is built here rather than reused from a host-side attribute: a
        # layout created in the host-side jit function is an SSA value of *that*
        # region, and kernel regions are isolated ("'cute.make_view' op using value
        # defined outside the region"). Layouts either come in as kernel parameters
        # or are constructed inside the kernel.
        sFM_red = storage.sFM_max_min_ptr.get_tensor(
            cute.make_layout(FLASHMASK_META_SLOTS)
        )
        # LSE / dPsum for the current m tile. One buffer each (the m loop is serialised
        # on them through mbar_stats_full / mbar_stats_empty), so a flat tile_m view is
        # all that is needed -- the (tile_m, stage) layouts in _setup_smem_layout only
        # exist to size the allocation.
        sLSE = storage.sLSE.get_tensor(cute.make_layout(self.tile_m))
        sdPsum = storage.sdPsum.get_tensor(cute.make_layout(self.tile_m))
        if cutlass.const_expr(self.kv_shared and not self.deterministic):
            # (elements of one slice, slot), living in sV's storage -- sV is dead under
            # kv_shared. Flat on purpose: the bulk reduce-add moves the slot verbatim,
            # so the SMEM order IS the accumulator's gmem order. deterministic drains
            # with per-thread atomics instead, so it gets no staging (out_stage == 0).
            sOutAccum = storage.sV.get_tensor(
                cute.make_layout((self.out_stage_elems, self.out_stage)),
                dtype=Float32,
            )
        # A-operand views for the output gemms (address comes from tA_addr).
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        tP = cute.make_tensor(tmem_ptr, tP_layout.outer)
        tdS = cute.make_tensor(tmem_ptr, tdS_layout.outer)
        # One view per output scratch slot; output chunk c uses slot
        # c % num_out_slots. Same layout in every slot, so the T2R copy atoms built
        # from slot 0 apply to all of them.
        _slot_base = lambda s: (
            tmem_ptr + self.tmem_out_offset + s * self.tmem_out_slot_cols
        )
        tdVtdV_slots = tuple(
            cute.make_tensor(
                _slot_base(s),
                thr_mma_dV.make_fragment_C(
                    thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
                ).layout,
            )
            for s in range(self.num_out_slots)
        )
        tdQtdQ_slots = tuple(
            cute.make_tensor(
                _slot_base(s),
                thr_mma_dQ.make_fragment_C(
                    thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
                ).layout,
            )
            for s in range(self.num_out_slots)
        )
        tdKtdK_slots = tuple(
            cute.make_tensor(
                _slot_base(s),
                thr_mma_dK.make_fragment_C(
                    thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
                ).layout,
            )
            for s in range(self.num_out_slots)
        )

        # The m loop: every barrier below is used exactly once per m iteration, so a
        # single phase bit per warp tracks all of them (mbarrier phases flip on each
        # completion; iteration j waits phase j & 1).
        seqlen_q = cute.size(mQ.shape[0])
        seqlen_k = cute.size(mK.shape[0])
        num_m_block = cute.ceil_div(seqlen_q, self.tile_m)

        # GQA / MQA: the grid's y is a *query* head; K, V and the dK / dV
        # accumulators are indexed by its kv head. Several query heads therefore
        # add into the same dK / dV rows, which the atomic reduce already handles.
        head_idx_kv = head_idx // self.qhead_per_kvhead

        # ---------------------------------------------------------------- m skip range
        # An m block whose every element is masked contributes nothing (P == 0 => dV, and
        # dS = P * (dP - dPsum) == 0 => dK, dQ), so the m loop walks
        #     [m_lo, m_hi) minus two contiguous bands [a_lo, a_hi) and [b_lo, b_hi)
        # instead of every block, with a_hi <= b_lo. The lower tail gives one band and,
        # for the 4-bound form, the upper tail the other. The per-element mask further
        # down stays the source of truth, which makes these bounds only need to be
        # CONSERVATIVE: skipping less than possible costs time, skipping a block that
        # still has an unmasked element is a wrong answer.
        #
        # At cta_group=2 the two CTAs of a pair own adjacent key blocks but share every
        # collective operation (the gemms below take cta_group=2, the loads are
        # multicast), so they must run the SAME number of m iterations. The range is
        # therefore computed over the pair's whole key range -- both CTAs read the same
        # bounds and apply the same expressions, so they agree by construction and no
        # cross-CTA reduction is needed. cta_group=1 keeps its exact previous range.
        n_block_pair = (n_block // self.cta_group_size) * self.cta_group_size
        m_lo = Int32(0)
        m_hi = num_m_block
        # Both bands start empty at the far end, which leaves [m_lo, m_hi) whole.
        a_lo = num_m_block
        a_hi = num_m_block
        b_lo = num_m_block
        b_hi = num_m_block
        if cutlass.const_expr(self.is_causal):
            # keep n_global <= m_global + seqlen_k - seqlen_q, so the first m block that
            # can hold an unmasked element is (n_lo + seqlen_q - seqlen_k) / tile_m --
            # the identical expression to block_info.py's get_m_block_min_max, evaluated
            # at the pair's lowest key (the least constraining of the two).
            m_lo = cutlass.max(
                Int32(0),
                (n_block_pair * self.tile_n + seqlen_q - seqlen_k) // self.tile_m,
            )
        if cutlass.const_expr(self.fm_bound_num > 0):
            fm_heads = cute.size(mFM.shape[1])
            fm_b = batch_idx if cute.size(mFM.shape[0]) > 1 else Int32(0)
            fm_h = head_idx // (cute.size(mQ.shape[2]) // fm_heads)
            # mFM keeps its full seqlen_k rows while overlap makes seqlen_k a segment
            # length, so every absolute key row read is offset by the segment. The
            # bounds themselves are QUERY rows and are never segmented.
            fm_row_offset = (
                overlap_segment_idx * seqlen_k
                if cutlass.const_expr(overlap_segment_idx is not None)
                else Int32(0)
            )
        if cutlass.const_expr(self.fm_bound_num > 0):
            # Reduce this key pair's per-column bounds to a few scalars. One warp does it
            # (32 lanes x 4 columns + a butterfly reduce, so every lane ends up with the
            # result and can write the same value), then the CTA barrier publishes them to
            # the load / mma / compute warps -- they must all see the SAME m range or they
            # deadlock, so this is computed once and read, never recomputed per warp.
            #
            # Two groups of scalars come out of it:
            #   - max(start) and min(end) of each tail give the fully masked bands, i.e.
            #     the m skip range below (only for the widths that range supports);
            #   - min / max of every bound column give the unmasked test of
            #     fm_need_mask in the compute warp, which lets an m block that no key of
            #     the pair masks skip the per-element flashmask predicate. That test only
            #     changes how a visited block is computed, never which blocks are visited
            #     or how many iterations run, so it is safe under `deterministic` and for
            #     all widths, unlike the skip range.
            assert self.tile_n % cute.arch.WARP_SIZE == 0
            if warp_idx == self.load_warp_id:
                lane = tidx % cute.arch.WARP_SIZE
                int_max = Int32(2**31 - 1)
                int_min = Int32(-(2**31))
                acc_ds = Int32(0)
                acc_end = int_max
                acc_us = Int32(0)
                acc_ue = int_max
                min_c0 = int_max
                max_c1 = int_min
                min_c2 = int_max
                max_c3 = int_min
                pair_cols = self.cta_group_size * self.tile_n
                for j in cutlass.range_constexpr(pair_cols // cute.arch.WARP_SIZE):
                    col = n_block_pair * self.tile_n + j * cute.arch.WARP_SIZE + lane
                    # Columns past seqlen_k are masked for every m by n_oob, so they must
                    # not constrain anything: each reduction uses its neutral element
                    # there. The same clamp covers a pair whose second key block falls
                    # entirely past seqlen_k.
                    in_range = col < seqlen_k
                    safe_col = fm_row_offset + cutlass.min(col, seqlen_k - 1)
                    c0 = mFM[fm_b, fm_h, safe_col, 0]
                    acc_ds = cutlass.max(acc_ds, c0 if in_range else Int32(0))
                    min_c0 = cutlass.min(min_c0, c0 if in_range else int_max)
                    if cutlass.const_expr(self.fm_bound_num >= 2):
                        c1 = mFM[fm_b, fm_h, safe_col, 1]
                        acc_end = cutlass.min(acc_end, c1 if in_range else int_max)
                        max_c1 = cutlass.max(max_c1, c1 if in_range else int_min)
                    if cutlass.const_expr(self.fm_bound_num == 4):
                        c2 = mFM[fm_b, fm_h, safe_col, 2]
                        c3 = mFM[fm_b, fm_h, safe_col, 3]
                        acc_us = cutlass.max(acc_us, c2 if in_range else Int32(0))
                        acc_ue = cutlass.min(acc_ue, c3 if in_range else int_max)
                        min_c2 = cutlass.min(min_c2, c2 if in_range else int_max)
                        max_c3 = cutlass.max(max_c3, c3 if in_range else int_min)
                acc_ds = utils.warp_reduce(acc_ds, lambda a, b: cutlass.max(a, b))
                acc_end = utils.warp_reduce(acc_end, lambda a, b: cutlass.min(a, b))
                min_c0 = utils.warp_reduce(min_c0, lambda a, b: cutlass.min(a, b))
                max_c1 = utils.warp_reduce(max_c1, lambda a, b: cutlass.max(a, b))
                sFM_red[0] = acc_ds
                sFM_red[1] = acc_end
                sFM_red[4] = min_c0
                sFM_red[5] = max_c1
                if cutlass.const_expr(self.fm_bound_num == 4):
                    acc_us = utils.warp_reduce(acc_us, lambda a, b: cutlass.max(a, b))
                    acc_ue = utils.warp_reduce(acc_ue, lambda a, b: cutlass.min(a, b))
                    min_c2 = utils.warp_reduce(min_c2, lambda a, b: cutlass.min(a, b))
                    max_c3 = utils.warp_reduce(max_c3, lambda a, b: cutlass.max(a, b))
                    sFM_red[2] = acc_us
                    sFM_red[3] = acc_ue
                    sFM_red[6] = min_c2
                    sFM_red[7] = max_c3
            cute.arch.barrier()
            # Bounds of the union of every key's masked rows, per band, for the unmasked
            # test. The semantics are the per-element mask's (has_end below):
            #   lower band [c0, c1) if has_end else [c0, seqlen_q)
            #   upper band (non-causal) [c2, c3) if has_end else [0, c1)
            # An m block [m0, m1) is untouched by a band if m1 <= its min start or
            # m0 >= its max end. Mixed cases (some keys below, some above) count as
            # touched, which is conservative.
            fm_has_end = cutlass.const_expr(
                (self.is_causal and self.fm_bound_num == 2)
                or (not self.is_causal and self.fm_bound_num == 4)
            )
            fm_lo_start_min = sFM_red[4]
            fm_lo_end_max = sFM_red[5] if cutlass.const_expr(fm_has_end) else seqlen_q
            if cutlass.const_expr(not self.is_causal):
                if cutlass.const_expr(fm_has_end):
                    fm_up_start_min, fm_up_end_max = sFM_red[6], sFM_red[7]
                else:
                    fm_up_start_min, fm_up_end_max = Int32(0), sFM_red[5]
        if cutlass.const_expr(self.fm_bound_num in (1, 2, 4) and not self.deterministic):
            # `deterministic` excludes itself from this narrowing on purpose. The dQ
            # semaphore uses `lock_value = n_block`, which is only a valid arrival count
            # if the set of CTAs that write a given m block is a PREFIX {0..N} of the n
            # axis. The causal m_lo above is monotone in n_block, so it keeps that
            # property; the flashmask-derived m_lo / m_hi / bands below are not, and a hole
            # in the sequence hangs the counter forever. Giving up the skip only costs
            # time -- the per-element mask still zeroes the masked-out elements.
            #
            # TODO: the skip and the counter are not actually exclusive. The small-headdim
            # sm100 kernel keeps both (see flash_bwd_sm100.py's dQ reduce loop, "Walk every
            # block but reduce only the ones the mma warp produced"): its drain loop walks
            # EVERY m block as a predicate and still runs the semaphore handshake for the
            # blocks it skips, so the arrival counts stay dense. flash_bwd_sm90.py does the
            # same catch-up for local masking. Porting that here would restore the skip for
            # deterministic; it means turning this drain's segment walk into a
            # walk-all-plus-predicate loop, so it is left as a follow-up.
            #
            # fm_bound_num == 4 (non-causal, both tails bounded) reduces four scalars --
            # the max start and min end of each tail -- and drops two bands, one per
            # tail; the three-segment walk below expresses both.
            #
            # max(start) and min(end) per tail were reduced and published above.
            max_ds = sFM_red[0]
            min_end = sFM_red[1]
            # Which m blocks are fully masked, from the reference semantics in
            # generate_startend_row_indices.py, evaluated for ALL columns at once
            # via max(start) / min(end):
            #   lower tail [ds, de) or [ds, seqlen_q)  -> block j masked if max_ds <= j*tm
            #   upper tail [0, ue)                     -> block j masked if (j+1)*tm <= min_ue
            #   upper tail [us, ue)                    -> both conditions, on us and ue
            first_masked = cute.ceil_div(max_ds, self.tile_m)
            if cutlass.const_expr(self.is_causal):
                if cutlass.const_expr(self.fm_bound_num == 2):
                    # band [ds, de): a middle range of m blocks drops out
                    a_lo = first_masked
                    a_hi = cutlass.max(a_lo, min_end // self.tile_m)
                else:
                    # band [ds, seqlen_q): everything from first_masked on drops out
                    m_hi = cutlass.min(m_hi, first_masked)
            elif cutlass.const_expr(self.fm_bound_num == 4):
                # Two bands, [ds, de) and [us, ue), each a middle range of m blocks.
                # As in flash_bwd_sm100.py's fm_skip_info: order them by their start and
                # merge them when they touch, so that a_hi <= b_lo. A block covered by
                # the lower tail for some keys and the upper tail for the others is
                # fully masked but kept, which is conservative.
                l_lo = first_masked
                l_hi = cutlass.max(l_lo, min_end // self.tile_m)
                u_lo = cute.ceil_div(sFM_red[2], self.tile_m)
                u_hi = cutlass.max(u_lo, sFM_red[3] // self.tile_m)
                u_first = u_lo <= l_lo
                a_lo = u_lo if u_first else l_lo
                a_hi = u_hi if u_first else l_hi
                b_lo = l_lo if u_first else u_lo
                b_hi = l_hi if u_first else u_hi
                b_lo = cutlass.max(b_lo, a_hi)
                b_hi = cutlass.max(b_hi, b_lo)
            else:
                # [0, ue) at the front and [ds, seqlen_q) at the back. Conservative:
                # different columns may be covered by different clauses, which two
                # scalars cannot express, so only the two ends are trimmed.
                m_lo = cutlass.max(m_lo, min_end // self.tile_m)
                m_hi = cutlass.min(m_hi, first_masked)
        # Iteration space: three segments, below band a, between the bands, and above
        # band b. a_hi >= a_lo, b_lo >= a_hi and b_hi >= b_lo hold by construction, so
        # the segments can neither overlap nor leave a gap; with band b empty at the far
        # end this is exactly the previous two-segment walk.
        n1 = cutlass.max(Int32(0), cutlass.min(a_lo, m_hi) - m_lo)
        seg2_base = cutlass.max(m_lo, a_hi)
        n2 = cutlass.max(Int32(0), cutlass.min(b_lo, m_hi) - seg2_base)
        seg3_base = cutlass.max(seg2_base, b_hi)
        num_iters = n1 + n2 + cutlass.max(Int32(0), m_hi - seg3_base)

        # gmem tiles: mK/mQ/mV/mdO are (s, d, h, b) after the transpose.
        mK_cur = mK[None, None, head_idx_kv, batch_idx]
        mQ_cur = mQ[None, None, head_idx, batch_idx]
        mV_cur = mV[None, None, head_idx_kv, batch_idx]
        mdO_cur = mdO[None, None, head_idx, batch_idx]
        gK = cute.local_tile(
            mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, None)
        )
        tSgK = thr_mma_S.partition_A(gK)

        if warp_idx == self.load_warp_id:
            # The register budget is per-warp state, not per-iteration work: setting it
            # inside the m loop re-issues setmaxnreg on every iteration.
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            # Every gmem read of the gathered KV happens in this warp, so one gate
            # ahead of the K prologue covers the whole CTA. There is no tile scheduler
            # here: a CTA owns one n block for its lifetime.
            if cutlass.const_expr(overlap_work_done_addr is not None):
                gate_batch_idx = batch_idx
                if cutlass.const_expr(self.overlap_bhsd_layout):
                    gate_batch_idx = batch_idx * cute.size(mK.shape[2]) + head_idx_kv
                _overlap_gate_bwd(
                    n_block,
                    tidx % cute.arch.WARP_SIZE,
                    seqlen_k,
                    gate_batch_idx,
                    cute.make_ptr(
                        Int32,
                        overlap_work_done_addr,
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    overlap_comm_rpb,
                    self.cta_group_size,
                    self.tile_n,
                )
            # K only depends on the n block, so one copy fn serves the whole kernel --
            # including the prologue, which puts the chunks the steady-state schedule
            # expects to be resident into place before the m loop.
            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), tSgK, sK
            )
            K_pre = cutlass.const_expr(self.sched_K[2])
            if cutlass.const_expr(len(K_pre) > 0):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar_Kpre_full,
                        cutlass.const_expr(len(K_pre) * self.tma_copy_bytes["K"]),
                    )
                for i in cutlass.range_constexpr(len(K_pre)):
                    load_K(K_pre[i][0], K_pre[i][1], tma_bar_ptr=mbar_Kpre_full)
            # LSE / dPsum are contiguous tile_m runs of f32, so they need no TMA
            # descriptor: a plain bulk copy moves the whole tile on one mbarrier.
            copy_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            stats_bytes = cutlass.const_expr(2 * self.tile_m * Float32.width // 8)
            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # The iteration counter drives the barrier phases; m_iter is the actual
                # block index, which skips the fully masked bands (see the skip range).
                m_iter = _bigd_m_block(it, m_lo, n1, seg2_base, n2, seg3_base)

                def wait_after(rec, empty_bars):
                    """Wait for the gemm that last read the buffer this fetch overwrites.

                    ``rec["after"]`` is the (pass index, chunk) of that reader, which
                    _reuse_schedule derived from the buffer's access cycle;
                    ``empty_bars`` maps a pass index to its ``*_in_empty`` barrier base.
                    A hit overwrites nothing, so it waits for nothing.

                    The wrapped case has to be skipped on the first iteration:
                    mbarrier_wait spins on try_wait.parity, which only returns once the
                    phase with that parity has COMPLETED, so a parity-1 wait on a fresh
                    barrier blocks forever instead of falling through.
                    """
                    if cutlass.const_expr(not rec["load"]):
                        return
                    bar = empty_bars[rec["after"][0]] + rec["after"][1]
                    if cutlass.const_expr(rec["after_prev_iter"]):
                        if it > 0:
                            cute.arch.mbarrier_wait(bar, phase ^ 1)
                    else:
                        cute.arch.mbarrier_wait(bar, phase)

                def publish(bar, nbytes):
                    """Arm `bar` for a fetch of `nbytes`, or just complete it if 0."""
                    with cute.arch.elect_one():
                        if cutlass.const_expr(nbytes > 0):
                            cute.arch.mbarrier_arrive_and_expect_tx(bar, nbytes)
                        else:
                            cute.arch.mbarrier_arrive(bar)

                # LSE / dPsum first: they are the compute warps' first dependency after
                # S, and one bulk copy each is cheap enough to issue ahead of the
                # operand TMAs. Single buffer, so the previous tile's readers have to be
                # done -- on the first iteration that arrival does not exist yet (a
                # parity-1 wait on a fresh mbarrier never falls through).
                if it > 0:
                    cute.arch.mbarrier_wait(mbar_stats_empty, phase ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_stats_full, stats_bytes)
                    cute.copy(
                        copy_stats,
                        cute.local_tile(
                            mLSE[batch_idx, head_idx, None], (self.tile_m,), (m_iter,)
                        ),
                        sLSE,
                        mbar_ptr=mbar_stats_full,
                    )
                    cute.copy(
                        copy_stats,
                        cute.local_tile(
                            mdPsum[batch_idx, head_idx, None], (self.tile_m,), (m_iter,)
                        ),
                        sdPsum,
                        mbar_ptr=mbar_stats_full,
                    )
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), thr_mma_S.partition_B(
                        cute.local_tile(
                            mQ_cur,
                            cute.select(self.mma_tiler_kq, mode=[1, 2]),
                            (m_iter, None),
                        )
                    ), sQ
                )
                # kv_shared needs dO^T's copy fn in the S pass below, so it gets its own
                # definition here. The split path keeps building it where it always did
                # (with the dP pass): moving that construction earlier is semantically a
                # no-op but it shifts where the DSL emits the address math, and measured
                # a clear regression on the split path. Same reason tdPrdOt stays inside
                # each branch in the mma warp.
                if cutlass.const_expr(self.kv_shared):
                    load_dO, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_dOt, 0, cute.make_layout(1), thr_mma_dP.partition_B(
                            cute.local_tile(
                                mdO_cur,
                                cute.select(self.mma_tiler_vdo, mode=[1, 2]),
                                (m_iter, None),
                            )
                        ), sdOt
                    )
                # S pass: K (A) and Q (B) for one d chunk land on the same barrier, so
                # each chunk keeps one full/empty pair. Either fetch may be a no-op when
                # the buffer already holds the chunk (K carries across m iterations, and
                # the dK pass leaves Q's last chunk behind), and then the barrier is
                # simply arrived instead of armed for bytes.
                #
                # Under kv_shared dO_c rides that same barrier and the V pass below is
                # gone entirely: V IS K, so the mma warp fuses S_c and dP_c and reads V
                # out of the sK stage. One full/empty pair per chunk then covers all
                # three operands, and dPin_full / dPin_empty go unused.
                #
                # The per-chunk dO record is looked up from a tuple built here rather
                # than with a ternary inside the loop: a const_expr conditional in a
                # range_constexpr body is not legal DSL.
                sched_dO_S = cutlass.const_expr(
                    self.sched_dO[0] if self.kv_shared else (None,) * self.num_d_chunks
                )
                for pos in cutlass.range_constexpr(self.num_d_chunks):
                    k = cutlass.const_expr(self.sched_K[0][pos])
                    q = cutlass.const_expr(self.sched_Q[0][pos])
                    assert k["chunk"] == q["chunk"]
                    c = cutlass.const_expr(k["chunk"])
                    o = cutlass.const_expr(sched_dO_S[pos])
                    wait_after(k, (mbar_Sin_empty, mbar_dQin_empty))
                    wait_after(q, (mbar_Sin_empty, mbar_dKin_empty))
                    if cutlass.const_expr(o is not None):
                        assert o["chunk"] == c, (
                            "the fused S + dP pass needs K, Q and dO to walk the chunks "
                            "in the same order"
                        )
                        # dO's pass-0 reader is the fused gemm now, so its release is
                        # Sin_empty; pass 1 is still the dV gemm.
                        wait_after(o, (mbar_Sin_empty, mbar_dVin_empty))
                    publish(
                        mbar_Sin_full + c,
                        cutlass.const_expr(
                            (self.tma_copy_bytes["K"] if k["load"] else 0)
                            + (self.tma_copy_bytes["Q"] if q["load"] else 0)
                            + (
                                self.tma_copy_bytes["dOt"]
                                if (o is not None and o["load"])
                                else 0
                            )
                        ),
                    )
                    if cutlass.const_expr(k["load"]):
                        load_K(c, k["stage"], tma_bar_ptr=mbar_Sin_full + c)
                    if cutlass.const_expr(q["load"]):
                        load_Q(c, q["stage"], tma_bar_ptr=mbar_Sin_full + c)
                    if cutlass.const_expr(o is not None and o["load"]):
                        load_dO(c, o["stage"], tma_bar_ptr=mbar_Sin_full + c)
                # V + dO^T: one dv chunk at a time, both into their single stage.
                # sV has its stage mode sliced off (rank 3), so cpasync.tma_partition
                # needs a gmem tile of matching rank: bake the chunk index into
                # local_tile and use single_stage=True, one copy fn per chunk (the
                # existing kernel does the same with load_V_low / load_V_high). sdOt
                # keeps its stage mode, so it uses the indexed form like Q.
                #
                # kv_shared skips this pass: dO went out with the S pass and V is the
                # sK stage the S gemm already loaded.
                if cutlass.const_expr(not self.kv_shared):
                    load_dO, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_dOt, 0, cute.make_layout(1), thr_mma_dP.partition_B(
                            cute.local_tile(
                                mdO_cur,
                                cute.select(self.mma_tiler_vdo, mode=[1, 2]),
                                (m_iter, None),
                            )
                        ), sdOt
                    )
                    for pos in cutlass.range_constexpr(self.num_dv_chunks):
                        o = cutlass.const_expr(self.sched_dO[0][pos])
                        c = cutlass.const_expr(o["chunk"])
                        gV_c = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_vdo, mode=[0, 2]),
                            (n_block, c),
                        )
                        load_V_c, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            thr_mma_dP.partition_A(gV_c),
                            sV,
                            single_stage=True,
                        )
                        # sV is a single buffer with one consumer, so its own gate is the
                        # previous chunk's dP gemm; sdOt is what the schedule tracks (its
                        # buffer is shared with the dV gemm's sdO view).
                        if cutlass.const_expr(c > 0):
                            cute.arch.mbarrier_wait(mbar_dPin_empty + (c - 1), phase)
                        else:
                            if it > 0:
                                cute.arch.mbarrier_wait(
                                    mbar_dPin_empty + (self.num_dv_chunks - 1), phase ^ 1
                                )
                        wait_after(o, (mbar_dPin_empty, mbar_dVin_empty))
                        publish(
                            mbar_dPin_full + c,
                            cutlass.const_expr(
                                self.tma_copy_bytes["V"]
                                + (self.tma_copy_bytes["dOt"] if o["load"] else 0)
                            ),
                        )
                        load_V_c(tma_bar_ptr=mbar_dPin_full + c)
                        if cutlass.const_expr(o["load"]):
                            load_dO(c, o["stage"], tma_bar_ptr=mbar_dPin_full + c)

                # dV pass: dO again, through its MN-major view. Its first chunk is the
                # one the dP pass just left in SMEM, so that fetch is gone.
                #
                # Under kv_shared the dV and dK passes are FUSED, one chunk of each per
                # step: the merged output gemm needs dO_c and Q_c together before it can
                # commit chunk c, and dO has a single stage, so publishing the whole dO
                # pass first would wait on a dVin_empty that only arrives after the merged
                # gemm -- which is still waiting for a Q chunk from the pass below.
                if cutlass.const_expr(self.kv_shared):
                    for pos in cutlass.range_constexpr(self.num_d_chunks):
                        o = cutlass.const_expr(self.sched_dO[1][pos])
                        q = cutlass.const_expr(self.sched_Q[1][pos])
                        assert o["chunk"] == q["chunk"], (
                            "kv_shared needs the dV and dK passes to walk the chunks in "
                            "the same order"
                        )
                        c = cutlass.const_expr(o["chunk"])
                        # dO's pass-0 reader is the fused S + dP gemm, so its release is
                        # Sin_empty (dPin_empty is unused under kv_shared).
                        wait_after(o, (mbar_Sin_empty, mbar_dVin_empty))
                        publish(
                            mbar_dVin_full + c,
                            cutlass.const_expr(
                                self.tma_copy_bytes["dOt"] if o["load"] else 0
                            ),
                        )
                        if cutlass.const_expr(o["load"]):
                            load_dO(c, o["stage"], tma_bar_ptr=mbar_dVin_full + c)
                        wait_after(q, (mbar_Sin_empty, mbar_dKin_empty))
                        publish(
                            mbar_dKin_full + c,
                            cutlass.const_expr(
                                self.tma_copy_bytes["Q"] if q["load"] else 0
                            ),
                        )
                        if cutlass.const_expr(q["load"]):
                            load_Q(c, q["stage"], tma_bar_ptr=mbar_dKin_full + c)
                else:
                    for pos in cutlass.range_constexpr(self.num_dv_chunks):
                        o = cutlass.const_expr(self.sched_dO[1][pos])
                        c = cutlass.const_expr(o["chunk"])
                        wait_after(o, (mbar_dPin_empty, mbar_dVin_empty))
                        publish(
                            mbar_dVin_full + c,
                            cutlass.const_expr(
                                self.tma_copy_bytes["dOt"] if o["load"] else 0
                            ),
                        )
                        if cutlass.const_expr(o["load"]):
                            load_dO(c, o["stage"], tma_bar_ptr=mbar_dVin_full + c)

                    # dK pass: Q again, through its MN-major view (sQt).
                    for pos in cutlass.range_constexpr(self.num_d_chunks):
                        q = cutlass.const_expr(self.sched_Q[1][pos])
                        c = cutlass.const_expr(q["chunk"])
                        wait_after(q, (mbar_Sin_empty, mbar_dKin_empty))
                        publish(
                            mbar_dKin_full + c,
                            cutlass.const_expr(
                                self.tma_copy_bytes["Q"] if q["load"] else 0
                            ),
                        )
                        if cutlass.const_expr(q["load"]):
                            load_Q(c, q["stage"], tma_bar_ptr=mbar_dKin_full + c)

                # dQ pass: K again, through its MN-major view (sKt). K does not depend on
                # m, so with the descending order this pass and the S pass together fetch
                # 4 of the 8 chunk accesses at two stages instead of all 8.
                for pos in cutlass.range_constexpr(self.num_d_chunks):
                    k = cutlass.const_expr(self.sched_K[1][pos])
                    c = cutlass.const_expr(k["chunk"])
                    wait_after(k, (mbar_Sin_empty, mbar_dQin_empty))
                    publish(
                        mbar_dQin_full + c,
                        cutlass.const_expr(self.tma_copy_bytes["K"] if k["load"] else 0),
                    )
                    if cutlass.const_expr(k["load"]):
                        load_K(c, k["stage"], tma_bar_ptr=mbar_dQin_full + c)

                phase ^= 1
        elif warp_idx == self.mma_warp_id:
            # TMEM is allocated ONCE per CTA, not once per m iteration. Allocating
            # inside the loop asked for a second 512-column region while the first was
            # still held (and after relinquish_tmem_alloc_permit, which forbids any
            # further allocation) -- the second allocation returned an address the
            # tcgen05 MMAs then wrote through, which is the illegal access that showed
            # up as cudaErrorLaunchFailure once num_m_block > 1.
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            cute.arch.alloc_tmem(Int32(self.tmem_alloc_cols), storage.tmem_holding_buf)
            cute.arch.sync_warp()
            cute.arch.relinquish_tmem_alloc_permit()
            # The K chunks fetched before the m loop (see the load warp's prologue).
            if cutlass.const_expr(len(self.sched_K[2]) > 0):
                cute.arch.mbarrier_wait(mbar_Kpre_full, 0)

            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # The iteration counter drives the barrier phases; m_iter is the actual
                # block index, which skips the fully masked bands (see the skip range).
                m_iter = _bigd_m_block(it, m_lo, n1, seg2_base, n2, seg3_base)
                tSrK = tiled_mma_S.make_fragment_A(sK)
                tSrQ = tiled_mma_S.make_fragment_B(sQ)
                if cutlass.const_expr(self.kv_shared):
                    # S and dP FUSED, one chunk of each per step. V IS K here (same rows,
                    # dv_chunk == d_chunk, so mma_tiler_vdo == mma_tiler_kq), and both
                    # gemms contract over the whole head dim while sK holds only
                    # d_chunks_resident stages -- so dP has to read chunk c while the S
                    # gemm's stage is still live. That is what removes the V TMA and sV's
                    # whole tile from SMEM; the price is interleaving two accumulators.
                    tdPrV = tiled_mma_dP.make_fragment_A(sK)
                    tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
                    for pos in cutlass.range_constexpr(self.num_d_chunks):
                        k = cutlass.const_expr(self.sched_K[0][pos])
                        c = cutlass.const_expr(k["chunk"])
                        # One barrier per chunk, carrying K_c, Q_c AND dO_c.
                        cute.arch.mbarrier_wait(mbar_Sin_full + c, phase)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_S,
                            tStS,
                            tSrK,
                            tSrQ,
                            sA=sK,
                            sB=sQ,
                            A_idx=k["stage"],
                            B_idx=0,
                            zero_init=(pos == 0),
                            cta_group=self.cta_group_size,
                        )
                        if cutlass.const_expr(pos == self.num_d_chunks - 1):
                            # S is complete here; publishing before the dP gemm below
                            # keeps the compute warps' softmax overlapped with the last
                            # dP round, the way the split passes did.
                            with cute.arch.elect_one():
                                tcgen05.commit(mbar_S_full)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_dP,
                            tdPtdP,
                            tdPrV,
                            tdPrdOt,
                            sA=sK,
                            sB=sdOt,
                            A_idx=k["stage"],
                            B_idx=0,
                            zero_init=(pos == 0),
                            cta_group=self.cta_group_size,
                        )
                        with cute.arch.elect_one():
                            # This one pair covers K, Q and dO: all three are read by the
                            # two gemms above and free once they retire.
                            tcgen05.commit(mbar_Sin_empty + c)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_dP_full)
                else:
                    for pos in cutlass.range_constexpr(self.num_d_chunks):
                        k = cutlass.const_expr(self.sched_K[0][pos])
                        c = cutlass.const_expr(k["chunk"])
                        cute.arch.mbarrier_wait(mbar_Sin_full + c, phase)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_S,
                            tStS,
                            tSrK,
                            tSrQ,
                            sA=sK,
                            sB=sQ,
                            A_idx=k["stage"],
                            B_idx=0,
                            zero_init=(pos == 0),
                            cta_group=self.cta_group_size,
                        )
                        with cute.arch.elect_one():
                            # One commit per gemm: two back-to-back tcgen05.commit calls
                            # do not reliably attach both barriers to the same MMA, which
                            # deadlocked the K stream. K and Q for a chunk therefore share
                            # one full/empty pair (waiting on chunk c-1 is stricter than
                            # K's real need of c-2, which is harmless).
                            tcgen05.commit(mbar_Sin_empty + c)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_S_full)

                    # dP^T = sum_c V_c @ dO_c^T  (contraction over head_dim_v)
                    tdPrV = tiled_mma_dP.make_fragment_A(sV)
                    tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
                    for c in cutlass.range_constexpr(self.num_dv_chunks):
                        cute.arch.mbarrier_wait(mbar_dPin_full + c, phase)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_dP,
                            tdPtdP,
                            tdPrV,
                            tdPrdOt,
                            sA=sV,
                            sB=sdOt,
                            # sV had its stage mode sliced off, so its A fragment is rank
                            # 3 and takes no index; sdOt kept its (single) stage mode, so
                            # its B fragment must be indexed or gemm_ptx_partial's crd2idx
                            # on a rank-4 layout fails.
                            A_idx=None,
                            B_idx=0,
                            zero_init=(c == 0),
                            cta_group=self.cta_group_size,
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_dPin_empty + c)
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_dP_full)

                # The compute warps read S and dP out, then write P and dS back as
                # bf16 into the upper halves of those same regions.
                cute.arch.mbarrier_wait(mbar_PdS_full, phase)

                # Output chunk at issue position `pos` writes scratch slot
                # pos % num_out_slots, so the slot's previous user is the chunk
                # num_out_slots positions earlier -- in this iteration if that position
                # still exists, otherwise in the previous one (the sequence runs
                # continuously across m iterations, hence the phase flip; Python's
                # negative indexing picks the right chunk either way). The segments run
                # descending, so this is no longer the same as "out_c - num_out_slots".
                def wait_out_slot_free(pos):
                    prev = cutlass.const_expr(pos - self.num_out_slots)
                    prev_oc = cutlass.const_expr(self.out_issue[prev])
                    if cutlass.const_expr(prev >= 0):
                        cute.arch.mbarrier_wait(mbar_out_empty + prev_oc, phase)
                    else:
                        if it > 0:
                            cute.arch.mbarrier_wait(mbar_out_empty + prev_oc, phase ^ 1)

                # dV_c = P^T @ dO_c   (A = P from TMEM). Descending, like the loads and
                # the drain: chunk num-1 is the one the dP pass left in SMEM.
                tdVrP = tiled_mma_dV.make_fragment_A(tP)
                tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
                # dS comes from TMEM (tdS), addressed by tA_addr rather than sA.
                tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
                dK_a_kwargs = dict(sA=None, tA_addr=self.tmem_dS_offset)
                tdKrQt = tiled_mma_dK.make_fragment_B(sQt)
                if cutlass.const_expr(self.kv_shared):
                    # dKV_c = dS^T @ Q_c + P^T @ dO_c, both into the SAME slot: the dK
                    # gemm zero-inits it, the dV gemm accumulates on top (zero_init=False
                    # is a UMMA accumulate-in-place), so one commit and one drain cover
                    # both halves of the shared tensor's gradient. The two gemms keep
                    # their own A operands (P and dS, both in TMEM) and their
                    # own B operands
                    # (sQt and sdO), so no operand handling changes -- only the
                    # accumulator is shared, and both slot views address the same columns
                    # (equal chunk width, same M).
                    for pos_in_seg in cutlass.range_constexpr(self.num_d_chunks):
                        c = cutlass.const_expr(self.num_d_chunks - 1 - pos_in_seg)
                        out_c = cutlass.const_expr(self.out_base_dK + c)
                        pos = cutlass.const_expr(self.out_pos(out_c))
                        # Both operands of chunk c, which is why the load warp fuses its
                        # dV and dK passes under kv_shared: waiting for Q_c here while
                        # the load warp still had a whole dO pass to finish would
                        # deadlock on dO's single stage.
                        cute.arch.mbarrier_wait(mbar_dKin_full + c, phase)
                        cute.arch.mbarrier_wait(mbar_dVin_full + c, phase)
                        wait_out_slot_free(pos)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_dK,
                            tdKtdK_slots[pos % self.num_out_slots],
                            tdKrdS,
                            tdKrQt,
                            sB=sQt,
                            A_idx=None,
                            B_idx=0,
                            zero_init=True,
                            cta_group=self.cta_group_size,
                            **dK_a_kwargs,
                        )
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_dV,
                            tdVtdV_slots[pos % self.num_out_slots],
                            tdVrP,
                            tdVrdO,
                            sA=None,
                            sB=sdO,
                            A_idx=None,
                            B_idx=0,
                            zero_init=False,
                            tA_addr=self.tmem_P_offset,
                            cta_group=self.cta_group_size,
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_out_full + out_c)
                else:
                    for pos_in_seg in cutlass.range_constexpr(self.num_dv_chunks):
                        c = cutlass.const_expr(self.num_dv_chunks - 1 - pos_in_seg)
                        out_c = cutlass.const_expr(self.out_base_dV + c)
                        pos = cutlass.const_expr(self.out_pos(out_c))
                        cute.arch.mbarrier_wait(mbar_dVin_full + c, phase)
                        wait_out_slot_free(pos)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_dV,
                            tdVtdV_slots[pos % self.num_out_slots],
                            tdVrP,
                            tdVrdO,
                            sA=None,
                            sB=sdO,
                            A_idx=None,
                            B_idx=0,
                            zero_init=True,
                            tA_addr=self.tmem_P_offset,
                            cta_group=self.cta_group_size,
                        )
                        with cute.arch.elect_one():
                            # Exactly one commit per gemm: back-to-back commits in a
                            # single elect_one did not reliably arm both barriers (that
                            # deadlocked the K stream). The matching *_in_empty signals
                            # are arrived by the compute warps, which only get there
                            # after out_full fired.
                            tcgen05.commit(mbar_out_full + out_c)

                    # dK_c = dS^T @ Q_c   (A = dS from TMEM)
                    for pos_in_seg in cutlass.range_constexpr(self.num_d_chunks):
                        c = cutlass.const_expr(self.num_d_chunks - 1 - pos_in_seg)
                        out_c = cutlass.const_expr(self.out_base_dK + c)
                        pos = cutlass.const_expr(self.out_pos(out_c))
                        cute.arch.mbarrier_wait(mbar_dKin_full + c, phase)
                        wait_out_slot_free(pos)
                        sm100_utils.gemm_ptx_w_idx(
                            tiled_mma_dK,
                            tdKtdK_slots[pos % self.num_out_slots],
                            tdKrdS,
                            tdKrQt,
                            sB=sQt,
                            A_idx=None,
                            B_idx=0,
                            zero_init=True,
                            cta_group=self.cta_group_size,
                            **dK_a_kwargs,
                        )
                        with cute.arch.elect_one():
                            tcgen05.commit(mbar_out_full + out_c)

                # dQ_c = dS^T @ K_c  (A = dS from SMEM in the (m, n) view; M = tile_m =
                # 128, so the 32-datapath T2R applies to its accumulator too)
                cute.arch.mbarrier_wait(mbar_dSsmem_full, phase)
                tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
                tdQrKt = tiled_mma_dQ.make_fragment_B(sKt)
                for pos_in_seg in cutlass.range_constexpr(self.num_d_chunks):
                    k = cutlass.const_expr(self.sched_K[1][pos_in_seg])
                    c = cutlass.const_expr(k["chunk"])
                    out_c = cutlass.const_expr(self.out_base_dQ + c)
                    pos = cutlass.const_expr(self.out_pos(out_c))
                    cute.arch.mbarrier_wait(mbar_dQin_full + c, phase)
                    wait_out_slot_free(pos)
                    sm100_utils.gemm_ptx_w_idx(
                        tiled_mma_dQ,
                        tdQtdQ_slots[pos % self.num_out_slots],
                        tdQrdS,
                        tdQrKt,
                        sA=sdS,
                        sB=sKt,
                        A_idx=None,
                        B_idx=k["stage"],
                        zero_init=True,
                        cta_group=self.cta_group_size,
                    )
                    with cute.arch.elect_one():
                        tcgen05.commit(mbar_out_full + out_c)

                phase ^= 1
            tmem_ptr_real = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
            )
            cute.arch.mbarrier_wait(mbar_tmem_dealloc, 0)
            cute.arch.dealloc_tmem(tmem_ptr_real, Int32(self.tmem_alloc_cols))

        elif warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            # Warpgroup-relative thread id: the T2R / R2T copies are built for 128
            # threads, so this branch indexes them with 0..127.
            compute_tidx = tidx - self.compute_warp_ids[0] * cute.arch.WARP_SIZE
            # kv_shared drains dK and dV through ONE accumulator, so they cannot carry
            # different scales any more: softmax_scale is folded into dS here (dK and dQ
            # both come from dS, dV does not) and the postprocess applies 1.0 to dKV and
            # dQ instead. This is where the cudnn DSA bwd folds it too. Recovered from
            # the log2 form the kernel is given rather than passed as a second parameter;
            # the round trip is exact to 1e-16 and dS is about to become bf16 anyway.
            dS_scale = softmax_scale_log2 * cutlass.Float32(math.log(2.0))
            if cutlass.const_expr(self.drain_split):
                # This warpgroup's share of the output drain: the ODD slices of every
                # chunk (the drain warps take the even ones). Built once -- the T2R is
                # partitioned over these 128 threads.
                (
                    thr_t2r_out_c,
                    out_slice_layout_c,
                    shape_out_slice_c,
                ) = _bigd_make_out_drain(self, compute_tidx, tmem_ptr)
                mdKaccum_cur_c = mdKaccum[batch_idx, head_idx_kv, None]
                mdQaccum_cur_c = mdQaccum[batch_idx, head_idx, None]
            phase = Int32(0)
            for it in cutlass.range(num_iters, unroll=1):
                # The iteration counter drives the barrier phases; m_iter is the actual
                # block index, which skips the fully masked bands (see the skip range).
                m_iter = _bigd_m_block(it, m_lo, n1, seg2_base, n2, seg3_base)
                # The S -> P -> dP -> dS round trip runs in chunks of
                # softmax_chunk_m columns instead of over the whole tile at once.
                #
                # Why: ncu on the pinned d512 config shows the kernel is spill bound --
                # heavy l1tex local ld/st, the tensor pipe idle, and most warp cycles
                # stalled on an L1TEX scoreboard.
                # One fragment is tile_n * tile_m / 128 = 128 f32 per thread and the
                # round trip keeps four of them live, plus the two packed R2T buffers:
                # 640 f32 against 128 registers, and the CTA already owns the whole
                # register file, so more registers cannot be had. The local traffic and
                # the bulk of the stall budget were that spill. Chunking makes the live
                # set 5 * W instead of 5 * tile_m. DSA's bwd sits at 32 f32 per thread
                # here (tile 64x64, split into two Rep(4) LDTMs).
                #
                # Everything below is built once: every chunk has the same shape and
                # the same thread mapping, only the TMEM column offset and the m
                # coordinate base move.
                W = cutlass.const_expr(self.softmax_chunk_m)
                # 32-datapath T2R. This is why the config uses tile_n=128: with
                # M = cta_group * tile_n = 128 these atoms apply and their value mode is
                # a pure column run, which is what makes the packed bf16 R2T store below
                # line up element-for-element (load rep W/4, store rep W/8 -- the packed
                # view is half as wide in f32 columns). At M=64 only the 16-datapath
                # atoms are available and that correspondence breaks: measured exact for
                # a constant A operand but ~7% off for A = m or A = n.
                mma_S_chunk = sm100_utils_basic.make_trivial_tiled_mma(
                    self.q_dtype,
                    tcgen05.OperandMajorMode.K,
                    tcgen05.OperandMajorMode.K,
                    self.acc_dtype,
                    self.cta_group,
                    (self.mma_tiler_kq[0], W),
                )
                thr_mma_S_c = mma_S_chunk.get_slice(0)
                tS_chunk_layout = thr_mma_S_c.make_fragment_C(
                    thr_mma_S_c.partition_shape_C((self.mma_tiler_kq[0], W))
                ).layout
                # dP gets its own chunk layout: the full-tile views this replaced took
                # P's shape from tStS (mma_tiler_kq, the S gemm) and dS's from tdPtdP
                # (mma_tiler_vdo, the dP gemm). They are two different MMAs, so do not
                # assume one layout describes both.
                mma_dP_chunk = sm100_utils_basic.make_trivial_tiled_mma(
                    self.q_dtype,
                    tcgen05.OperandMajorMode.K,
                    tcgen05.OperandMajorMode.K,
                    self.acc_dtype,
                    self.cta_group,
                    (self.mma_tiler_vdo[0], W),
                )
                thr_mma_dP_c = mma_dP_chunk.get_slice(0)
                tdP_chunk_layout = thr_mma_dP_c.make_fragment_C(
                    thr_mma_dP_c.partition_shape_C((self.mma_tiler_vdo[0], W))
                ).layout
                tmem_load_atom = cute.make_copy_atom(
                    tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(W // 4)), Float32
                )
                tiled_copy_t2r = tcgen05.make_tmem_copy(
                    tmem_load_atom, cute.make_tensor(tStS.iterator, tS_chunk_layout)
                )
                thr_copy_t2r = tiled_copy_t2r.get_slice(compute_tidx)
                # partition_D takes a tensor already shaped like the accumulator, i.e.
                # partitioned by the MMA first -- a flat (tile_n, W) identity
                # tensor does not match the tiler and fails op creation.
                cS = thr_mma_S_c.partition_C(
                    cute.make_identity_tensor((self.mma_tiler_kq[0], W))
                )
                tScS = thr_copy_t2r.partition_D(cS)
                # Four separate f32 fragments, on purpose. Tried and REVERTED: aliasing
                # them in pairs (P overwriting S, dS overwriting dP) to cut the live
                # count in half measured much slower. make_fragment lowers to an alloca,
                # and reading and writing the same alloca inside one unrolled loop
                # defeats its promotion to registers, so the whole buffer lands in local
                # memory. Left separate, ptxas sees that S dies after the P loop (and dP
                # after the dS loop) and reuses those registers itself. Shrinking the
                # fragments is the fix; aliasing is not.
                tSrS = cute.make_fragment(tScS.shape, Float32)
                tSrP = cute.make_fragment(tScS.shape, Float32)
                tSrdP = cute.make_fragment(tScS.shape, Float32)
                tSrdS = cute.make_fragment(tScS.shape, Float32)
                frag_len = cutlass.const_expr(cute.size(tSrS))
                # This chunk's W values of LSE / dPsum, staged out of SMEM once per
                # chunk instead of read per element.
                #
                # The element loops below index LSE / dPsum by the *m* coordinate of
                # each element, which is mode 1 of tScS and a compile-time constant, so
                # the reads were tile_m scalar ld.shared per thread per m tile with all
                # 32 lanes of a warp asking for the same address: at W=32 that is
                # 2 x 32 x num_softmax_chunks = 256 shared loads per thread per m tile,
                # which is the same order as the drain's 1536 vector atomics and the
                # other half of the L1TEX traffic this kernel stalls on. A chunk's W
                # values are contiguous, so one autovec_copy replaces 32 scalar loads
                # with 8 x ld.shared.v4 (the recipe flash_bwd_sm100.py uses for its LSE
                # / dPsum s2r). Costs W registers each; affordable only at 12 warps.
                #
                # Separate fragments, not one reused buffer: their live ranges do not
                # overlap (LSE dies at the end of the P loop, dPsum is only live in the
                # dS loop) so ptxas shares the registers anyway, and aliasing one alloca
                # is what defeats promotion (see the note above).
                tSrLSE = cute.make_fragment(W, Float32)
                tSrdPsum = cute.make_fragment(W, Float32)

                # R2T machinery: P and dS go back into the upper halves of the S / dP
                # regions as bf16, where they are the A operands of the dV / dK gemms.
                # With M = cta_group * tile_n = 128 the 32-datapath atoms apply, so the
                # store is the proven fwd pattern: an f32 register fragment whose bytes
                # are viewed as bf16, and a store rep that is half the load rep (the
                # packed view is half as wide in f32 columns).
                tile_p_like_f32_c = cutlass.const_expr(W // 32 * self.q_dtype.width)
                tmem_store_atom = cute.make_copy_atom(
                    tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(W // 8)), Float32
                )
                tP_chunk_layout = cute.composition(
                    tS_chunk_layout,
                    cute.make_layout((self.tile_n, tile_p_like_f32_c)),
                )
                tdS_chunk_layout = cute.composition(
                    tdP_chunk_layout,
                    cute.make_layout((self.tile_n, tile_p_like_f32_c)),
                )
                cP = cute.make_identity_tensor((self.tile_n, tile_p_like_f32_c))
                thr_store_P = tcgen05.make_tmem_copy(
                    tmem_store_atom,
                    cute.make_tensor(
                        tStS.iterator + self.tmem_s_to_p_offset, tP_chunk_layout
                    ),
                ).get_slice(compute_tidx)
                thr_store_dS = tcgen05.make_tmem_copy(
                    tmem_store_atom,
                    cute.make_tensor(
                        tdPtdP.iterator + self.tmem_s_to_p_offset, tdS_chunk_layout
                    ),
                ).get_slice(compute_tidx)
                tSrP_r2t_f32 = cute.make_fragment(
                    thr_store_P.partition_S(cP).shape, Float32
                )
                tSrdS_r2t_f32 = cute.make_fragment(
                    thr_store_dS.partition_S(cP).shape, Float32
                )
                tSrP_r2t = cute.make_tensor(
                    cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS.layout
                )
                tSrdS_r2t = cute.make_tensor(
                    cute.recast_ptr(tSrdS_r2t_f32.iterator, dtype=self.ds_dtype), tSrS.layout
                )
                # Two bf16 per f32 lane: the packed fragment must hold exactly the
                # elements the T2R load produced, or part of P / dS stays uninitialized
                # and the gemm returns NaN.
                assert 2 * cute.size(tSrP_r2t_f32) == frag_len, (
                    "packed R2T fragment holds %d bf16, T2R produced %d"
                    % (2 * cute.size(tSrP_r2t_f32), frag_len)
                )
                # dS -> SMEM in its natural (n, m) orientation; the dQ gemm reads the
                # same bytes through the transposed sdS view.
                #
                # Vectorised r2s, built from the T2R copy's destination TV layout -- the
                # recipe flash_bwd_postprocess.py uses: get_smem_store_op picks the
                # widest legal store for the
                # (layout, dtype) pair and make_tiled_copy re-tiles it onto exactly the
                # thread/value mapping the T2R produced, so the register fragment can go
                # out as 16-byte stores instead of W scalar ones. NB: make_tiled_copy_D
                # is NOT the tool here -- it hands each thread twice the elements and
                # does not expose layout_tv.
                #
                # If this breaks, the symptom is precise: dS feeds dK through TMEM and dQ
                # through SMEM, so dQ wrong while dK stays right means the store mapping
                # is off.
                smem_store_atom = sm100_utils_basic.get_smem_store_op(
                    LayoutEnum.ROW_MAJOR,
                    self.ds_dtype,
                    Float32,
                    tiled_copy_t2r,
                )
                thr_r2s_dS = cute.make_tiled_copy(
                    smem_store_atom,
                    layout_tv=tiled_copy_t2r.layout_dst_tv_tiled,
                    tiler_mn=tiled_copy_t2r.tiler_mn,
                ).get_slice(compute_tidx)
                # partition_C needs a plain (tile_n, m) view: handed the raw A-operand
                # layout it broadcasts instead of slicing (measured: a stride-0 mode),
                # and partition_D then hands every thread the whole tile. Composing the
                # swizzled layout down to (n, m) keeps the A-operand addresses.
                sdSt_nm = cute.make_tensor(
                    sdSt.iterator,
                    cute.composition(
                        sdSt.layout, cute.make_layout((self.tile_n, self.tile_m))
                    ),
                )

                # LSE / dPsum are read from SMEM below (the load warp stages them on
                # mbar_stats_full): the m index is a per-ELEMENT coordinate, so reading
                # them from gmem there cost tile_m dependent scalar loads per thread per
                # m tile, all 32 lanes of a warp asking for the same value.
                #
                # Masks, applied to P rather than to S: P == 0 kills the element's
                # contribution to dV, and dS = P * (dP - dPsum) == 0 kills it for dK and
                # dQ too. Doing it on P also means LSE / dPsum garbage in the padded rows
                # cannot turn into NaN (a select on the result, not on the exponent).
                #
                # S^T is (n, m) = (key, query): mode 0 of tScS is the key index and is
                # thread-derived (one thread owns one whole key row, for every chunk),
                # mode 1 is the query index and is a per-element compile-time constant.
                # So the key-side predicates are loop- and chunk-invariant and only the
                # query side is per element.
                # (Measured on B200 -- note this is the OPPOSITE of what mask.py's
                # apply_mask_sm100_transposed assumes, which is why that helper is not
                # used here.)
                #
                # TMA already zero-fills the out-of-range Q / K / dO tiles; what it
                # cannot do is stop exp2(0 - lse_pad) from being a nonzero P.
                n_global = n_block * self.tile_n + tScS[0][0]
                n_oob = n_global >= seqlen_k
                m_base = m_iter * self.tile_m
                if cutlass.const_expr(self.is_causal):
                    # Bottom-right aligned, the FA convention (mask.py's
                    # causal_row_offset = 1 + seqlen_k - n_block*tile_n - seqlen_q - ...):
                    # keep n_global <= m_global + seqlen_k - seqlen_q.
                    causal_m_min = n_global - (seqlen_k - seqlen_q)
                if cutlass.const_expr(self.fm_bound_num > 0):
                    # flashmask: startend_row_indices is [b, h_fm, seqlen_k, bound_num]
                    # and gives, PER KEY COLUMN, bounds on the QUERY rows to mask. The
                    # key column is this thread's row of S^T, so the bounds are read
                    # once per thread (mask.py re-reads them per element because there
                    # the key is the per-element coordinate).
                    #
                    # Semantics, verbatim from the reference
                    # (test_flashmask/generate_startend_row_indices.py):
                    #   has_end = (causal and bound_num == 2) or (not causal and == 4)
                    #   lower tail: mask rows [idx0, idx1) if has_end else [idx0, seqlen_q)
                    #   causal    : plus the causal mask (handled above)
                    #   not causal: plus the upper tail, [idx2, idx3) if has_end
                    #               else [0, idx1)
                    # fm_b / fm_h come from the kernel's top level (the skip range needs
                    # them too).
                    # Threads whose key is past seqlen_k are masked by n_oob anyway, but
                    # the read itself still has to stay in bounds.
                    fm_row = mFM[
                        fm_b, fm_h, fm_row_offset + cutlass.min(n_global, seqlen_k - 1), None
                    ]
                    has_end = cutlass.const_expr(
                        (self.is_causal and self.fm_bound_num == 2)
                        or (not self.is_causal and self.fm_bound_num == 4)
                    )
                    fm_ds = fm_row[0]
                    fm_de = fm_row[1] if cutlass.const_expr(has_end) else seqlen_q
                    if cutlass.const_expr(not self.is_causal):
                        if cutlass.const_expr(has_end):
                            fm_us, fm_ue = fm_row[2], fm_row[3]
                        else:
                            fm_us, fm_ue = Int32(0), fm_row[1]
                    # Per m block, uniform across the CTA: does any key of the pair mask
                    # any row of this block? If not, the block is unmasked and the
                    # per-element flashmask predicate below is skipped (causal and
                    # out-of-range masking still apply). The bounds are the union bounds
                    # published at the top level; see the reduction there.
                    m_end_blk = m_base + self.tile_m
                    fm_clear = m_end_blk <= fm_lo_start_min or m_base >= fm_lo_end_max
                    if cutlass.const_expr(not self.is_causal):
                        fm_clear = fm_clear and (
                            m_end_blk <= fm_up_start_min or m_base >= fm_up_end_max
                        )
                    fm_need_mask = not fm_clear

                # S^T, then P = exp2(S * scale_log2 - lse[m]). S^T is (n, m) = (key,
                # query), and softmax normalizes over keys, so LSE and dPsum are
                # indexed by the *m* coordinate of each element.
                cute.arch.mbarrier_wait(mbar_S_full, phase)
                cute.arch.mbarrier_wait(mbar_stats_full, phase)
                for cmi in cutlass.range_constexpr(self.num_softmax_chunks):
                    # DESCENDING chunk order, and it has to be: P is stored into the
                    # UPPER HALF of the S region (tmem_s_to_p_offset = tile_m // 2,
                    # bf16 needs half the columns) and dS likewise into dP's, so a
                    # chunk's R2T store lands on columns another chunk may not have
                    # read yet. Ascending order was measured wrong exactly this way:
                    # P of chunk 0 goes to f32 columns [64, 80), which is inside
                    # chunk 2's S range [64, 96) -- chunks 0 and 1 came out right and
                    # 2 and 3 read clobbered S / dP (dq cosine 0.85, dkv 0.90).
                    # Descending is always safe: chunk cm stores into
                    # [tile_m/2 + cm*W/2, ...), and tile_m/2 + cm*W/2 >= cm*W for any
                    # cm*W <= tile_m, i.e. never below the columns already consumed.
                    cm = cutlass.const_expr(self.num_softmax_chunks - 1 - cmi)
                    # TMEM column offsets: the f32 accumulators advance by W columns per
                    # chunk, the packed bf16 P / dS regions by W // 2 (two bf16 per f32
                    # column).
                    col_f32 = cutlass.const_expr(cm * W)
                    col_packed = cutlass.const_expr(cm * (W // 2))
                    m_off = cutlass.const_expr(cm * W)
                    tStS_c = cute.make_tensor(tStS.iterator + col_f32, tS_chunk_layout)
                    tdPtdP_c = cute.make_tensor(
                        tdPtdP.iterator + col_f32, tdP_chunk_layout
                    )
                    tStP_c = cute.make_tensor(
                        tStS.iterator + self.tmem_s_to_p_offset + col_packed,
                        tP_chunk_layout,
                    )
                    tStdS_c = cute.make_tensor(
                        tdPtdP.iterator + self.tmem_s_to_p_offset + col_packed,
                        tdS_chunk_layout,
                    )

                    cute.copy(thr_copy_t2r, thr_copy_t2r.partition_S(tStS_c), tSrS)
                    cute.arch.fence_view_async_tmem_load()
                    # This chunk's W LSE values, one vectorised s2r instead of a
                    # scalar ld.shared per element.
                    cute.autovec_copy(cute.local_tile(sLSE, (W,), (cm,)), tSrLSE)
                    for i in cutlass.range_constexpr(frag_len):
                        # const_expr, not just an int: mode 1 of tScS is the m
                        # coordinate and has to be a compile-time constant for
                        # tSrLSE[.] to stay in registers. If it ever becomes dynamic
                        # this raises at trace time instead of silently moving the
                        # fragment to local memory.
                        mi = cutlass.const_expr(tScS[i][1])
                        m_idx = cutlass.const_expr(mi + m_off)
                        p = cute.math.exp2(
                            tSrS[i] * softmax_scale_log2 - tSrLSE[mi], fastmath=True
                        )
                        m_global = m_base + m_idx
                        bad = n_oob or m_global >= seqlen_q
                        if cutlass.const_expr(self.is_causal):
                            bad = bad or m_global < causal_m_min
                        tSrP[i] = 0.0 if bad else p
                    if cutlass.const_expr(self.fm_bound_num > 0):
                        # Flashmask predicate, only for partially masked m blocks. A
                        # separate pass over P keeps the unmasked blocks free of it.
                        if fm_need_mask:
                            for i in cutlass.range_constexpr(frag_len):
                                m_fm = m_base + cutlass.const_expr(tScS[i][1] + m_off)
                                fm_bad = m_fm >= fm_ds and m_fm < fm_de
                                if cutlass.const_expr(not self.is_causal):
                                    fm_bad = fm_bad or (m_fm >= fm_us and m_fm < fm_ue)
                                tSrP[i] = 0.0 if fm_bad else tSrP[i]

                    # dP^T, then dS^T = P * (dP - dPsum[m]). The wait sits after the
                    # FIRST chunk's P pass so that pass still overlaps the mma warp's
                    # dP gemms.
                    if cutlass.const_expr(cmi == 0):
                        cute.arch.mbarrier_wait(mbar_dP_full, phase)
                    cute.copy(thr_copy_t2r, thr_copy_t2r.partition_S(tdPtdP_c), tSrdP)
                    cute.arch.fence_view_async_tmem_load()
                    cute.autovec_copy(cute.local_tile(sdPsum, (W,), (cm,)), tSrdPsum)
                    for i in cutlass.range_constexpr(frag_len):
                        mi = cutlass.const_expr(tScS[i][1])
                        if cutlass.const_expr(self.kv_shared):
                            tSrdS[i] = (
                                tSrP[i] * (tSrdP[i] - tSrdPsum[mi]) * dS_scale
                            )
                        else:
                            tSrdS[i] = tSrP[i] * (tSrdP[i] - tSrdPsum[mi])

                    # R2T of this chunk's P and dS, then its dS slice to SMEM.
                    for i in cutlass.range_constexpr(frag_len):
                        tSrP_r2t[i] = tSrP[i].to(self.q_dtype)
                        tSrdS_r2t[i] = tSrdS[i].to(self.ds_dtype)
                    cute.copy(thr_store_P, tSrP_r2t_f32, thr_store_P.partition_D(tStP_c))
                    # The dK gemm reads dS out of TMEM, so this R2T has a consumer; the
                    # dQ gemm reads the SMEM copy written below.
                    cute.copy(
                        thr_store_dS,
                        tSrdS_r2t_f32,
                        thr_store_dS.partition_D(tStdS_c),
                    )
                    tdSsdS = thr_copy_t2r.partition_D(
                        thr_mma_S_c.partition_C(
                            cute.local_tile(sdSt_nm, (self.tile_n, W), (0, cm))
                        )
                    )
                    assert cute.size(tdSsdS) == frag_len, (
                        "dS r2s destination has %d elements per thread, T2R produced %d"
                        % (cute.size(tdSsdS), frag_len)
                    )
                    cute.copy(
                        thr_r2s_dS,
                        cute.make_tensor(tSrdS_r2t.iterator, tdSsdS.shape),
                        tdSsdS,
                    )
                # P and dS are complete for the whole tile: publish them to the mma
                # warp (TMEM) and to the dQ gemm (SMEM). One arrive each per m
                # iteration, which is what the barriers were initialised for.
                cute.arch.fence_view_async_tmem_store()
                cute.arch.mbarrier_arrive(mbar_PdS_full)
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(mbar_dSsmem_full)
                # Every element of LSE / dPsum has been consumed by now, so the next m
                # tile's bulk copy may overwrite them.
                cute.arch.mbarrier_arrive(mbar_stats_empty)

                # dV / dK / dQ: without the split the drain warpgroup handles
                # all of them and this warpgroup is done once P / dS are in TMEM and
                # SMEM. With it, this warpgroup takes the odd slices of every chunk.
                # This sits AFTER the PdS_full / dSsmem_full arrivals, so the output
                # gemms it waits on are already unblocked.
                if cutlass.const_expr(self.drain_split):
                    _bigd_drain_wg_iteration(
                        self,
                        phase,
                        m_iter,
                        compute_tidx,
                        thr_t2r_out_c,
                        out_slice_layout_c,
                        shape_out_slice_c,
                        slice_lo=1,
                        slot_idx=1,
                        barrier_id=5,
                        release_in_bars=False,
                        mdKaccum_cur=mdKaccum_cur_c,
                        mdQaccum_cur=mdQaccum_cur_c,
                        seqlen_k=seqlen_k,
                        num_m_block=num_m_block,
                        n_block=n_block,
                        tmem_ptr=tmem_ptr,
                        sOutAccum=sOutAccum,
                        mbar_out_full=mbar_out_full,
                        mbar_out_empty=mbar_out_empty,
                        mbar_dKin_empty=mbar_dKin_empty,
                        mbar_dVin_empty=mbar_dVin_empty,
                        mbar_dQin_empty=mbar_dQin_empty,
                    )

                phase ^= 1

            if cutlass.const_expr(self.drain_split):
                # This warpgroup's bulk groups read staging slot 1; they must be done
                # before the CTA exits and the buffer is handed to the next one.
                if compute_tidx < cute.arch.WARP_SIZE:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(
                    barrier_id=5,
                    number_of_threads=cute.arch.WARP_SIZE
                    * len(self.compute_warp_ids),
                )

            # TMEM is released once, after the whole m loop -- the mma warp holds the
            # single allocation for the CTA's lifetime. Both the compute and the drain
            # warpgroup have to be done with TMEM before it goes away.
            cute.arch.mbarrier_arrive(mbar_tmem_dealloc)

        elif warp_idx <= self.drain_warp_ids[-1]:
            # ---------------------------------------------------------- drain warps
            # dV / dK / dQ leave TMEM through a fp32 gmem accumulator: T2R one
            # dQ_reduce_ncol-column slice into registers, then red.global.add.v4.f32
            # it into the accumulator. The add is what makes the m loop (dK/dV) and
            # the n grid (dQ) accumulate without any cross-CTA handshake.
            #
            # This used to run at the tail of the compute warps' iteration. It is its
            # own warpgroup so that the slices of iteration i overlap the softmax /
            # dS math of iteration i+1.
            # The handshake is unchanged: mbar_out_full[c] from the mma warp, and
            # mbar_out_empty[c] / *_in_empty[c] back to it -- only their arrive counts
            # moved to this warpgroup.
            #
            # Accumulator element order is identical to what the SMEM-staged version
            # wrote (see the note at the atomics), so FlashAttentionBackwardPostprocess
            # and _unblock_accum are unchanged.
            #
            # The postprocess still has to be *called* per head_dim slice: its
            # 1CTA path stages a whole tile_m x head_dim fp32 tile in SMEM
            # (at head_dim 512 that is 128*512*4 = 256KB) and holds it in
            # registers, so it has to be driven as e.g. 4 x 128 with
            # raw_storage_d instead of once at the full head_dim.
            cute.arch.setmaxregister_increase(self.num_regs_drain)
            drain_tidx = tidx - self.drain_warp_ids[0] * cute.arch.WARP_SIZE
            num_drain_threads = cutlass.const_expr(
                cute.arch.WARP_SIZE * len(self.drain_warp_ids)
            )
            ncol = cutlass.const_expr(self.dQ_reduce_ncol)
            # Bulk reduce-add staging (kv_shared): one slice per bulk group, and a
            # named barrier for this warpgroup only -- barrier 0 is the CTA-wide one
            # used during init.
            out_reduce_bytes = cutlass.const_expr(
                self.tile_m * ncol * Float32.width // 8
            )
            drain_barrier_id = cutlass.const_expr(4)
            assert num_drain_threads == self.tile_m and self.tile_n == self.tile_m, (
                "the accumulator layout assumes one row per drain thread"
            )
            # gmem accumulator per (batch, head); the tile index is the key block for
            # dK / dV and the m iteration for dQ.
            mdVaccum_cur = mdVaccum[batch_idx, head_idx_kv, None]
            mdKaccum_cur = mdKaccum[batch_idx, head_idx_kv, None]
            mdQaccum_cur = mdQaccum[batch_idx, head_idx, None]
            # ONE T2R per dQ_reduce_ncol-column slice, not one per chunk.
            #
            # This is what the whole drain hinges on: taking the drain out collapses local
            # ld/st to near zero and cuts the runtime by roughly two thirds. So
            # essentially all of the local traffic is this drain, and its volume is
            # exactly "every T2R fragment element written once and read once". The
            # fragments were not living in registers at all. Slicing the T2R makes each
            # fragment one slice instead of a whole chunk, which is what shrinks it.
            #
            # Byte order is unchanged, so FlashAttentionBackwardPostprocess and
            # _unblock_accum keep working: the old code took fragment elements
            # [s*ncol, (s+1)*ncol) for slice s, and the T2R value mode is a pure column
            # run (element i <-> column i), so that is the same data a T2R at column
            # offset s*ncol produces.
            #
            # All three outputs share this machinery: dV's, dK's and dQ's accumulators
            # all have M = 128 (cta_group * tile_n for the K-side pair, tile_m for dQ),
            # so a single (128, ncol) slice layout and copy atom covers them and only
            # the TMEM column offset differs.
            thr_t2r_out, out_slice_layout, shape_out_slice = _bigd_make_out_drain(
                self, drain_tidx, tmem_ptr
            )
            flen_slice = cutlass.const_expr(cute.size(shape_out_slice))
            assert (
                self.mma_tiler_pdo[0] == self.mma_tiler_dsq[0]
                and self.mma_tiler_pdo[0] == self.mma_tiler_dsk[0]
            ), (
                "the shared slice T2R assumes all three output accumulators have the "
                "same M: %d / %d / %d"
                % (self.mma_tiler_pdo[0], self.mma_tiler_dsq[0], self.mma_tiler_dsk[0])
            )
            assert num_drain_threads * flen_slice == self.tile_m * ncol, (
                "slice T2R gives %d elements per thread over %d threads, staging wants "
                "tile_m * ncol = %d" % (flen_slice, num_drain_threads, self.tile_m * ncol)
            )
            # gmem accumulator for this CTA's block. Two levels of blocking:
            #   [head_dim slice][row block][4-col group][row][4 cols]
            # The outer slice exists so the shared postprocess can be called once
            # per slice with head_dim = accum_slice: each slice is a contiguous
            # region that looks exactly like a head_dim=accum_slice accumulator.
            # dK / dV are indexed by the key block (the grid's x), dQ by the m
            # iteration.
            num_n_block = cute.ceil_div(seqlen_k, self.tile_n)
            phase = Int32(0)
            if cutlass.const_expr(self.drain_split):
                # Split drain: this warpgroup takes the EVEN slices of every chunk
                # and releases the SMEM operands; the compute warpgroup takes the odd
                # slices (see its branch). Slot 0 and named barrier 4 are this
                # warpgroup's; the pre-split path below is what the configs that
                # cannot split (no kv_shared) still take.
                for it in cutlass.range(num_iters, unroll=1):
                    m_iter = _bigd_m_block(it, m_lo, n1, seg2_base, n2, seg3_base)
                    _bigd_drain_wg_iteration(
                        self,
                        phase,
                        m_iter,
                        drain_tidx,
                        thr_t2r_out,
                        out_slice_layout,
                        shape_out_slice,
                        slice_lo=0,
                        slot_idx=0,
                        barrier_id=4,
                        release_in_bars=True,
                        mdKaccum_cur=mdKaccum_cur,
                        mdQaccum_cur=mdQaccum_cur,
                        seqlen_k=seqlen_k,
                        num_m_block=num_m_block,
                        n_block=n_block,
                        tmem_ptr=tmem_ptr,
                        sOutAccum=sOutAccum,
                        mbar_out_full=mbar_out_full,
                        mbar_out_empty=mbar_out_empty,
                        mbar_dKin_empty=mbar_dKin_empty,
                        mbar_dVin_empty=mbar_dVin_empty,
                        mbar_dQin_empty=mbar_dQin_empty,
                    )
                    phase ^= 1
                # This warpgroup's bulk groups read staging slot 0; they must be done
                # before the CTA exits and the buffer is handed to the next one.
                if drain_tidx < cute.arch.WARP_SIZE:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(
                    barrier_id=drain_barrier_id, number_of_threads=num_drain_threads
                )
                cute.arch.mbarrier_arrive(mbar_tmem_dealloc)
            else:
                # Deterministic dK / dV: with qhead_per_kvhead > 1 several query-head CTAs
                # add into the same dK / dV rows. Serialise them in a fixed
                # (head_idx % qhead_per_kvhead) order by holding one lock across the WHOLE
                # m loop -- a per-m-block lock would have head 0 wait on head Q-1, i.e. on
                # a HIGHER linear blockIdx, which deadlocks at less than full occupancy.
                # One lock covers dV as well: under kv_shared dV is drained together with
                # dK, and otherwise both drains sit inside this same critical section.
                #
                # Cost: the lock spans the whole m loop, so the qhead_per_kvhead CTAs of a
                # kv head group run their m loops strictly one after another, not just
                # their dK / dV writebacks -- throughput for deterministic GQA drops by
                # roughly that factor. flash_bwd_sm90.py can scope its two locks to the
                # epilogue only because it reduces dK / dV once, at the end; here every m
                # iteration reduces into the same dK / dV rows, so a narrower critical
                # section would need a per-m-block counter walked in the same direction as
                # dQ's (waiting on a LOWER linear blockIdx) instead of the head ordinal.
                need_dKV_lock = cutlass.const_expr(
                    self.deterministic and self.qhead_per_kvhead > 1
                )
                if cutlass.const_expr(need_dKV_lock):
                    dKV_lock = mdK_semaphore[batch_idx, head_idx_kv, n_block, None].iterator
                    barrier.wait_eq(
                        dKV_lock,
                        drain_tidx,
                        0,
                        Int32(head_idx % self.qhead_per_kvhead),
                    )
                    cute.arch.barrier(
                        barrier_id=drain_barrier_id, number_of_threads=num_drain_threads
                    )
                if cutlass.const_expr(self.deterministic):
                    mdQ_sem_cur = mdQ_semaphore[batch_idx, head_idx, None, None]
                for it in cutlass.range(num_iters, unroll=1):
                    # Same iteration mapping as the mma / compute warps: the counter
                    # drives the barrier phases, m_iter is the actual block index (it
                    # indexes dQ's accumulator, so it must match exactly).
                    m_iter = _bigd_m_block(it, m_lo, n1, seg2_base, n2, seg3_base)
                    # Iterated with range_constexpr, not `for ... in outputs`: a bare
                    # Python for over a tuple is rewritten by the DSL into a dynamic loop
                    # region and then it tries to flatten the tuple's contents.
                    # All three outputs live at the same scratch base; only the slot and
                    # the slice offset move.
                    out_base_ptr = tmem_ptr + self.tmem_out_offset
                    if cutlass.const_expr(self.kv_shared):
                        # dV was accumulated into dK's slot by the mma warp, so there is one
                        # dKV drain per chunk and it releases BOTH SMEM operands (sQt for the
                        # dK gemm, sdO for the dV gemm).
                        outputs = (
                            (self.num_d_chunks, self.d_chunk, mdKaccum_cur,
                             self.accum_slice_d, num_n_block, n_block, self.out_base_dK,
                             mbar_dKin_empty, mbar_dVin_empty),
                            (self.num_d_chunks, self.d_chunk, mdQaccum_cur,
                             self.accum_slice_d, num_m_block, m_iter, self.out_base_dQ,
                             mbar_dQin_empty, None),
                        )
                    else:
                        outputs = (
                            (self.num_dv_chunks, self.dv_chunk, mdVaccum_cur,
                             self.accum_slice_dv, num_n_block, n_block, self.out_base_dV,
                             mbar_dVin_empty, None),
                            (self.num_d_chunks, self.d_chunk, mdKaccum_cur,
                             self.accum_slice_d, num_n_block, n_block, self.out_base_dK,
                             mbar_dKin_empty, None),
                            (self.num_d_chunks, self.d_chunk, mdQaccum_cur,
                             self.accum_slice_d, num_m_block, m_iter, self.out_base_dQ,
                             mbar_dQin_empty, None),
                        )
                    # Iterated with range_constexpr, not `for ... in outputs`: a bare
                    # Python for over a tuple is rewritten by the DSL into a dynamic loop
                    # region and then it tries to flatten the tuple's contents.
                    for oi in cutlass.range_constexpr(len(outputs)):
                        (nchunks, chunk_w, maccum, hd_slice, num_blocks,
                         block_idx, base, in_bar, in_bar2) = outputs[oi]
                        # dQ is always the LAST entry, and it is the only output whose
                        # accumulator row block (m_iter) is shared with other CTAs of this
                        # grid's x dimension. Deterministic pins the order of those adds:
                        # spin until the counter for this m block equals our own n_block,
                        # reduce, then hand it to n_block + 1.
                        #
                        # `lock_value = n_block` is only an arrival count because the CTAs
                        # that touch a given m block form a PREFIX {0..N} of the n axis --
                        # which is why the flashmask narrowing is disabled under this flag
                        # (see the m skip range above).
                        is_dQ = cutlass.const_expr(oi == len(outputs) - 1)
                        if cutlass.const_expr(self.deterministic and is_dQ):
                            dQ_lock = mdQ_sem_cur[(m_iter, None)].iterator
                            barrier.wait_eq(dQ_lock, drain_tidx, 0, Int32(n_block))
                            cute.arch.barrier(
                                barrier_id=drain_barrier_id,
                                number_of_threads=num_drain_threads,
                            )

                        chunks_per_slice = cutlass.const_expr(hd_slice // chunk_w)
                        # Offsets are plain pointer arithmetic instead of nested local_tile
                        # because the outer (per-slice) extent is dynamic.
                        slice_stride = num_blocks * (self.tile_m * hd_slice)
                        block_base = block_idx * (self.tile_m * hd_slice)
                        # DESCENDING, the same order the mma warp issues in. The scratch
                        # slots gate the mma warp on this warpgroup, so draining in a
                        # different order than the mma warp issues deadlocks as soon as the
                        # two orders diverge by more than num_out_slots.
                        for pos_in_seg in cutlass.range_constexpr(nchunks):
                            c = cutlass.const_expr(nchunks - 1 - pos_in_seg)
                            out_c = cutlass.const_expr(base + c)
                            chunk_base = cutlass.const_expr(
                                (c % chunks_per_slice) * (self.tile_m * chunk_w)
                            )
                            slice_idx = cutlass.const_expr(c // chunks_per_slice)
                            cute.arch.mbarrier_wait(mbar_out_full + out_c, phase)
                            # The SMEM operand this gemm read (sdO for dV, sQt for dK, sKt
                            # for dQ) is free the moment the gemm completes, and out_full IS
                            # that completion (it carries a tcgen05.commit). Releasing it
                            # here rather than after the drain below takes this chunk's T2R
                            # and its vector atomics off the load warp's critical path: with
                            # one stage per buffer, the next m iteration's fetch of that
                            # chunk cannot start until this arrival, so the m loop used to
                            # serialise load -> gemm -> drain -> load. out_empty stays where
                            # it is: it guards the TMEM slot, which is what the T2R reads.
                            cute.arch.mbarrier_arrive(in_bar + c)
                            if cutlass.const_expr(in_bar2 is not None):
                                cute.arch.mbarrier_arrive(in_bar2 + c)
                            elem_base = slice_idx * slice_stride + block_base + chunk_base
                            # Slot the mma warp wrote this chunk into: by ISSUE position, not
                            # by out_c, because the segments run descending.
                            slot_base = cutlass.const_expr(
                                (self.out_pos(out_c) % self.num_out_slots)
                                * self.tmem_out_slot_cols
                            )
                            for s in cutlass.range_constexpr(chunk_w // ncol):
                                # One ncol-column slice per pass: T2R it, then reduce it into
                                # the fp32 gmem accumulator with red.global.add.v4.f32
                                # straight out of the registers.
                                #
                                # ONE fragment live. Keeping all of a chunk's slices in
                                # flight instead is what spills (see the register budget
                                # above for the measurement) -- at 384 threads ptxas caps
                                # every thread at 168 registers whatever setmaxnreg says.
                                #
                                # This replaced a SMEM staging round trip (a vectorised r2s
                                # into a staging buffer, two named barriers around it, one
                                # elected thread issuing a 32KB cp.reduce.async.bulk.add.f32).
                                # Measured split of the drain before that change: the gmem
                                # reduce was somewhat more than half of it and the on-chip
                                # T2R + staging the rest -- and at the one staging slot the
                                # SMEM budget allowed at ncol=64 that staging was fully
                                # serialised: wait for every outstanding reduce to have read
                                # the slot, barrier, fill, fence, barrier, issue, with two
                                # whole-warpgroup barriers per slice. DSA drains
                                # its dKV the same way this does now (dsa_bwd_sm100.py's
                                # scatter_dkv_atomic: float4 atomics from registers, no
                                # staging).
                                #
                                # kv_shared brings the staging back, with the three things
                                # that version got wrong fixed: TWO slots (dropping sV paid
                                # for them) so the fill overlaps the previous reduce, 16
                                # slices per m tile instead of 24 (dV's segment is merged
                                # into dK's), and an r2s that never forms the fragment's
                                # address so it stays in registers. The split path below is
                                # untouched.
                                #
                                # Byte layout is preserved exactly, which is what keeps
                                # FlashAttentionBackwardPostprocess and _unblock_accum valid.
                                # The old path was: tiled_copy_1d(f32, 128 threads, 4 elems)
                                # put thread t's fragment element r*4+v at staging position
                                # r*512 + t*4 + v, and the bulk copy moved the buffer to gmem
                                # in order. So element r*4+v belongs at gmem offset
                                # r*(num_drain_threads*4) + t*4 + v -- which is what the 16
                                # vector atomics below write. Each is 16B aligned (t*4 f32)
                                # and a warp's 32 lanes cover 512 contiguous bytes.
                                frag = cute.make_fragment(shape_out_slice, Float32)
                                tmem_src = cute.make_tensor(
                                    out_base_ptr + slot_base + s * ncol, out_slice_layout
                                )
                                cute.copy(
                                    thr_t2r_out, thr_t2r_out.partition_S(tmem_src), frag
                                )
                                cute.arch.fence_view_async_tmem_load()
                                if cutlass.const_expr(s == chunk_w // ncol - 1):
                                    # Every column of the slot is in registers now -- the
                                    # atomics below read registers, not TMEM -- so the slot
                                    # goes back to the mma warp here rather than after this
                                    # last slice's 16 atomics have issued. Free: no extra
                                    # live state, unlike pipelining the slices themselves.
                                    cute.arch.mbarrier_arrive(mbar_out_empty + out_c)
                                # Staging slot for this slice, by its ordinal in the whole
                                # iteration's drain sequence. All segments have the same
                                # chunk width under kv_shared (asserted at the outputs), and
                                # the per-iteration slice count is even, so the rotation does
                                # not drift across m iterations.
                                slot = cutlass.const_expr(
                                    (
                                        oi * nchunks * (chunk_w // ncol)
                                        + pos_in_seg * (chunk_w // ncol)
                                        + s
                                    )
                                    % max(self.out_stage, 1)
                                )
                                if cutlass.const_expr(
                                    self.kv_shared and not self.deterministic
                                ):
                                    # `deterministic` takes the register-atomic branch
                                    # below instead: the bulk reduce is an async-proxy
                                    # operation, so two reduces to the SAME accumulator
                                    # address from different m iterations are unordered
                                    # unless every one of them is drained with
                                    # cp_async_bulk_wait_group(0, read=False) first. The
                                    # register path needs no such drain -- each address is
                                    # written by exactly one thread (gbase carries
                                    # drain_tidx * 4), so same-thread same-address
                                    # coherence already orders it.
                                    #
                                    # Stage the slice in SMEM and let ONE bulk reduce-add
                                    # move all 32KB of it, instead of 128 threads x 16
                                    # red.global.add.v4.f32. Same bytes, but the drain warps
                                    # no longer hold a dependency on anything past L1TEX:
                                    # the store is shared-memory only and the global side is
                                    # the TMA unit's problem, tracked by a bulk group.
                                    #
                                    # The r2s goes through a re-viewed fragment: the
                                    # inner mode is the 4 contiguous elements so the
                                    # copy vectorises, the outer one strides by the
                                    # warpgroup's thread count.
                                    sslot = sOutAccum[None, slot].iterator + drain_tidx * 4
                                    cute.autovec_copy(
                                        cute.make_tensor(
                                            frag.iterator,
                                            cute.make_layout(
                                                (4, flen_slice // 4), stride=(1, 4)
                                            ),
                                        ),
                                        cute.make_tensor(
                                            sslot,
                                            cute.make_layout(
                                                (4, flen_slice // 4),
                                                stride=(1, num_drain_threads * 4),
                                            ),
                                        ),
                                    )
                                    cute.arch.fence_view_async_shared()
                                    cute.arch.barrier(
                                        barrier_id=drain_barrier_id,
                                        number_of_threads=num_drain_threads,
                                    )
                                    if drain_tidx < cute.arch.WARP_SIZE:
                                        with cute.arch.elect_one():
                                            copy_utils.cpasync_reduce_bulk_add_f32(
                                                sOutAccum[None, slot].iterator,
                                                maccum.iterator
                                                + (elem_base + s * (self.tile_m * ncol)),
                                                out_reduce_bytes,
                                            )
                                        cute.arch.cp_async_bulk_commit_group()
                                        # Leaves out_stage - 1 groups in flight, i.e. the
                                        # OTHER slot's reduce may still be running while the
                                        # next slice fills this one.
                                        cute.arch.cp_async_bulk_wait_group(
                                            self.out_stage - 1, read=True
                                        )
                                    cute.arch.barrier(
                                        barrier_id=drain_barrier_id,
                                        number_of_threads=num_drain_threads,
                                    )
                                else:
                                    gbase = maccum.iterator + (
                                        elem_base + s * (self.tile_m * ncol) + drain_tidx * 4
                                    )
                                    for r in cutlass.range_constexpr(flen_slice // 4):
                                        copy_utils.atomic_add_fp32x4(
                                            frag[r * 4 + 0],
                                            frag[r * 4 + 1],
                                            frag[r * 4 + 2],
                                            frag[r * 4 + 3],
                                            gbase + r * (num_drain_threads * 4),
                                        )

                        if cutlass.const_expr(self.deterministic and is_dQ):
                            # Every drain thread issued its own atomics, but only thread 0
                            # publishes the release. red.release only orders the RELEASING
                            # thread's own prior writes, so the other 127 have to fence
                            # their own at device scope and then rendezvous before the
                            # counter moves. fence.acq_rel.gpu is CUDA C's
                            # __threadfence(); the named barrier after it is CTA scope
                            # only, which is why the fence cannot be dropped.
                            cute.arch.fence_acq_rel_gpu()
                            cute.arch.barrier(
                                barrier_id=drain_barrier_id,
                                number_of_threads=num_drain_threads,
                            )
                            barrier.arrive_inc(dQ_lock, drain_tidx, 0, 1)

                    phase ^= 1

                if cutlass.const_expr(need_dKV_lock):
                    # dK / dV for this key block are complete; hand the kv head to the next
                    # query head in the group. Same release pattern as the dQ hand-off:
                    # device-scope fence per thread, then the rendezvous, then one thread
                    # bumps the counter.
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.barrier(
                        barrier_id=drain_barrier_id, number_of_threads=num_drain_threads
                    )
                    barrier.arrive_inc(dKV_lock, drain_tidx, 0, 1)

                if cutlass.const_expr(self.kv_shared and not self.deterministic):
                    # The staged path leaves bulk groups in flight, and their SMEM source is
                    # this CTA's staging buffer, so they have to have READ it before the CTA
                    # exits and the buffer is handed to the next one.
                    if drain_tidx < cute.arch.WARP_SIZE:
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.barrier(
                        barrier_id=drain_barrier_id, number_of_threads=num_drain_threads
                    )
                # Otherwise nothing to drain: red.global.add.v4.f32 is a fire-and-forget
                # reduction with no bulk groups and no SMEM source to protect.
                cute.arch.mbarrier_arrive(mbar_tmem_dealloc)

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)


@functools.lru_cache(maxsize=None)
def bigd_host_config(head_dim: int, head_dim_v: int, cta_group: int = 1):
    """Return the BigD configuration values required by ``interface.py``.

    Args:
        head_dim: Query/key head dimension.
        head_dim_v: Value head dimension.
        cta_group: Number of CTAs participating in each MMA.

    Returns:
        Tile sizes and accumulator slice widths used by host-side allocation and
        postprocessing.

    tile_m / tile_n have to match the interface's m_block_size / n_block_size (they
    size the accumulators and drive the postprocess grid), and the accumulator slice
    widths drive the postprocess loop, so both are read from the kernel object rather
    than duplicated.
    """
    kernel = FlashAttentionBackwardSm100BigD.from_shape(
        head_dim, head_dim_v, cta_group=cta_group
    )
    return {
        "tile_m": kernel.tile_m,
        "tile_n": kernel.tile_n,
        "accum_slice_d": kernel.accum_slice_d,
        "accum_slice_dv": kernel.accum_slice_dv,
    }
