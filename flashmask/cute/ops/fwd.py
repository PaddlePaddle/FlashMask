# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""Forward kernel orchestration.

Framework-agnostic: allocation, dtype and stream access go through a
`flashmask.backend` adapter (``be``).
"""

import math
from typing import Callable, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ...backend import Tensor, get_backend
from .. import utils
from ..barrier import spin_timeout_ns_from_env
from ..block_sparsity import (
    BlockSparseTensorsPaddle,
    normalize_block_sparse_tensors,
    to_cute_block_sparse_tensors,
)
from ..flash_fwd_sm90 import FlashAttentionForwardSm90
from ..flash_fwd_sm100 import FlashAttentionForwardSm100
from ..flashmask_utils import (
    FlashMaskInfoPaddle,
    build_flashmask_block_lists,
    prepare_block_maxmin,
    reduce_block_count,
    to_cute_flashmask_info,
)

from ._common import (
    _dist_world_size,
    _get_overlap_runtime,
    _same_storage,
    maybe_contiguous,
    num_splits_heuristic,
)
from .combine import _flash_attn_fwd_combine

__all__ = ["_flash_attn_fwd"]


# --- moved verbatim from the former cute/interface.py ---
def _flash_attn_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsPaddle] = None,
    return_lse: bool = False,
    out: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
    startend_row_indices: Optional[Tensor] = None,
    block_logit: Optional[Tensor] = None,
    block_size: int = 64,
    block_bos: Optional[Tensor] = None,
    group=None,
) -> Tuple[Tensor, Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        block_logit: Optional pre-allocated fp32 tensor [batch, num_heads, seqlen_q, num_blocks] that,
            when provided, receives the fused per-(query, key-block) max of the post-mask,
            SCALED logit `softmax_scale * q*k^T` -- i.e. the exact value fed
            into softmax. Storing the scaled logit puts every head on one head-independent scale, so a
            downstream `block_logit - LSE` yields log(max attention weight in the block), which is comparable
            across heads for cross-head Top-K block selection. Computed in the softmax epilogue and written in
            place; kept out of the returned tuple. num_blocks must be >= ceil(seqlen_k / block_size). The fused
            reduction respects the same causal / flashmask masking applied to the attention. Only supported on
            SM 10.x (non split-KV).
            IMPORTANT: the kernel only writes key-blocks that the attention loop actually visits. Key-blocks
            that are entirely skipped by masking (e.g. fully-future blocks under causal / flashmask, or blocks
            past seqlen_k when num_blocks*block_size > seqlen_k) are NEVER written. The caller MUST
            pre-initialize block_logit to -inf so those entries read as -inf (masked) rather than stale
            garbage; do not pass an uninitialized / reused buffer.
        block_size: Key-block width (in tokens) used to bucket the fused block-logit reduction. Must divide
            the kernel's n_block_size. Ignored when block_logit is None.
        group: CP group enabling FM-4 overlap. If size > 1, k/v are the local S_local shard and
            the mask key axis has length S_local * group_size (chunk r = rank r); pass it unshifted.
    """
    be = get_backend(q, k, v)

    q, k, v = [maybe_contiguous(t, be) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    batch_size, seqlen_q = q.shape[:2]
    total_q = batch_size * seqlen_q

    compute_capability = (
        be.device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    # --- SM100 large head_dim: the folded-accumulator single pass ---
    # head_dim > 256 keeps head_dim_v whole and runs the folded (m_block_size == 64)
    # accumulator config below. There used to be a fallback that split head_dim_v into
    # several <= 256-wide passes; it is gone because every extra pass costs a FULL
    # redundant Q*K^T. For d=dv=512 the split is 1.5x the useful FLOPs, and at dv=128 it
    # is 2.5x; both measured slower than the single pass by the same order.
    #
    # The features the folded accumulator cannot express are therefore not supported at
    # head_dim_v > 256 -- assert instead of silently taking a measurably slower route. See
    # also the folded_acc asserts in FlashAttentionForwardSm100.__init__.
    if compute_capability == 10 and head_dim > 256 and v.shape[-1] > 256:
        assert v.shape[-1] <= 512, (
            f"head_dim={head_dim}/{v.shape[-1]}: the folded accumulator supports "
            "head_dim_v <= 512 (O takes head_dim_v / 2 TMEM columns)"
        )
        assert num_splits == 1, (
            "head_dim_v > 256 does not support split-KV: the folded accumulator shares "
            "each query row between two threads, which the split-KV row map cannot express"
        )
        assert block_logit is None, (
            "head_dim_v > 256 does not support block_logit (folded accumulator)"
        )
        assert block_sparse_tensors is None, (
            "head_dim_v > 256 does not support block sparsity (folded accumulator)"
        )

    # num_stages=2 exceeds Hopper's per-block shared-memory limit at hdim > 128
    # (cuLaunchKernel returns CUDA_ERROR_INVALID_VALUE), so shrink the N tile to 64.
    # Set BEFORE the flashmask nblock max/min + block-list
    # generation below so prepare_block_maxmin / reduce_block_count /
    # compute_flashmask_block_lists and the kernel all agree on n_block_size
    # (otherwise the per-n_block arrays disagree in length -> reshape error).
    if compute_capability == 9 and head_dim > 128:
        n_block_size = 64
        # d256/dv256 dense fwd is tensor-core bound; a larger KV tile (80 vs 64,
        # matching mainline FA3) cuts KV-loop iterations/overhead and feeds the
        # MMA better, and it still fits Hopper's smem limit because mma_pv_is_rs
        # means there is no P buffer. Only widen the dense
        # (non-flashmask) d256 case: flashmask keeps 64 so the block max/min scan
        # + block-list tiling (kBlockN) agree (and leave headroom for s_rowidx);
        # head_dim=192 keeps 64 (dv=128 tile is a different smem shape).
        if head_dim == 256 and v.shape[-1] == 256 and startend_row_indices is None:
            n_block_size = 80

    # SM100 per-pass config for head_dim=576 (head_dim_v <= 256 after the split above).
    # m_block stays 128: the softmax TMEM load atom needs a QK accumulator with M=128.
    # With m=128 the KV tile must shrink to 32 to fit SMEM, since Q alone takes most of
    # the budget (sO overlaps sQ) and K/V share one buffer.
    # n_block must stay a multiple of 32 (tilePlikeFP32 = n//32*16).
    is_bigd_fwd = compute_capability == 10 and head_dim > 256
    if is_bigd_fwd:
        m_block_size = 128
        n_block_size = 32

    # SM100 big-d (d>256, dv<=256 per pass): the 1-CTA config is forced to n=32 because sQ
    # alone eats most of the SMEM budget at m=128, which leaves far too little
    # FLOP-per-SMEM-byte to saturate the tensor core. A 2-CTA (tcgen05 CTA-pair) UMMA
    # splits the MMA's N across the pair, so sK/sV per CTA is halved and n can double at
    # the same footprint, multiplying the FLOPs per KV tile load.
    #
    # m_block_size stays 128 for every per-pass (dv <= 256) config. The MMA's M is split
    # across the pair, so a CTA owns m_block_size accumulator rows; every TMEM copy in
    # softmax/correction/epilogue is a 128-thread (4-warp) tiled copy whose thread->row map
    # is 1:1 only when the CTA owns 128 rows. At m_block_size=64 CuTe spreads those 64 rows
    # over all 128 TMEM lanes by splitting N (thread t and t+64 then share a row, each
    # holding half of it); that FOLDED layout is what the single-pass config below uses, and
    # it needs the row_max exchange / P-in-SMEM / physical-lane epilogue handling in
    # flash_fwd_sm100.py. Do not set m_block_size=64 for a config that does not want it.
    use_2cta_instrs = False
    if is_bigd_fwd:
        # Enabled for both the dense and the flashmask path. An earlier "flashmask hangs
        # with the CTA pair" report turned out to be the test harness running out of
        # memory while building its reference attn_bias, not a kernel deadlock, so there
        # is no reason to keep flashmask on the slower 1-CTA config.
        use_2cta_instrs = True
    if use_2cta_instrs:
        # m_block_size 128 keeps one accumulator row per thread. 64 folds the accumulator
        # (64 rows spread over all 128 TMEM lanes, N split between the lane halves), which
        # HALVES its TMEM column cost and is the only reason dv=512 fits in ONE pass --
        # i.e. the only way to avoid the redundant QK of a multi-pass split. The price is
        # half the arithmetic intensity per KV tile (the pair covers 128 query rows
        # instead of 256).
        if v.shape[-1] > 256:
            m_block_size = 64
        else:
            m_block_size = 128
        # n=64 doubles the 1-CTA KV tile at identical per-CTA SMEM (the pair splits the MMA's
        # N, so sK/sV per CTA is halved). The folded config additionally requires
        # n_block_size * sizeof(dtype) == 128B, i.e. one P row is exactly one swizzle period
        # of the A-operand SMEM layout (asserted in softmax_loop).
        n_block_size = 64
    fwd_cta_group_size = 2 if use_2cta_instrs else 1

    # Each SM100 CTA processes q_stage * m_block_size query rows; single_q_stage
    # (d>192, d==dv) uses q_stage=1 to fit the TMEM budget. Must match
    # FlashAttentionForwardSm100.q_stage (= 1 if single_q_stage else 2). Computed
    # once here so the flashmask valid_block_count and the block-sparse M-block
    # normalization below share the identical M granularity.
    q_stage = 1 if ((head_dim > 192 and head_dim == v.shape[-1]) or is_bigd_fwd) else 2
    # Query rows per work tile. This is the M granularity that the flashmask
    # valid_block_count / block lists are built at; with a 2-CTA UMMA one work tile is
    # handled by a CTA *pair*, so it covers cta_group_size * q_stage * m_block_size rows.
    fwd_m_tile_rows = q_stage * m_block_size * fwd_cta_group_size
    # FM-4 overlap: when a CP `group` is given, K/V come in LOCAL (B, S_local, H, D)
    # and the gathered KV lives in the NCCL-window SRBuffer. Bootstrap+init the comm
    # singleton once, run the sparse all-gather on the internal comm_stream, then
    # below swap the K/V cute tensors for SRBuffer-backed views with
    # seqlen_k = S_local*nranks. The SM100 load warp gates each K/V tile on the
    # non-splitted AG kernel's write_ptr row frontier.

    # A paddle Group exposes .world_size, a torch ProcessGroup does not;
    # ask the collective adapter instead of duck-typing the handle.
    overlap_nranks = _dist_world_size(be, group) if group is not None else 1
    enable_overlap = overlap_nranks > 1
    if enable_overlap and compute_capability != 10:
        raise NotImplementedError("FM-4 overlap fwd is only supported on SM100")
    overlap_view_args = None
    overlap_bhsd_layout = None
    if enable_overlap:
        overlap_runtime = _get_overlap_runtime()
        overlap_runtime.check_inputs(
            q, k, v, startend_row_indices, causal, overlap_nranks, be
        )
        # One storage for K and V makes the gathered V region a bit-for-bit copy of the
        # gathered K region, so the comm layer transports K only and hands out the K region
        # for V. _same_storage stays last: it raises rather than guess.
        overlap_kv_shared = (
            is_bigd_fwd
            and k.dtype == v.dtype
            and list(k.shape) == list(v.shape)
            and tuple(be.strides(k)) == tuple(be.strides(v))
            and _same_storage(k, v)
        )
        overlap_runtime.ensure_initialized(k, group, kv_shared=overlap_kv_shared)
        # The kernel indexes gathered K/V in this rank's SRBuffer order; reorder the mask to match.
        startend_row_indices = overlap_runtime.sr_order_mask(startend_row_indices, fwd=True)
        overlap_bhsd_layout = overlap_runtime.use_bhsd_layout()
        overlap_stream = overlap_runtime.current_stream_handle(be)
        # Launch AG and retain the gathered SRBuffer view; its FULL S_total shape
        # drives host-side validation and the runtime-dimension kernel arguments.
        # One AG with all acks on the comm stream: an exception after it leaves nothing half-booked.
        overlap_runtime.start_forward_ag(k, v, startend_row_indices, overlap_stream)
        overlap_ag_args = overlap_runtime.forward_ag_args()
        overlap_view_args = overlap_ag_args.view

    cute_flashmask_info = None
    if startend_row_indices is not None:
        fm_batch_size = startend_row_indices.shape[0]
        fm_heads = startend_row_indices.shape[1]
        num_m_blocks = (seqlen_q + fwd_m_tile_rows - 1) // fwd_m_tile_rows
        flashmask_info = FlashMaskInfoPaddle(
            is_causal=causal,
            startend_row_indices=startend_row_indices,
        )
        # valid_block_count (produced by reduce_block_count) feeds only the SM100
        # path and the bwd 2CTA density heuristic. The SM90 fwd kernel iterates the
        # block-sparse lists built below (build_flashmask_block_lists) and never
        # reads valid_block_count, so skip both the extra [b,h,num_m_blocks] alloc
        # and the reduce_block_count kernel launch there. Per-call fixed overhead
        # dominates the high-sparsity / short-seq configs, so this is pure win.
        if compute_capability != 9:
            flashmask_info.valid_block_count = be.empty([fm_batch_size, fm_heads, num_m_blocks], dtype=be.int32)
        prepare_block_maxmin(flashmask_info, kBlockN=n_block_size)
        cute_flashmask_info = to_cute_flashmask_info(flashmask_info)
        if compute_capability != 9:
            reduce_block_count(
                cute_flashmask_info, causal, fwd_m_tile_rows, n_block_size, seqlen_q, be
            )

    if enable_overlap:
        # K/V are still the LOCAL framework tensors here (used only for update_kv +
        # dtype); the kernel reads the gathered SRBuffer, so seqlen_k is the full
        # gathered length S_total = S_local * nranks from the SRBuffer view.
        seqlen_k = overlap_view_args.shape[1]
    else:
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if enable_overlap:
        # Skip the [batch, seqlen_k, ...] shape assert: k/v are LOCAL (S_local),
        # while seqlen_k is the gathered S_total. The SRBuffer view (built below
        # from overlap_view_args) carries the gathered shape into the kernel.
        assert num_head_kv == overlap_view_args.shape[2]
        assert head_dim == overlap_view_args.shape[3]
    else:
        assert list(k.shape) == [batch_size, seqlen_k, num_head_kv, head_dim], (
            f"expect k with shape {[batch_size, seqlen_k, num_head_kv, head_dim]}, received {k.shape=}"
        )
        assert list(v.shape) == [batch_size, seqlen_k, num_head_kv, head_dim_v]

    assert q.dtype in [be.float16, be.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    if learnable_sink is not None:
        assert list(learnable_sink.shape) == [
            num_head,
        ]
        assert learnable_sink.dtype in be.learnable_sink_dtypes, (
            "learnable_sink must be float16, bfloat16 or float32"
        )

    assert all(
        t is None or be.is_gpu(t)
        for t in (
            q,
            k,
            v,
            learnable_sink,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    # head_dim <= 256 for the symmetric / single_q_stage paths. Larger head_dim is the SM100
    # big-d path (single_q_stage with q_stage=1); head_dim_v > 256 stays whole on the folded
    # (m_block_size == 64) accumulator, which supports up to head_dim_v = 512.
    assert (
        head_dim <= 256
        or (compute_capability == 10 and head_dim_v <= 512)
    ), (
        "head_dim must be <= 256, or head_dim>256 with head_dim_v<=512 on SM100 "
        f"(got {head_dim}/{head_dim_v})"
    )
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_framework_dtype = q.dtype
    q_batch_seqlen_shape = [batch_size, seqlen_q]
    lse_shape = [batch_size, num_head, seqlen_q]
    requires_grad = any(be.requires_grad(t) for t in (q, k, v))

    if out is None:
        # then stored, so fully-masked rows/blocks (incl. generate_empty_mask) come
        # out as 0. So on SM90 use an uninitialized buffer to skip the O
        # zero-fill that dominates short-seq / high-sparsity calls (measured fwd speedup).
        # On SM100 the flashmask fwd skips the O store for a fully-masked query block
        # (valid_block_count == 0) and has no zero-writer for it, so that O tile would
        # be left uninitialized -- keep the zero-filled allocation on SM100.
        out = (be.empty if compute_capability == 9 else be.zeros)(
            shape=[*q_batch_seqlen_shape, num_head, head_dim_v], dtype=out_framework_dtype
        )
    else:
        expected_out_shape = [*q_batch_seqlen_shape, num_head, head_dim_v]
        assert list(out.shape) == expected_out_shape, (
            f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        )
        assert out.dtype == out_framework_dtype, (
            f"out tensor dtype {out.dtype} does not match expected dtype {out_framework_dtype}"
        )
        assert be.is_gpu(out), (
            "out tensor must be on a CUDA device to match the inputs"
        )

    if lse is None:
        lse = (
            be.full(shape=lse_shape, fill_value=float('-inf'), dtype=be.float32)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert list(lse.shape) == lse_shape, (
            f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        )
        assert lse.dtype == be.float32, (
            f"lse tensor dtype {lse.dtype} does not match expected dtype be.float32"
        )
        assert be.is_gpu(lse), "lse tensor must be on CUDA device"

    dtype = be.to_cute_dtype(q.dtype)
    learnable_sink_tensor = (
        from_dlpack(learnable_sink.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if learnable_sink is not None
        else None
    )

    # flashmask (SM90): route through the block-sparse list-driven path so the
    # kernel iterates only surviving KV blocks (arbitrary/mid-range skip) while
    # keeping intra_wg_overlap. Build per-(b,h,m_block) surviving-block lists from
    # the flashmask nblock max/min (kBlockM=m_block_size, kBlockN=n_block_size).
    # split_full=True: partially-masked blocks go to the mask list (they get the
    # per-element flashmask apply + rowidx staging), fully-visible blocks go to the
    # full list (no rowidx pipeline / no element mask). This keeps correctness for
    # partial masks (sliding-window etc.) while avoiding the per-block rowidx
    # mbarrier sync on the fully-visible blocks (perf).
    if (
        compute_capability == 9
        and cute_flashmask_info is not None
        and block_sparse_tensors is None
        and seqlen_q is not None
    ):
        fm_full_cnt, fm_full_idx, fm_mask_cnt, fm_mask_idx = build_flashmask_block_lists(
            flashmask_info,
            causal,
            m_block_size,
            n_block_size,
            seqlen_q,
            seqlen_k,
            num_head,
            split_full=True,
        )
        # mask list = partial blocks (element-masked), full list = fully-visible
        # blocks (no element mask). Both are consumed; the producer stages rowidx
        # only for the (contiguous, processed-first) mask-list prefix.
        block_sparse_tensors = BlockSparseTensorsPaddle(
            mask_block_cnt=fm_mask_cnt,
            mask_block_idx=fm_mask_idx,
            full_block_cnt=fm_full_cnt,
            full_block_idx=fm_full_idx,
        )

    sparse_tensors = None
    if block_sparse_tensors is not None:
        if seqlen_q is None:
            raise ValueError(
                "Block sparsity requires fixed-length sequences (seqlen_q must be known)."
            )
        m_block_size_block = m_block_size
        if compute_capability == 10:
            # One SM100 CTA handles q_stage * tile_m rows; keep this in lockstep
            # with the flashmask valid_block_count granularity above.
            m_block_size_block = fwd_m_tile_rows
        expected_m_blocks = (seqlen_q + m_block_size_block - 1) // m_block_size_block
        expected_n_blocks = (seqlen_k + n_block_size - 1) // n_block_size
        block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=(batch_size, num_head, expected_m_blocks),
            expected_index_shape=(batch_size, num_head, expected_m_blocks, expected_n_blocks),
        )
        sparse_tensors = to_cute_block_sparse_tensors(block_sparse_tensors)

    use_block_sparsity = sparse_tensors is not None

    if causal:
        window_size_right = 0
    local = window_size_left is not None or window_size_right is not None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
        else:
            causal, local = False, True

    current_stream = cuda.CUstream(be.cuda_stream())

    # NOTE: do NOT bump n_block_size to 192 for the dense d=128 (non-causal,
    # non-flashmask) case. tile_n=192 with num_stages=2 sits right at Hopper's smem
    # ceiling -> 1 block/SM and near-zero L1, making Full
    # measurably slower than tile_n=128 (FA4 uses 128 and is faster here too).
    # Keep the default 128 to match FA4's dense config.
    if compute_capability == 10:
        if pack_gqa and (128 % qhead_per_kvhead != 0):
            pack_gqa = False
        # TODO: fix GQA + SplitKV
        if pack_gqa and num_splits != 1:
            pack_gqa = False
        # The big-d fwd configs cannot pack q heads into the M dim. PackGQA derives each
        # row's q_head (and its O destination) straight from tidx, which neither single_q_stage's
        # q_stage=1 row map nor the folded (m_block_size=64) accumulator provides -- in the
        # folded layout a row is shared by threads t and t + m_block_size. Gate it here
        # instead of letting the kernel assert. GQA/MQA stays correct via qhead_per_kvhead,
        # just without the packing optimization (which only pays off when seqlen_q per head
        # is too small to fill the M tile, i.e. decode).
        if is_bigd_fwd:
            pack_gqa = False
        # Single q-stage (q_stage=1) for d=dv=256, and for the head_dim=576 per-pass
        # config: both need q_stage=1 so O (head_dim_v cols) fits alongside S/P in the
        # 512-col TMEM.
        single_q_stage = (head_dim > 192 and head_dim == head_dim_v) or is_bigd_fwd

    if num_splits < 1:
        max_seqlen_k = seqlen_k
        max_seqlen_q = seqlen_q
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = (
            max_seqlen_k
            if not local
            else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        )
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size - 1) // m_block_size
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            be.multi_processor_count(),
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = be.empty(
            shape=[num_splits, *q_batch_seqlen_shape, num_head, head_dim_v], dtype=be.float32
        )
        lse_partial = be.empty(shape=[num_splits, *lse_shape], dtype=be.float32)

    q_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, out if not is_split_kv else out_partial)
    ]
    if enable_overlap:
        # K/V live in the NCCL-window SRBuffer (gathered device memory), so there is no
        # dlpack capsule to wrap; the kernel builds the views from their addr instead.
        k_tensor = None
        v_tensor = None
    else:
        k_tensor, v_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (k, v)
        ]
    if is_split_kv:
        lse_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse_partial.ndim - 1
        )
    elif lse is not None:
        lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
    else:
        lse_tensor = None

    if block_logit is not None:
        assert compute_capability == 10, (
            "block_logit (fused block-score) is only supported on SM 10.x"
        )
        assert not is_split_kv, "block_logit is not supported with split-KV"
        assert not pack_gqa, (
            "block_logit requires pack_gqa=False. block_logit is indexed by the "
            "query head (head_idx) and the query row (m_block*m_block_size+tidx); "
            "under pack_gqa=True the query heads are packed into the M/row dim and "
            "head_idx is the KV head, so the block-score write would target wrong "
            "locations. NOTE: pack_gqa defaults to (qhead_per_kvhead > 1), so GQA "
            "callers MUST pass pack_gqa=False explicitly when requesting block_logit."
        )
        assert block_logit.dtype == be.float32, (
            f"block_logit must be float32; got {block_logit.dtype}"
        )
        assert be.is_gpu(block_logit), "block_logit must be on CUDA"
        assert block_logit.ndim == 4, (
            f"block_logit must be [batch, num_heads, seqlen_q, num_blocks]; got ndim={block_logit.ndim}"
        )
        block_seqlen_k = seqlen_k if enable_overlap else k.shape[1]
        _nb_min = (block_seqlen_k + block_size - 1) // block_size
        assert block_logit.shape[-1] >= _nb_min, (
            f"block_logit num_blocks={block_logit.shape[-1]} < ceil(seqlen_k/block_size)={_nb_min}"
        )
        block_logit_tensor = from_dlpack(
            block_logit.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=block_logit.ndim - 1)
    else:
        block_logit_tensor = None

    # Optional per-query document start (bos) for DOCUMENT-relative block
    # bucketing (pack-equivalence). Without it the kernel buckets by absolute
    # packed-sequence block, which is only correct for single-document (bos==0)
    # inputs. Shape [B, S] int32, aligned with block_logit's query dim.
    if block_bos is not None:
        assert block_logit is not None, (
            "block_bos requires block_logit (it drives its relative bucketing)"
        )
        assert block_bos.dtype == be.int32, (
            f"block_bos must be int32; got {block_bos.dtype}"
        )
        assert be.is_gpu(block_bos), "block_bos must be on CUDA"
        assert block_bos.ndim == 2, (
            f"block_bos must be [B, S]; got ndim={block_bos.ndim}"
        )
        assert list(block_bos.shape) == [batch_size, seqlen_q], (
            f"block_bos must be [B={batch_size}, S={seqlen_q}]; got {block_bos.shape}"
        )
        block_bos_tensor = from_dlpack(
            block_bos.detach(), assumed_align=4
        ).mark_layout_dynamic(leading_dim=block_bos.ndim - 1)
    else:
        block_bos_tensor = None

    if use_block_sparsity:
        if pack_gqa:
            raise NotImplementedError(
                "Block sparsity is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )


    # Build SRBuffer views inside the MLIR context. Dimensions stay runtime Int32;
    # making this layout static changes the TMA descriptor and reads wrong bytes.
    if enable_overlap:
        overlap_view = overlap_ag_args.view
        overlap_k_addr = cutlass.Int64(overlap_view.k_addr)
        overlap_v_addr = cutlass.Int64(overlap_view.v_addr)
        overlap_write_ptr_addr = cutlass.Int64(overlap_ag_args.write_ptr_addr)
        _ob, _os, _oh, _od = overlap_view.shape
        overlap_b = cutlass.Int32(_ob)
        overlap_s = cutlass.Int32(_os)
        overlap_h = cutlass.Int32(_oh)
        overlap_d = cutlass.Int32(_od)
        overlap_kv_chunk_size = overlap_ag_args.kv_chunk_size
        overlap_spin_timeout_ns = cutlass.Int64(spin_timeout_ns_from_env())
    else:
        overlap_k_addr = None
        overlap_v_addr = None
        overlap_write_ptr_addr = None
        overlap_b = None
        overlap_s = None
        overlap_h = None
        overlap_d = None
        overlap_kv_chunk_size = None
        overlap_spin_timeout_ns = None
        overlap_bhsd_layout = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        use_block_sparsity,
        lse is None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        be.to_cute_dtype(learnable_sink.dtype) if learnable_sink is not None else None,
        m_block_size,
        n_block_size,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        # flashmask
        startend_row_indices.shape[3] if startend_row_indices is not None else None,
        single_q_stage if compute_capability == 10 else False,
        use_2cta_instrs,
        block_logit is None,
        block_size,
        block_bos is None,
    ) + (
        # SRBuffer K/V require a distinct artifact only when overlap is active.
        (overlap_bhsd_layout, overlap_kv_chunk_size) if enable_overlap else ()
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        if compute_capability == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=m_block_size,
                tile_n=n_block_size,
                # num_stages=1,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=True,
                mma_pv_is_rs=True,
            )
        elif compute_capability == 10:
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                is_persistent=not causal
                and not local
                and not is_split_kv,
                single_q_stage=single_q_stage,
                has_block_logit=block_logit is not None,
                block_size=block_size,
                has_block_bos=block_bos is not None,
                use_2cta_instrs=use_2cta_instrs,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink_tensor,
            blocksparse_tensors=sparse_tensors,
            flashmask_info=cute_flashmask_info,
            **(
                {
                    "mBlockLogit": block_logit_tensor,
                    "mBlockBos": block_bos_tensor,
                }
                if compute_capability == 10
                else {}
            ),
            **(
                {
                    "overlap_k_addr": overlap_k_addr,
                    "overlap_v_addr": overlap_v_addr,
                    "overlap_write_ptr_addr": overlap_write_ptr_addr,
                    "overlap_b": overlap_b,
                    "overlap_s": overlap_s,
                    "overlap_h": overlap_h,
                    "overlap_d": overlap_d,
                    "overlap_spin_timeout_ns": overlap_spin_timeout_ns,
                    "overlap_kv_chunk_size": overlap_kv_chunk_size,
                    "overlap_bhsd_layout": overlap_bhsd_layout,
                }
                if enable_overlap
                else {}
            ),
            stream=current_stream,
        )
    # Runtime address and shape scalars are re-supplied below; compile-time
    # overlap_kv_chunk_size is captured by the compiled callable.

    _flash_attn_fwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        learnable_sink=learnable_sink_tensor,
        blocksparse_tensors=sparse_tensors,
        flashmask_info=cute_flashmask_info,
        **(
            {
                "mBlockLogit": block_logit_tensor,
                "mBlockBos": block_bos_tensor,
            }
            if compute_capability == 10
            else {}
        ),
        **(
            {
                "overlap_k_addr": overlap_k_addr,
                "overlap_v_addr": overlap_v_addr,
                "overlap_write_ptr_addr": overlap_write_ptr_addr,
                "overlap_b": overlap_b,
                "overlap_s": overlap_s,
                "overlap_h": overlap_h,
                "overlap_d": overlap_d,
                "overlap_spin_timeout_ns": overlap_spin_timeout_ns,
            }
            if enable_overlap
            else {}
        ),
        stream=current_stream,
    )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}
