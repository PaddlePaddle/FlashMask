# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""Backward kernel orchestration (preprocess / main / postprocess / dsink).

Framework-agnostic: allocation, dtype and stream access go through a
`flashmask.backend` adapter (``be``).
"""

import math
from typing import Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flashmask.backend import Tensor, get_backend
from flashmask.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flashmask.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flashmask.cute.flash_bwd_sink import FlashAttentionBackwardDsink
from flashmask.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flashmask.cute.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flashmask.cute.flash_bwd_sm100_bigd import (
    FlashAttentionBackwardSm100BigD,
    bigd_host_config,
)
from flashmask.cute.flashmask_utils import (
    FlashMaskInfoPaddle,
    prepare_block_maxmin,
    to_cute_flashmask_info,
)

from ._common import (
    _dist_world_size,
    _get_overlap_runtime,
    _make_fake_bwd_tensors,
    _same_storage,
    _tile_size_bwd_sm90,
    maybe_contiguous,
)

try:
    from flashmask.utils import accum_zero_axis1_kv
except ImportError:
    accum_zero_axis1_kv = None

__all__ = ["_flash_attn_bwd"]


# --- moved verbatim from the former cute/interface.py ---
def _flash_attn_bwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    flashmask_info: Optional[Union[FlashMaskInfoPaddle, Tensor]] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    learnable_sink: Optional[Tensor] = None,
    deterministic: bool = False,
    kv_postprocess_start: Optional[int] = None,
    kv_postprocess_end: Optional[int] = None,
    group=None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """Backward pass for FlashAttention.

    group: CP group enabling the FM-4 overlap. With a group of size > 1, k/v are the local
    S_local shard, dk/dv come back reduce-scattered to S_local, and the mask key axis (dim 2)
    has length S_local * group_size in group-rank chunk order. The library converts it to SR
    order; do NOT pre-roll it.
    """
    be = get_backend(q, k, v)
    compute_capability = be.device_capability()[0]
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    num_head, head_dim = q.shape[-2:]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]

    # Large head dims (MLA-shaped 576/512 and friends) go through a separate SM100
    # backward kernel: dV (512 cols) + dK/dQ (576 cols each) + S/dP do not fit the
    # 512-col TMEM budget, so that kernel chunks the head_dim axes and drains every
    # output through the fp32 gmem accumulators. See flash_bwd_sm100_bigd.py.
    is_bigd_bwd = compute_capability == 10 and (head_dim > 256 or head_dim_v > 256)
    assert is_bigd_bwd or (head_dim <= 256 and head_dim_v <= 256), (
        f"backward does not support head_dim={head_dim}, head_dim_v={head_dim_v} "
        f"on sm_{compute_capability}0"
    )

    # KV shared: the caller handed the SAME buffer as k and v (the MLA / sparse-attn
    # convention, v = kv[..., :head_dim_v]). Then dV and dK are two halves of ONE
    # gradient -- autograd adds the grad of the aliased input twice -- so the kernel can
    # accumulate them into a single TMEM slot and flush them once. Detected instead of
    # asked for: the signature stays (q, k, v).
    #
    # dv then comes back all-zero and dk carries dK + dV: the kernel never writes
    # dv_accum, so dv is allocated zeroed and its postprocess is skipped.
    #
    # Only the shapes the merge is implemented for. 576/512 has no chunk width that
    # divides both axes, so the kernel pads the dv axis to 576 and lets the dO TMA
    # zero-fill columns 512..575; that keeps d_chunk at the measured 192 instead
    # of narrowing it.
    kv_shared = (
        is_bigd_bwd
        and (head_dim, head_dim_v) in ((512, 512), (576, 512))
        and k.dtype == v.dtype
        and list(k.shape[:-1]) == list(v.shape[:-1])
        and v.shape[-1] <= k.shape[-1]
        and tuple(be.strides(k)) == tuple(be.strides(v))
        and _same_storage(k, v)
    )

    m_block_size = 128
    n_block_size = 128

    bigd_cfg = None
    if is_bigd_bwd:
        # The kernel solves its own tile config; the accumulator shapes and the
        # postprocess grid below are built from m_block_size / n_block_size, so they
        # have to agree with it.
        bigd_cfg = bigd_host_config(head_dim, head_dim_v)
        m_block_size = bigd_cfg["tile_m"]
        n_block_size = bigd_cfg["tile_n"]

    # SM100 d=256/dv=256 runs the 2-CTA backward, whose accumulators are folded
    # (64 rows per CTA). Folding is what halves their TMEM column cost and lets dK
    # and dV both stay resident -- no dK reduce, no dKV postprocess -- and it
    # requires a 64-row KV tile. Set here, BEFORE prepare_block_maxmin / the
    # block-list generation below, for the same reason as the SM90 case.
    if compute_capability == 10 and head_dim == 256 and head_dim_v == 256:
        n_block_size = 64

    # SM90: finalize n_block_size (the head_dim-dependent bwd N tile from
    # _tile_size_bwd_sm90, e.g. 64 for head_dim=256) BEFORE prepare_block_maxmin /
    # the block-list generation below, so they all agree with the actual kernel's N
    # tile (same class of bug as the fwd n_block_size fix: otherwise the flashmask
    # per-n_block max/min arrays are sized/scanned with kBlockN=128 while the kernel
    # and any block-list consumer expect the real n_block_size -> wrong blocks used
    # -> wrong dQ/dK/dV, or a shape mismatch if list-driven).
    if compute_capability == 9:
        n_block_size = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            False,
            sparse_block_size_q=None,
            flashmask=flashmask_info is not None,
            deterministic=deterministic,
        ).n_block_size

    # FM-4 backward consumes one split-AG segment at a time and gates each KV tile
    # on the producer's per-work completion bitmap.
    # A paddle Group exposes .world_size, a torch ProcessGroup does not;
    # ask the collective adapter instead of duck-typing the handle.
    enable_overlap = group is not None and _dist_world_size(be, group) > 1
    if enable_overlap:
        if compute_capability != 10:
            raise NotImplementedError("FM-4 overlap bwd is only supported on SM100")
        overlap_runtime = _get_overlap_runtime()
        assert flashmask_info is not None, (
            "overlap bwd requires flashmask_info (group-rank KV-chunk order)"
        )
        assert kv_postprocess_start is None and kv_postprocess_end is None, (
            "overlap bwd owns the KV segment postprocess range"
        )
        assert k.dtype == be.bfloat16, "overlap SRBuffer is bf16"
        assert head_dim_v == head_dim, (
            "the SRBuffer is a single K/V pair of equal-width slabs, so the V view is "
            "built from K's head_dim"
        )
        assert not causal, "overlap bwd does not support causal yet"
        if isinstance(flashmask_info, FlashMaskInfoPaddle):
            flashmask_info = flashmask_info.startend_row_indices
        overlap_runtime.ensure_initialized(k, v, group, kv_shared=kv_shared)
        # Permute before the block max/min below: the chunk-skip mask and the kernel
        # both index the gathered K/V in this rank's SRBuffer order.
        flashmask_info = overlap_runtime.sr_order_mask(flashmask_info, fwd=False)

    cute_flashmask_info = None
    num_flashmask_tensors = 0

    if flashmask_info is not None and isinstance(flashmask_info, be.tensor_types):
        flashmask_info = FlashMaskInfoPaddle(
            startend_row_indices=flashmask_info,
            is_causal=causal,
        )
    if flashmask_info is not None:
        assert isinstance(flashmask_info, FlashMaskInfoPaddle)
        # No valid_block_count here: unlike the forward, no backward kernel reads it.
        # It only ever fed a host-side density heuristic; the 2CTA path now skips
        # fully-masked m blocks itself, so that heuristic and the block-count scan
        # behind it are both gone.
        prepare_block_maxmin(flashmask_info, kBlockN=n_block_size)
        cute_flashmask_info = to_cute_flashmask_info(flashmask_info)
        num_flashmask_tensors = 2 * flashmask_info.startend_row_indices.shape[-1]

    overlap_view_args = None
    overlap_bhsd_layout = None
    overlap_segment_idx = None
    if enable_overlap:
        startend_row_indices = flashmask_info.startend_row_indices
        overlap_bhsd_layout = overlap_runtime.use_bhsd_layout()
        overlap_stream = overlap_runtime.current_stream_handle(be)
        overlap_ag_args = overlap_runtime.start_backward_ag(
            k, v, startend_row_indices, overlap_stream
        )
        overlap_view_args = overlap_ag_args.kv_view(0)
        segment_seqlen = overlap_ag_args.segment_seqlen
        full_seqlen_k = startend_row_indices.shape[2]
        assert full_seqlen_k == segment_seqlen * overlap_ag_args.num_segments
        assert segment_seqlen % n_block_size == 0
        segment_nblocks = segment_seqlen // n_block_size
        assert segment_nblocks % 4 == 0, (
            "FM-3 segment mask metadata requires a 4-block-aligned segment"
        )
        # Keep the full mask tensors. The SM100 loader applies the segment offset
        # while indexing, preserving the original batch/head stride without copies.
        cute_flashmask_info = to_cute_flashmask_info(flashmask_info)
        # All segment-local scheduler and semaphore shapes use this length. The
        # original local length is retained by k/v and by the final dK/dV outputs.
        seqlen_k = segment_seqlen

    if compute_capability == 9:
        sparse_q = None
        local = False
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
            flashmask=cute_flashmask_info is not None,
            deterministic=deterministic,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
    else:
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1

        # The big-headdim kernel always runs 1 CTA per MMA; the other SM100 kernel
        # uses a 2-CTA cluster for large head_dim (or d==128 dense).
        need_large_cluster = (head_dim > 128) or (head_dim == 128 and flashmask_info is None)
        if is_bigd_bwd:
            cluster_size = 1
        else:
            cluster_size = 2 if need_large_cluster else 1
        use_2cta_instrs = cluster_size == 2

    q, k, v, out, dout, lse = [
        maybe_contiguous(t, be)
        for t in (q, k, v, out, dout, lse)
    ]

    batch_size, seqlen_q = q.shape[:2]
    total_q = batch_size * seqlen_q

    batch_size, seqlen_k = k.shape[:2]
    total_k = batch_size * seqlen_k

    if enable_overlap:
        # k/v are local inputs, while the grad kernel consumes one gathered SRBuffer
        # segment at a time. The segment view supplies the scheduler and dK/dV shape.
        seqlen_k = overlap_view_args.shape[1]
        total_k = batch_size * seqlen_k
        assert num_head_kv == overlap_view_args.shape[2]
        assert head_dim == overlap_view_args.shape[3]
    else:
        assert list(k.shape) == [batch_size, seqlen_k, num_head_kv, head_dim]
        assert list(v.shape) == [batch_size, seqlen_k, num_head_kv, head_dim_v]

    assert list(out.shape) == [batch_size, seqlen_q, num_head, head_dim_v]
    assert list(dout.shape) == [batch_size, seqlen_q, num_head, head_dim_v]
    assert list(lse.shape) == [batch_size, num_head, seqlen_q], (
        "lse must have shape (batch_size, num_head, seqlen_q)"
    )

    assert q.dtype in [be.float16, be.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    assert lse.dtype == be.float32, "lse must be float32"
    assert all(
        t is None or be.is_gpu(t)
        for t in (q, k, v, out, dout, lse)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256 or is_bigd_bwd, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if compute_capability == 10:
        pack_gqa = False  # override for now

    # TODO: check if this is the right rounding
    # Round head_dim to multiple of 64 for SM100 to ensure tiled_copy_2d compatibility
    # in postprocess (128 threads must divide tile_hdim/copy_elems evenly)
    hdim_round_to = 64 if compute_capability == 10 else 32
    head_dim_rounded = (head_dim + hdim_round_to - 1) // hdim_round_to * hdim_round_to
    head_dim_v_rounded = (head_dim_v + hdim_round_to - 1) // hdim_round_to * hdim_round_to

    # dq: dq_accum -> dq postprocess always writes the full m_block range, so empty_like
    # is safe on fixed-seqlen path and avoids a redundant bf16 fill.
    # dk/dv: only safe to skip the zero-fill when postprocess writes every row, i.e.
    # when dk_accum/dv_accum is in use (qhead_per_kvhead > 1 or big-d). In the
    # GQA-ratio==1 path the main bwd kernel writes dk/dv directly, and
    # FlashMask can skip whole n_blocks (no Q rows attend) — those rows would stay
    # garbage with empty_like and break correctness. Fall back to zeros_like there.
    kv_postprocess_full = (
        (qhead_per_kvhead > 1) or is_bigd_bwd
    )
    dq = be.empty_like(q)
    # Native RS also writes the final local dK/dV directly into these tensors.
    kv_write_full = enable_overlap or kv_postprocess_full
    dk = be.empty_like(k) if kv_write_full else be.zeros_like(k)
    # kv_shared: dk carries dK + dV and nothing writes dv, so it is the zero gradient.
    dv = be.empty_like(v) if kv_write_full and not kv_shared else be.zeros_like(v)

    # ---- Compute shapes for fp32 accum workspaces ----
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    dq_accum_shape = [batch_size, num_head, seqlen_q_rounded * head_dim_rounded]
    dpsum_shape = [batch_size, num_head, seqlen_q_rounded]
    dpsum = be.empty(shape=dpsum_shape, dtype=be.float32)
    lse_log2 = be.empty(shape=dpsum_shape, dtype=be.float32)

    # The big-headdim kernel never writes dK / dV directly: both leave TMEM one
    # head_dim chunk at a time through the fp32 accumulators.
    need_kv_accum = (
        qhead_per_kvhead > 1 or is_bigd_bwd
    )
    if need_kv_accum:
        seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
        num_n_blocks = seqlen_k_rounded // n_block_size
        if cluster_size == 2 and num_n_blocks % cluster_size != 0:
            seqlen_k_rounded = seqlen_k_rounded + n_block_size
        dk_accum_shape = [batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded]
        dv_accum_shape = [batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded]

    kv_postprocess_enabled = kv_postprocess_start is not None or kv_postprocess_end is not None

    def _kv_postprocess_range():
        if not kv_postprocess_enabled:
            return 0, seqlen_k
        start = 0 if kv_postprocess_start is None else int(kv_postprocess_start)
        end = seqlen_k if kv_postprocess_end is None else int(kv_postprocess_end)
        assert 0 <= start <= end <= seqlen_k, (
            f"invalid kv_postprocess range [{start}, {end}) for seqlen_k={seqlen_k}"
        )
        assert start % n_block_size == 0, (
            f"kv_postprocess_start must be aligned to n_block_size={n_block_size}, got {start}"
        )
        assert end % n_block_size == 0, (
            f"kv_postprocess_end must be aligned to n_block_size={n_block_size}, got {end}"
        )
        return start, end

    kv_post_start, kv_post_end = _kv_postprocess_range()


    # ---- Compute shapes for fp32 accum workspaces ----
    def _numel(shape):
        n = 1
        for d in shape:
            n *= d
        return n

    zero_kv_accum_range = kv_postprocess_enabled and need_kv_accum and accum_zero_axis1_kv is not None
    zero_specs = []  # list of (key, shape, numel)
    if need_kv_accum and not zero_kv_accum_range:
        zero_specs.append(("dk_accum", dk_accum_shape, _numel(dk_accum_shape)))
        if not kv_shared:
            zero_specs.append(("dv_accum", dv_accum_shape, _numel(dv_accum_shape)))

    _accum_buffers = {}
    if len(zero_specs) >= 2:
        _zero_total = sum(s[2] for s in zero_specs)
        _zero_big = be.zeros(shape=[_zero_total], dtype=be.float32)
        _off = 0
        for key, shape, numel in zero_specs:
            _accum_buffers[key] = _zero_big[_off : _off + numel].reshape(shape)
            _off += numel
        # Keep _zero_big alive in this frame so views remain valid.
    elif len(zero_specs) == 1:
        key, shape, _ = zero_specs[0]
        _accum_buffers[key] = be.zeros(shape=shape, dtype=be.float32)

    dq_accum = be.empty(shape=dq_accum_shape, dtype=be.float32)
    if need_kv_accum:
        if zero_kv_accum_range:
            # if start/end range is given, we can initialize only part of the accum buffer
            dk_accum = be.empty(shape=dk_accum_shape, dtype=be.float32)
            dv_accum = be.empty(shape=dv_accum_shape, dtype=be.float32)
            dk_zero_hdim, dv_zero_hdim = head_dim_rounded, head_dim_v_rounded
            dk_split, dv_split = False, False
            accum_zero_axis1_kv(
                dk_accum,
                dv_accum,
                kv_post_start,
                kv_post_end - kv_post_start,
                dk_zero_hdim,
                dv_zero_hdim,
                dk_split,
                dv_split,
            )
        else:
            dk_accum = _accum_buffers["dk_accum"]
            # Nothing writes dv_accum under kv_shared, so it aliases dk_accum rather
            # than costing a second fp32 buffer.
            dv_accum = dk_accum if kv_shared else _accum_buffers["dv_accum"]

    dtype = be.to_cute_dtype(q.dtype)
    q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out, dout, dq, dk, dv)
    ]
    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse.ndim - 1
    )
    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (dq_accum, dpsum, lse_log2)
    ]
    if need_kv_accum:
        dk_accum_tensor, dv_accum_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (dk_accum, dv_accum)
        ]
    if deterministic:
        dQ_semaphore = be.zeros(
            shape=[batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size], dtype=be.int32
        )
    else:
        dQ_semaphore = None

    if deterministic and (qhead_per_kvhead > 1):
        dK_semaphore = be.zeros(
            shape=[batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2], dtype=be.int32
        )
        dV_semaphore = be.zeros(
            shape=[batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2], dtype=be.int32
        )
    else:
        dK_semaphore = None
        dV_semaphore = None

    # The framework does not expose the physical layout, so assert the tensor is contiguous here.
    if dQ_semaphore is not None:
        assert dQ_semaphore.is_contiguous()
    if dK_semaphore is not None:
        assert dK_semaphore.is_contiguous()
    if dV_semaphore is not None:
        assert dV_semaphore.is_contiguous()
    # Must match the compile-time fake semaphore layout (make_fake_tensor in
    # _make_fake_bwd_tensors: leading_dim = ndim-1, other strides sym_int64/dynamic),
    # exactly like every other bwd tensor above. Using convert_from_dlpack_leading_static
    # here (compact_shape_dynamic) mismatched the traced fake layout, so at runtime the
    # kernel read shape values as strides -> garbage semaphore address -> the dQ
    # deterministic wait_eq spun on uninitialized memory and hung for n_block >= 2.
    dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)
        if t is not None
        else None
        for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
    ]
    current_stream = cuda.CUstream(be.cuda_stream())

    # Rebuild segment-local SRBuffer views inside the MLIR context. Split AG
    # publishes one completion flag per communication work item.
    if enable_overlap:
        overlap_view = overlap_view_args
        overlap_k_addr = cutlass.Int64(overlap_view.k_addr)
        overlap_v_addr = cutlass.Int64(overlap_view.v_addr)
        overlap_work_done_addr = cutlass.Int64(overlap_ag_args.work_done_addr)
        _ob, _os, _oh, _od = overlap_view.shape
        overlap_b = cutlass.Int32(_ob)
        overlap_s = cutlass.Int32(_os)
        overlap_h = cutlass.Int32(_oh)
        overlap_d = cutlass.Int32(_od)
        overlap_segment_idx = cutlass.Int32(0)
        overlap_comm_rpb = overlap_ag_args.comm_rpb
        # The bwd gate waits only on the work item holding a cluster tile's last row.
        assert overlap_comm_rpb % (cluster_size * n_block_size) == 0, (
            f"comm_rpb {overlap_comm_rpb} must be a multiple of cluster_size * n_block_size "
            f"({cluster_size} * {n_block_size})"
        )
        overlap_dk_send_addr, overlap_dv_send_addr = overlap_ag_args.dkv_send_addrs(0)
        overlap_dk_addr = (
            None if need_kv_accum else cutlass.Int64(overlap_dk_send_addr)
        )
        overlap_dv_addr = (
            None if need_kv_accum else cutlass.Int64(overlap_dv_send_addr)
        )
    else:
        overlap_k_addr = None
        overlap_v_addr = None
        overlap_work_done_addr = None
        overlap_dk_addr = None
        overlap_dv_addr = None
        overlap_b = None
        overlap_s = None
        overlap_h = None
        overlap_d = None
        overlap_comm_rpb = None
        overlap_bhsd_layout = None

    compile_key_pre = (compute_capability, dtype, head_dim, head_dim_v, head_dim_rounded, m_block_size, num_threads)
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype,
            head_dim,
            head_dim_v,
            m_block_size,
            dq_head_dim=head_dim_rounded,
        )
        # Compile with FA4-style fake tensors (all dims are sym_int, strides have
        # divisibility=8/4 → 128-bit alignment statically guaranteed).
        (
            f_mQ, f_mK, f_mV, f_mO, f_mdO, f_mdQ, f_mdK, f_mdV,
            f_mLSE, f_mLSElog2, f_mPdPsum, f_mdQaccum, f_mdKaccum, f_mdVaccum,
            _f_mdQ_semaphore, _f_mdK_semaphore, _f_mdV_semaphore,
        ) = _make_fake_bwd_tensors(dtype, has_gqa=qhead_per_kvhead > 1)
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            f_mO,
            f_mdO,
            f_mPdPsum,
            f_mLSE,
            f_mLSElog2,
            f_mdQaccum,
            None,  # mdLSE
            current_stream,
        )
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        o_tensor,
        do_tensor,
        dpsum_tensor,
        lse_tensor,
        lse_log2_tensor,
        dq_accum_tensor,
        None,  # mdLSE
        current_stream,
    )

    # Backward kernel: compute dk, dv, dq_accum.
    if compute_capability == 9:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            num_flashmask_tensors,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            deterministic,
        )
    else:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            num_flashmask_tensors,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            deterministic,
            # kv_shared changes the output gemms and the drain, so it is a different
            # compiled artifact for the same shapes.
            kv_shared,
            # overlap: an overlap grad kernel (K/V rebuilt from SRBuffer addr) and a
            # plain one for the same shapes are different compiled artifacts.
            enable_overlap,
            overlap_bhsd_layout,
            overlap_comm_rpb,
        )

    # SM100/SM110 uses default from function signature (384).
    if compute_capability not in [9, 12]:
        num_threads = 384

    if compile_key not in _flash_attn_bwd.compile_cache:
        if compute_capability == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=False,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                # subtile_factor=subtile_factor,
                # dQ_single_wg=dQ_single_wg,
            )

            # Compile with FA4-style fake tensors (fully-symbolic dims with stride
            # divisibility hints). Pass the real cute_flashmask_info at compile so
            # the flashmask code path (const_expr enable flags) is generated; its
            # tensors are marked layout-dynamic so the kernel stays shape-generic.
            (
                f_mQ, f_mK, f_mV, f_mO_unused, f_mdO,
                f_mdQ_unused, f_mdK, f_mdV,
                f_mLSE_unused, f_mLSElog2, f_mPdPsum,
                f_mdQaccum, f_mdKaccum, f_mdVaccum,
                f_mdQ_semaphore, f_mdK_semaphore, f_mdV_semaphore,
            ) = _make_fake_bwd_tensors(
                dtype, has_gqa=qhead_per_kvhead > 1, deterministic=deterministic, cluster_size=1
            )
            _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
                fa_bwd_obj,
                f_mQ,
                f_mK,
                f_mV,
                f_mdO,
                f_mLSElog2,
                f_mPdPsum,
                f_mdQaccum,
                f_mdK if not need_kv_accum else f_mdKaccum,
                f_mdV if not need_kv_accum else f_mdVaccum,
                softmax_scale,
                mdQ_semaphore=f_mdQ_semaphore,
                mdK_semaphore=f_mdK_semaphore,
                mdV_semaphore=f_mdV_semaphore,
                flashmask_info=cute_flashmask_info,
                stream=current_stream,
            )
        else:
            if is_bigd_bwd:
                # Solves its own tile / chunk config from (head_dim, head_dim_v); the
                # remaining knobs are 1-CTA only for now. fm_bound_num has to come in
                # here because the startend_row_indices tensor reaches the kernel with a
                # fully dynamic layout, so its width is not visible at trace time.
                fa_bwd_obj = FlashAttentionBackwardSm100BigD.from_shape(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    qhead_per_kvhead=qhead_per_kvhead,
                    deterministic=deterministic,
                    kv_shared=kv_shared,
                    fm_bound_num=(
                        0
                        if cute_flashmask_info is None
                        else flashmask_info.startend_row_indices.shape[-1]
                    ),
                )
                assert not overlap_bhsd_layout, (
                    "bigd bwd reads a BSHD SRBuffer; the communicator picks BHSD only for D <= 256"
                )
                layout_kwargs = {}
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                )
                layout_kwargs = {"overlap_bhsd_layout": overlap_bhsd_layout}
            _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
                fa_bwd_obj,
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_tensor if not need_kv_accum else dk_accum_tensor,
                dv_tensor if not need_kv_accum else dv_accum_tensor,
                softmax_scale,
                mdQ_semaphore=dQ_semaphore_tensor,
                mdK_semaphore=dK_semaphore_tensor,
                mdV_semaphore=dV_semaphore_tensor,
                flashmask_info=cute_flashmask_info,
                overlap_k_addr=overlap_k_addr,
                overlap_v_addr=overlap_v_addr,
                overlap_work_done_addr=overlap_work_done_addr,
                overlap_segment_idx=overlap_segment_idx,
                overlap_dk_addr=overlap_dk_addr,
                overlap_dv_addr=overlap_dv_addr,
                overlap_b=overlap_b,
                overlap_s=overlap_s,
                overlap_h=overlap_h,
                overlap_d=overlap_d,
                overlap_comm_rpb=overlap_comm_rpb,
                **layout_kwargs,
                stream=current_stream,
            )
    if compute_capability == 9:
        _flash_attn_bwd.compile_cache[compile_key](
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not need_kv_accum else dk_accum_tensor,
            dv_tensor if not need_kv_accum else dv_accum_tensor,
            softmax_scale,
            mdQ_semaphore=dQ_semaphore_tensor,
            mdK_semaphore=dK_semaphore_tensor,
            mdV_semaphore=dV_semaphore_tensor,
            flashmask_info=cute_flashmask_info,
            stream=current_stream,
        )
    else:
        def _run_bwd_main(
            segment_flashmask_info,
            segment_k_addr,
            segment_v_addr,
            segment_dk_addr,
            segment_dv_addr,
            segment_idx=None,
        ):
            _flash_attn_bwd.compile_cache[compile_key](
                q_tensor,
                k_tensor,
                v_tensor,
                do_tensor,
                lse_log2_tensor,
                dpsum_tensor,
                dq_accum_tensor,
                dk_tensor if not need_kv_accum else dk_accum_tensor,
                dv_tensor if not need_kv_accum else dv_accum_tensor,
                softmax_scale,
                mdQ_semaphore=dQ_semaphore_tensor,
                mdK_semaphore=dK_semaphore_tensor,
                mdV_semaphore=dV_semaphore_tensor,
                flashmask_info=segment_flashmask_info,
                overlap_k_addr=segment_k_addr,
                overlap_v_addr=segment_v_addr,
                overlap_work_done_addr=overlap_work_done_addr,
                overlap_segment_idx=(
                    None if segment_idx is None else cutlass.Int32(segment_idx)
                ),
                overlap_dk_addr=segment_dk_addr,
                overlap_dv_addr=segment_dv_addr,
                overlap_b=overlap_b,
                overlap_s=overlap_s,
                overlap_h=overlap_h,
                overlap_d=overlap_d,
                stream=current_stream,
            )

        if not enable_overlap:
            _run_bwd_main(
                cute_flashmask_info,
                overlap_k_addr,
                overlap_v_addr,
                overlap_dk_addr,
                overlap_dv_addr,
            )

    num_threads = 256 if compute_capability == 9 else 128
    arch = compute_capability * 10

    # The dK/dV postprocess decodes the fp32 accum with its 2-CTA branch in
    # flash_bwd_postprocess.py, which hard-codes row_groups = 2: it reads a
    # panel of (128 threads x ncol) values and interprets threads 0..63 as rows
    # 0..63 of the block and threads 64..127 as the SAME 64 rows at hdim + 128.
    # That is exactly what the folded 2CTA bwd kernel writes with a 64-row KV
    # tile, so the postprocess block must span TWO of its n_blocks -> 128 rows.
    # (seqlen_k_rounded is padded to a multiple of 2 * n_block_size whenever
    # cluster_size == 2, so the pairing never runs off the end.) It is also the
    # only workable choice mechanically: tcgen05.make_tmem_copy cannot slice a
    # 1-CTA accumulator with fewer than 128 rows -- at block_size=64 every
    # Ld32x32b Repetition is rejected.
    n_block_size_kv_post = (
        max(n_block_size, 128) if compute_capability == 10 else n_block_size
    )
    # ... and the same folding is why the dK/dV postprocess must use the 2-CTA
    # decode branch: it is the only one that reads a panel as "threads 0..63 ->
    # rows 0..63, threads 64..127 -> the same rows at hdim + hdim/2". The plain
    # branch assumes row == thread index, i.e. an unfolded 128-row writer.
    dKV_accum_folded = (
        compute_capability == 10 and use_2cta_instrs and n_block_size < 128
    )

    kv_postprocess_enabled = kv_postprocess_start is not None or kv_postprocess_end is not None

    def _kv_postprocess_range():
        if not kv_postprocess_enabled:
            return 0, seqlen_k
        start = 0 if kv_postprocess_start is None else int(kv_postprocess_start)
        end = seqlen_k if kv_postprocess_end is None else int(kv_postprocess_end)
        assert 0 <= start <= end <= seqlen_k, (
            f"invalid kv_postprocess range [{start}, {end}) for seqlen_k={seqlen_k}"
        )
        assert start % n_block_size == 0, (
            f"kv_postprocess_start must be aligned to n_block_size={n_block_size}, got {start}"
        )
        return start, end

    kv_post_start, kv_post_end = _kv_postprocess_range()

    def _slice_kv_out(t):
        if not kv_postprocess_enabled:
            return t
        return t[:, kv_post_start:kv_post_end, :, :]

    def _slice_kv_accum(t, accum_hdim):
        if not kv_postprocess_enabled:
            return t
        return t[..., kv_post_start * accum_hdim:kv_post_end * accum_hdim]

    def _to_cute(t):
        return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=t.ndim - 1
        )

    def _kv_out_cute(t, tensor):
        return _to_cute(_slice_kv_out(t)) if kv_postprocess_enabled else tensor

    def _kv_accum_cute(t, tensor, accum_hdim):
        return _to_cute(_slice_kv_accum(t, accum_hdim)) if kv_postprocess_enabled else tensor

    def _postprocess_run(
        d_accum_t,
        d_out_t,
        scale,
        hd,
        block_size,
        atom_layout,
        swapAB,
        use_2cta,
        cluster,
        cache_tag,
        raw_output_addr=None,
        raw_b=None,
        raw_s=None,
        raw_h=None,
        raw_d=None,
        raw_storage_d=None,
    ):
        compile_key_post = (
            dtype,
            hd,
            arch,
            block_size,
            num_threads,
            atom_layout,
            swapAB,
            use_2cta,
            cluster,
            cache_tag,
            raw_output_addr is not None,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, hd, arch, block_size, num_threads, atom_layout, swapAB,
                use_2cta_instrs=use_2cta, cluster_size=cluster,
            )

            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                d_accum_t, d_out_t, scale,
                raw_output_addr=raw_output_addr,
                raw_b=raw_b,
                raw_s=raw_s,
                raw_h=raw_h,
                raw_d=raw_d,
                raw_storage_d=raw_storage_d,
                stream=current_stream,
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            d_accum_t, d_out_t, scale,
            raw_output_addr=raw_output_addr,
            raw_b=raw_b,
            raw_s=raw_s,
            raw_h=raw_h,
            raw_d=raw_d,
            raw_storage_d=raw_storage_d,
            stream=current_stream,
        )

    if enable_overlap:
        raw_b = cutlass.Int32(batch_size)
        raw_s = cutlass.Int32(segment_seqlen)
        raw_h = cutlass.Int32(num_head_kv)
        raw_storage_d_k = cutlass.Int32(head_dim)
        raw_storage_d_v = cutlass.Int32(head_dim_v)

        def _raw_addr(addr, element_offset=0):
            return cutlass.Int64(int(addr) + 2 * element_offset)

    if is_bigd_bwd:
        # The big-headdim accumulators are blocked as
        #   [head_dim slice][row block][4-col group][row][4 cols]
        # so each slice is byte-identical to a head_dim=slice accumulator. The shared
        # postprocess cannot swallow head_dim 576 in one go (it stages a whole
        # tile x head_dim fp32 tile in SMEM and registers), so it runs once per slice,
        # writing a last-dim slice of the output -- legal because the postprocess
        # writes dQ/dK/dV with plain gmem copies rather than TMA.
        assert head_dim == head_dim_rounded and head_dim_v == head_dim_v_rounded, (
            "the big-headdim postprocess writes whole head_dim slices, so head_dim "
            f"must already be a multiple of 64 (got {head_dim} / {head_dim_v})"
        )
        slice_d = bigd_cfg["accum_slice_d"]
        slice_dv = bigd_cfg["accum_slice_dv"]
        # With kv_shared the kernel folded softmax_scale into dS, so dK and dQ arrive
        # already scaled (dV never carried the scale -- that asymmetry is exactly why it
        # had to move into dS for a merged dKV accumulator).
        bigd_scale = cutlass.Float32(1.0) if kv_shared else softmax_scale

        def _bigd_postprocess(
            accum, out, scale, hd, hd_slice, block, seq_rounded,
            tag, raw_addr=None, raw_storage_d=None,
        ):
            # raw_addr routes the slice into the overlap send buffer instead of `out`,
            # which then only carries the layout for the trace.
            accum_slice_elems = seq_rounded * hd_slice
            for j in range(hd // hd_slice):
                accum_slice = _to_cute(
                    accum[..., j * accum_slice_elems : (j + 1) * accum_slice_elems]
                )
                if raw_addr is None:
                    _postprocess_run(
                        accum_slice,
                        _to_cute(out[..., j * hd_slice : (j + 1) * hd_slice]),
                        scale, hd_slice, block, 1, False, False, 1,
                        tag,
                    )
                else:
                    _postprocess_run(
                        accum_slice, out,
                        scale, hd_slice, block, 1, False, False, 1,
                        tag,
                        _raw_addr(raw_addr, j * hd_slice), raw_b, raw_s, raw_h,
                        cutlass.Int32(hd_slice), raw_storage_d,
                    )

    if enable_overlap:
        def _run_overlap_dkv_postprocess(dk_send_addr, dv_send_addr):
            if is_bigd_bwd:
                _bigd_postprocess(
                    dk_accum, dk_tensor, bigd_scale, head_dim_rounded, slice_d,
                    n_block_size, seqlen_k_rounded,
                    "ovl_bigd_dk",
                    dk_send_addr, raw_storage_d_k,
                )
                # kv_shared merges dV into dK, and the comm layer then leaves the dV
                # send buffer unread, so filling it with zeros is pure overhead.
                if not kv_shared:
                    _bigd_postprocess(
                        dv_accum, dv_tensor, cutlass.Float32(1.0), head_dim_v_rounded,
                        slice_dv, n_block_size, seqlen_k_rounded,
                        "ovl_bigd_dv",
                        dv_send_addr, raw_storage_d_v,
                    )
            else:
                _postprocess_run(
                    _kv_accum_cute(dk_accum, dk_accum_tensor, head_dim_rounded),
                    _kv_out_cute(dk, dk_tensor), softmax_scale,
                    head_dim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                    dKV_accum_folded, cluster_size, "ovl_dk",
                    _raw_addr(dk_send_addr), raw_b, raw_s, raw_h,
                    cutlass.Int32(head_dim), raw_storage_d_k,
                )
                _postprocess_run(
                    _kv_accum_cute(dv_accum, dv_accum_tensor, head_dim_v_rounded),
                    _kv_out_cute(dv, dv_tensor), cutlass.Float32(1.0),
                    head_dim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                    dKV_accum_folded, cluster_size, "ovl_dv",
                    _raw_addr(dv_send_addr), raw_b, raw_s, raw_h,
                    cutlass.Int32(head_dim_v), raw_storage_d_v,
                )

        for segment_idx in range(overlap_ag_args.num_segments):
            segment_view = overlap_ag_args.kv_view(segment_idx)
            segment_dk_send_addr, segment_dv_send_addr = (
                overlap_ag_args.dkv_send_addrs(segment_idx)
            )
            segment_k_addr = cutlass.Int64(segment_view.k_addr)
            segment_v_addr = cutlass.Int64(segment_view.v_addr)
            segment_dk_addr = (
                None
                if need_kv_accum
                else cutlass.Int64(segment_dk_send_addr)
            )
            segment_dv_addr = (
                None
                if need_kv_accum
                else cutlass.Int64(segment_dv_send_addr)
            )

            if not need_kv_accum:
                overlap_runtime.wait_dkv_buffer(segment_idx, overlap_stream)
            _run_bwd_main(
                cute_flashmask_info,
                segment_k_addr,
                segment_v_addr,
                segment_dk_addr,
                segment_dv_addr,
                segment_idx=segment_idx,
            )
            if need_kv_accum:
                overlap_runtime.wait_dkv_buffer(segment_idx, overlap_stream)
                _run_overlap_dkv_postprocess(
                    segment_dk_send_addr, segment_dv_send_addr
                )
            overlap_runtime.run_backward_rs(dk, dv, segment_idx, overlap_stream)

            if segment_idx + 1 < overlap_ag_args.num_segments:
                if need_kv_accum:
                    dk_accum.zero_()
                    # kv_shared aliases dv_accum onto dk_accum, so zeroing it again
                    # would only undo the line above.
                    if not kv_shared:
                        dv_accum.zero_()
                overlap_runtime.start_backward_segment(
                    segment_idx + 1, overlap_stream
                )
                if deterministic:
                    dQ_semaphore.zero_()
                    if dK_semaphore is not None:
                        dK_semaphore.zero_()
                        dV_semaphore.zero_()
        overlap_runtime.wait_backward_rs(overlap_stream)

    if enable_overlap:
        if is_bigd_bwd:
            # dK / dV already left through the per-segment raw-address postprocess.
            _bigd_postprocess(
                dq_accum, dq, bigd_scale, head_dim_rounded, slice_d,
                m_block_size, seqlen_q_rounded,
                "bigd_dq",
            )
        else:
            _postprocess_run(
                dq_accum_tensor, dq_tensor, softmax_scale,
                head_dim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
                use_2cta_instrs, 1, "ovl_dq",
            )
    elif is_bigd_bwd:
        for accum, out, scale, hd, hd_slice, block, seq_rounded, tag in (
            (dq_accum, dq, bigd_scale, head_dim_rounded, slice_d,
             m_block_size, seqlen_q_rounded, "bigd_dq"),
            (dk_accum, dk, bigd_scale, head_dim_rounded, slice_d,
             n_block_size, seqlen_k_rounded, "bigd_dk"),
            (dv_accum, dv, cutlass.Float32(1.0), head_dim_v_rounded, slice_dv,
             n_block_size, seqlen_k_rounded, "bigd_dv"),
        ):
            # kv_shared merged dV into dK: dv is already the zero gradient and dv_accum
            # aliases dk_accum, so this postprocess would write dK into dv.
            if kv_shared and tag == "bigd_dv":
                continue
            _bigd_postprocess(
                accum, out, scale, hd, hd_slice, block, seq_rounded,
                tag,
            )
    else:
        # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
        _postprocess_run(
            dq_accum_tensor, dq_tensor, softmax_scale,
            head_dim, m_block_size, AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs, 1, "dq",
        )

        if qhead_per_kvhead > 1:
            _postprocess_run(
                _kv_accum_cute(dk_accum, dk_accum_tensor, head_dim_rounded),
                _kv_out_cute(dk, dk_tensor),
                softmax_scale,
                head_dim, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                dKV_accum_folded, cluster_size, "dk",
            )
            _postprocess_run(
                _kv_accum_cute(dv_accum, dv_accum_tensor, head_dim_v_rounded),
                _kv_out_cute(dv, dv_tensor),
                cutlass.Float32(1.0),
                head_dim_v, n_block_size_kv_post, AtomLayoutNdKV, dKV_swapAB,
                dKV_accum_folded, cluster_size, "dv",
            )

    # ---- learnable_sink gradient ----
    # dsink[h] = -sum_{b,s} exp2(sink[h]*log2e - lse_log2[b,h,s]) * delta[b,h,s]
    # where delta == dpsum and lse_log2 == lse * log2e are both already produced by
    # the preprocess kernel above. Padded rows have dpsum == 0 (lse_log2 is 0.0 there,
    # not +inf), so the product exp2(...) * dpsum == 0 and they contribute nothing. A
    # standalone cute-dsl reduction kernel (one block per head, no atomics ->
    # deterministic) consumes the existing preprocess outputs instead of launching
    # multiple Paddle ops.
    dsink = None
    if learnable_sink is not None:
        sink_dtype = learnable_sink.dtype
        dsink = be.empty(shape=[num_head], dtype=be.float32)
        sink_tensor, dsink_tensor = [
            from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (learnable_sink, dsink)
        ]
        compile_key_dsink = (compute_capability, be.to_cute_dtype(sink_dtype), num_threads)
        if compile_key_dsink not in _flash_attn_bwd.compile_cache_dsink:
            fa_bwd_dsink = FlashAttentionBackwardDsink(num_threads=num_threads)
            _flash_attn_bwd.compile_cache_dsink[compile_key_dsink] = cute.compile(
                fa_bwd_dsink,
                dpsum_tensor,
                lse_log2_tensor,
                sink_tensor,
                dsink_tensor,
                current_stream,
            )
        _flash_attn_bwd.compile_cache_dsink[compile_key_dsink](
            dpsum_tensor,
            lse_log2_tensor,
            sink_tensor,
            dsink_tensor,
            current_stream,
        )
        dsink = be.astype(dsink, sink_dtype)

    return dq, dk, dv, dsink


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}
_flash_attn_bwd.compile_cache_dsink = {}
