# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Preallocated forward scratch and the per-step plan (#26).

`FwdWorkspace` owns every buffer `_flash_attn_fwd` would otherwise allocate per call, sized
once from static upper bounds. A serving loop builds it once, calls `plan` once per batch
step (the result is shared by every layer of that step), and passes `workspace=` to each
forward call. The run path then allocates no device memory and reads nothing back to the
host, so it can be captured in a CUDA graph with fixed shapes.

Framework-agnostic: allocation and stream access go through a `flashmask.backend` adapter.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from ...backend import Tensor, get_backend
from ..flashmask_utils import _compute_nblock_seqlen
from ._common import sm100_split_kv_supported

__all__ = ["FwdWorkspace", "tensormap_workspace_ctas"]

# Lower bounds of the tile sizes any SM100 forward config uses, so buffers indexed per
# tile are large enough for every config: n_block_size >= 32 (the 1-CTA big-d config),
# query rows per work tile >= 64.
_MIN_TILE_N = 32
_MIN_TILE_M_ROWS = 64
_PLAN_THREADS = 128


def tensormap_workspace_ctas(batch_size, seqlen_q, q_mult, m_block_size, n_heads_sched, num_splits, num_sms):
    """Rows of the per-CTA V descriptor workspace (#23): an upper bound on the grid size."""
    n_q_rows = batch_size * seqlen_q * q_mult
    m_blocks_bound = n_q_rows // m_block_size + 2 * batch_size + 1
    return max(m_blocks_bound * n_heads_sched * num_splits, 2 * num_sms) + 8


def _cute_i32(t):
    return from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)


@cute.kernel
def _plan_num_splits_kernel(
    mSeqUsedK: Optional[cute.Tensor],
    mSplitsIn: Optional[cute.Tensor],
    mSplitsOut: cute.Tensor,
    batch: Int32,
    seqlen_k: Int32,
    tile_n: Int32,
    blocks_per_split: Int32,
    max_splits: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    b = bidx * _PLAN_THREADS + tidx
    if b < batch:
        n = Int32(1)
        if const_expr(mSplitsIn is not None):
            n = mSplitsIn[b]
        else:
            seqlen = seqlen_k
            if const_expr(mSeqUsedK is not None):
                seqlen = mSeqUsedK[b]
            n_blocks = (seqlen + tile_n - 1) // tile_n
            n = (n_blocks + blocks_per_split - 1) // blocks_per_split
        mSplitsOut[b] = cutlass.max(cutlass.min(n, max_splits), Int32(1))


@cute.jit
def _plan_num_splits_cute(
    mSeqUsedK: Optional[cute.Tensor],
    mSplitsIn: Optional[cute.Tensor],
    mSplitsOut: cute.Tensor,
    batch: Int32,
    seqlen_k: Int32,
    tile_n: Int32,
    blocks_per_split: Int32,
    max_splits: Int32,
    stream: cuda.CUstream,
):
    _plan_num_splits_kernel(
        mSeqUsedK, mSplitsIn, mSplitsOut, batch, seqlen_k, tile_n, blocks_per_split, max_splits
    ).launch(
        grid=[(batch + _PLAN_THREADS - 1) // _PLAN_THREADS, 1, 1],
        block=[_PLAN_THREADS, 1, 1],
        stream=stream,
    )


def _plan_num_splits(be, seqused_k, splits_in, splits_out, batch, seqlen_k, tile_n, blocks_per_split, max_splits):
    args = (
        _cute_i32(seqused_k) if seqused_k is not None else None,
        _cute_i32(splits_in) if splits_in is not None else None,
        _cute_i32(splits_out),
        Int32(batch),
        Int32(seqlen_k),
        Int32(tile_n),
        Int32(blocks_per_split),
        Int32(max_splits),
    )
    stream = cuda.CUstream(be.cuda_stream())
    key = (seqused_k is not None, splits_in is not None)
    if key not in _plan_num_splits.compile_cache:
        _plan_num_splits.compile_cache[key] = cute.compile(_plan_num_splits_cute, *args, stream)
    _plan_num_splits.compile_cache[key](*args, stream)


_plan_num_splits.compile_cache = {}


def _view(buf, shape):
    """The first prod(shape) elements of a flat buffer, as a contiguous view."""
    numel = 1
    for s in shape:
        numel *= s
    assert numel <= buf.shape[0], f"workspace too small: {list(shape)} needs {numel} > {buf.shape[0]} elements"
    return buf[:numel].view(list(shape))


class FwdWorkspace:
    """Every buffer the SM100 forward allocates per call, allocated once.

    Args:
        ref: a framework tensor on the target device with the attention dtype (e.g. q).
            Only its framework, device and dtype are read.
        max_batch, max_seqlen_q, max_seqlen_k: upper bounds of B, seqlen_q and the KV
            extent seqlen_k (max_pages * page_size with paged KV).
        num_heads, num_heads_kv, head_dim, head_dim_v: fixed per model; a call must match.
        max_splits: the split-KV grid size. 1 disables split-KV. Above 1 every call runs
            max_splits splits and batch b uses the first num_splits_dynamic[b] of them
            (filled by `plan`, initialized to max_splits); the rest exit early.
        num_vecs: startend_row_indices.shape[3] when calls pass a column mask, else None.
        mask_heads: startend_row_indices.shape[1]; defaults to num_heads, the upper bound.

    Not held: out and lse. They are the call's results, so pass out= (and lse=) to make a
    call allocation-free. One workspace serves one stream at a time: calls overlapping on
    two streams would share the partials and the scheduler counter.
    """

    def __init__(
        self,
        ref: Tensor,
        *,
        max_batch: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_heads: int,
        num_heads_kv: int,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        max_splits: int = 1,
        num_vecs: Optional[int] = None,
        mask_heads: Optional[int] = None,
    ):
        be = get_backend(ref)
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        assert num_heads % num_heads_kv == 0, "num_heads must be divisible by num_heads_kv"
        assert 1 <= max_splits <= 256, "max_splits must be in [1, 256] (the combine kernel's limit)"
        if max_splits > 1 and not sm100_split_kv_supported(head_dim, head_dim_v, ref.element_size()):
            raise NotImplementedError(
                f"SplitKV is not supported for head_dim={head_dim}, head_dim_v={head_dim_v} on SM100"
            )
        assert num_vecs in (None, 1, 2, 4), "num_vecs must be 1, 2 or 4"
        self.be = be
        self.dtype = ref.dtype
        self.max_batch = max_batch
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.max_splits = max_splits
        self.num_vecs = num_vecs
        self.mask_heads = num_heads if mask_heads is None else mask_heads
        # The KV tile of every SM100 config that supports split-KV; plan counts blocks in it.
        self.n_block_size = 128

        i32 = be.int32
        self.num_splits_dynamic = be.full([max_batch], max_splits, dtype=i32)
        if max_splits > 1:
            self.out_partial = be.empty(
                [max_splits * max_batch * max_seqlen_q * num_heads * head_dim_v], dtype=be.float32
            )
            self.lse_partial = be.empty([max_splits * max_batch * num_heads * max_seqlen_q], dtype=be.float32)
        else:
            self.out_partial = self.lse_partial = None
        # Zeroed once; DynamicPersistentTileScheduler resets it at the end of each launch.
        self.sched_counter = be.zeros([2], dtype=i32)
        qhpk = num_heads // num_heads_kv
        n_ctas = tensormap_workspace_ctas(
            max_batch, max_seqlen_q, qhpk, _MIN_TILE_M_ROWS, num_heads, max_splits, be.multi_processor_count()
        )
        self.tensormap = be.empty([n_ctas, 16], dtype=be.int64)
        self.nblock_buffers = []
        self.column_scratch = None
        self.valid_block_count = None
        if num_vecs is not None:
            mh = self.mask_heads
            nblock_numel = max_batch * mh * _compute_nblock_seqlen(max_seqlen_k, _MIN_TILE_N)
            self.nblock_buffers = [be.empty([nblock_numel], dtype=i32) for _ in range(2 * num_vecs)]
            if num_vecs > 1:
                # startend_row_indices[..., i] is strided; the scan reads a contiguous copy.
                self.column_scratch = be.empty([max_batch * mh * max_seqlen_k], dtype=i32)
            m_rows = max_seqlen_q * qhpk
            self.valid_block_count = be.empty(
                [max_batch * mh * (-(-m_rows // _MIN_TILE_M_ROWS))], dtype=i32
            )

    def plan(
        self,
        seqused_k: Optional[Tensor] = None,
        num_splits_dynamic: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        seqlen_k: Optional[int] = None,
        blocks_per_split: Optional[int] = None,
    ) -> Optional[Tensor]:
        """Fill num_splits_dynamic for one batch step, on the device. Returns its [B] view.

        Run once per step after seqused_k is updated; every layer's forward then reads it.
        No allocation and no device-to-host copy.

        - num_splits_dynamic given (int32 [B] on the device, e.g. counts the host chose):
          copied into the workspace, clamped to [1, max_splits].
        - else each batch gets ceil(n_blocks[b] / blocks_per_split) splits, clamped to
          [1, max_splits], where n_blocks[b] counts 128-key blocks of seqused_k[b] (or of
          seqlen_k without seqused_k). blocks_per_split defaults to
          ceil(n_blocks(max_seqlen_k) / max_splits): a request as long as max_seqlen_k
          uses every split and shorter ones fewer, so each split walks about the same
          number of blocks.

        Returns None (and does nothing) when max_splits == 1.
        """
        if self.max_splits == 1:
            return None
        src = num_splits_dynamic if num_splits_dynamic is not None else seqused_k
        if batch_size is None:
            batch_size = src.shape[0] if src is not None else self.max_batch
        assert batch_size <= self.max_batch, f"batch {batch_size} > max_batch {self.max_batch}"
        for name, t in (("seqused_k", seqused_k), ("num_splits_dynamic", num_splits_dynamic)):
            if t is not None:
                assert t.dtype == self.be.int32 and t.ndim == 1 and self.be.is_gpu(t), f"{name} must be int32 1-D on GPU"
                assert t.shape[0] >= batch_size, f"{name} needs {batch_size} entries"
        seqlen_k = self.max_seqlen_k if seqlen_k is None else seqlen_k
        if blocks_per_split is None:
            n_blocks_max = -(-self.max_seqlen_k // self.n_block_size)
            blocks_per_split = max(1, -(-n_blocks_max // self.max_splits))
        assert blocks_per_split >= 1
        out = self.num_splits_dynamic[:batch_size]
        _plan_num_splits(
            self.be, seqused_k, num_splits_dynamic, out, batch_size, seqlen_k,
            self.n_block_size, blocks_per_split, self.max_splits,
        )
        return out

    # -- used by _flash_attn_fwd ------------------------------------------------

    def check_call(self, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_kv, head_dim, head_dim_v, dtype):
        fixed = dict(num_heads=num_heads, num_heads_kv=num_heads_kv, head_dim=head_dim, head_dim_v=head_dim_v, dtype=dtype)
        for name, value in fixed.items():
            assert value == getattr(self, name), f"workspace {name}={getattr(self, name)}, call has {value}"
        for name, value, bound in (
            ("batch", batch_size, self.max_batch),
            ("seqlen_q", seqlen_q, self.max_seqlen_q),
            ("seqlen_k", seqlen_k, self.max_seqlen_k),
        ):
            assert value <= bound, f"{name}={value} exceeds the workspace bound {bound}"

    def check_mask(self, mask):
        assert self.num_vecs == mask.shape[3], (
            f"workspace num_vecs={self.num_vecs}, startend_row_indices has {mask.shape[3]}"
        )
        assert mask.shape[1] <= self.mask_heads, f"mask heads {mask.shape[1]} > mask_heads {self.mask_heads}"
        assert mask.shape[0] <= self.max_batch and mask.shape[2] <= self.max_seqlen_k, (
            f"startend_row_indices {list(mask.shape)} exceeds max_batch / max_seqlen_k"
        )

    def flashmask_kwargs(self):
        """Arguments for prepare_block_maxmin that keep it inside the workspace."""
        return dict(
            alloc=self.nblock_alloc(),
            column_scratch=self.column_view if self.column_scratch is not None else None,
        )

    def partials(self, batch_size, seqlen_q):
        """O / LSE partials with the split axis folded into batch, split outermost."""
        s = self.max_splits
        return (
            _view(self.out_partial, [s * batch_size, seqlen_q, self.num_heads, self.head_dim_v]),
            _view(self.lse_partial, [s * batch_size, self.num_heads, seqlen_q]),
        )

    def nblock_alloc(self):
        """An allocator for prepare_block_maxmin that hands out the nblock buffers in turn."""
        buffers = iter(self.nblock_buffers)
        return lambda shape: _view(next(buffers), shape)

    def valid_block_count_view(self, shape):
        return _view(self.valid_block_count, shape)

    def column_view(self, shape):
        return _view(self.column_scratch, shape)

    def tensormap_for(self, n_ctas):
        assert n_ctas <= self.tensormap.shape[0], (
            f"tensormap workspace has {self.tensormap.shape[0]} rows, the call needs {n_ctas}"
        )
        return self.tensormap
