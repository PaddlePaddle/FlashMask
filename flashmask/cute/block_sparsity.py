# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""
Block-sparse KV-block list data structures and their validation/conversion.

This module only holds the *plumbing* for the block-sparse kernel path: the
`BlockSparseTensors` / `BlockSparseTensorsPaddle` tuples that describe which KV
blocks each query block must visit (the mask list -- partially masked blocks --
and the optional full list -- fully visible blocks), plus the checks and the
framework->cute conversion. It does NOT compute those block lists.

The block lists are produced by the caller. Today the only producer is flashmask
(`flashmask.cute.flashmask_utils.build_flashmask_block_lists`, wired in
`flashmask.cute.ops.fwd`); precomputed lists may also be passed in from a
frontend. The former generic mask->block reference computation has been removed.

This module is imported by the kernels, so it must not import a framework at
module scope -- everything framework-specific goes through a
`flashmask.backend` adapter instead.
"""

from typing import Tuple, Optional, NamedTuple
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..backend import Tensor, get_backend


class BlockSparseTensors(NamedTuple):
    mask_block_cnt: cute.Tensor
    mask_block_idx: cute.Tensor
    full_block_cnt: Optional[cute.Tensor]
    full_block_idx: Optional[cute.Tensor]
    # Optional block-sparse scheduling offsets; None for the flashmask fixed-length reuse.
    cu_total_m_blocks: Optional[cute.Tensor] = None
    cu_block_idx_offsets: Optional[cute.Tensor] = None

    def __new_from_mlir_values__(self, values):
        values = list(values)
        # Pad missing trailing optional fields (full_* and cu_* are None for the
        # fixed-length / flashmask cases, so they don't appear as MLIR values).
        if len(values) == 2:
            values = values + [None, None, None, None]
        elif len(values) == 4:
            values = values + [None, None]
        return BlockSparseTensors(*values)


class BlockSparseTensorsPaddle(NamedTuple):
    mask_block_cnt: Tensor
    mask_block_idx: Tensor
    full_block_cnt: Optional[Tensor] = None
    full_block_idx: Optional[Tensor] = None
    cu_total_m_blocks: Optional[Tensor] = None
    cu_block_idx_offsets: Optional[Tensor] = None


def _expand_sparsity_tensor(
    tensor: Tensor,
    expected_shape: Tuple[int, ...],
    tensor_name: str,
) -> Tensor:
    """Check if we need to expand the tensor to expected shape, and do so if possible."""
    needs_expand = tensor.shape != expected_shape
    if not needs_expand:
        return tensor
    can_expand = all(map(lambda cur, tgt: cur == tgt or cur == 1, tensor.shape, expected_shape))
    if not can_expand:
        raise ValueError(
            f"{tensor_name} with shape {tensor.shape} cannot be expanded to expected shape {expected_shape}."
        )
    return tensor.expand(*expected_shape).contiguous()


def _check_and_expand_block(
    name: str,
    cnt: Optional[Tensor],
    idx: Optional[Tensor],
    expected_count_shape: Tuple[int, int, int],
    expected_index_shape: Tuple[int, int, int, int],
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    if (cnt is None) != (idx is None):
        raise ValueError(
            f"{name}_block_cnt and {name}_block_idx must both be provided or both be None"
        )
    if cnt is None or idx is None:
        return None, None
    be = get_backend(cnt, idx)
    if cnt.dtype != be.int32 or idx.dtype != be.int32:
        raise ValueError(f"{name}_block tensors must have dtype int32")
    if be.device_of(cnt) != be.device_of(idx):
        raise ValueError(f"{name}_block_cnt and {name}_block_idx must be on the same device")
    if not cnt.is_cuda or not idx.is_cuda:
        raise ValueError(f"{name}_block tensors must live on CUDA")
    expanded_cnt = _expand_sparsity_tensor(cnt, expected_count_shape, f"{name}_block_cnt")
    expanded_idx = _expand_sparsity_tensor(idx, expected_index_shape, f"{name}_block_idx")
    return expanded_cnt, expanded_idx


def normalize_block_sparse_tensors(
    tensors: BlockSparseTensorsPaddle,
    *,
    expected_count_shape: Tuple[int, int, int],
    expected_index_shape: Tuple[int, int, int, int],
) -> BlockSparseTensorsPaddle:
    if tensors.mask_block_cnt is None or tensors.mask_block_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    mask_cnt, mask_idx = _check_and_expand_block(
        "mask",
        tensors.mask_block_cnt,
        tensors.mask_block_idx,
        expected_count_shape,
        expected_index_shape,
    )
    if mask_cnt is None or mask_idx is None:
        raise ValueError("mask_block_cnt and mask_block_idx must be provided for block sparsity.")

    full_cnt, full_idx = _check_and_expand_block(
        "full",
        tensors.full_block_cnt,
        tensors.full_block_idx,
        expected_count_shape,
        expected_index_shape,
    )
    if full_cnt is not None:
        be = get_backend(mask_cnt, full_cnt)
        if be.device_of(mask_cnt) != be.device_of(full_cnt):
            raise ValueError("All block sparse tensors must be on the same device")

    return BlockSparseTensorsPaddle(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
    )


def is_block_sparsity_enabled(tensors: BlockSparseTensorsPaddle) -> bool:
    return any(t is not None for t in (tensors.full_block_cnt, tensors.mask_block_cnt))


def to_cute_block_sparse_tensors(tensors: BlockSparseTensorsPaddle) -> Optional[BlockSparseTensors]:
    # `tensors` is produced by flashmask (build_flashmask_block_lists) or passed
    # in precomputed; this only wraps the framework tensors as cute tensors.
    if not is_block_sparsity_enabled(tensors):
        return None

    mask_block_cnt_tensor = from_dlpack(
        tensors.mask_block_cnt.detach(), assumed_align=4
    ).mark_layout_dynamic(leading_dim=2)
    mask_block_idx_tensor = from_dlpack(
        tensors.mask_block_idx.detach(), assumed_align=4
    ).mark_layout_dynamic(leading_dim=3)
    full_block_cnt_tensor = (
        from_dlpack(tensors.full_block_cnt.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=2
        )
        if tensors.full_block_cnt is not None
        else None
    )
    full_block_idx_tensor = (
        from_dlpack(tensors.full_block_idx.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=3
        )
        if tensors.full_block_idx is not None
        else None
    )

    cu_total_m_blocks_tensor = (
        from_dlpack(tensors.cu_total_m_blocks.detach(), assumed_align=4).mark_layout_dynamic()
        if tensors.cu_total_m_blocks is not None
        else None
    )
    cu_block_idx_offsets_tensor = (
        from_dlpack(tensors.cu_block_idx_offsets.detach(), assumed_align=4).mark_layout_dynamic()
        if tensors.cu_block_idx_offsets is not None
        else None
    )

    return BlockSparseTensors(
        mask_block_cnt_tensor,
        mask_block_idx_tensor,
        full_block_cnt_tensor,
        full_block_idx_tensor,
        cu_total_m_blocks_tensor,
        cu_block_idx_offsets_tensor,
    )
