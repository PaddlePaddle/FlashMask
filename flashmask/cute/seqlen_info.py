# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from . import copy_utils


"""
This consolidates all the info related to sequence length. This is so that we can do all
the gmem reads once at the beginning of each tile, rather than having to repeat these reads
to compute various things like n_block_min, n_block_max, etc.
"""


@dataclass(frozen=True)
class SeqlenInfo:
    offset: Int32
    offset_padded: Int32
    seqlen: Int32

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_static: Int32,
        tile: cutlass.Constexpr[int] = 128,
    ):
        # Fixed-length only: offsets are always 0 and seqlen is the static shape.
        return SeqlenInfo(0, 0, seqlen_static)

    def offset_batch(
        self,
        mT: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        """Offset a tensor by batch index. batch dim is at position `dim`, seqlen is at dim=0."""
        idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mT) - 1 - dim)
        return mT[idx]


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: Int32
    offset_k: Int32
    padded_offset_q: Int32
    padded_offset_k: Int32
    seqlen_q: Int32
    seqlen_k: Int32
    m_block_offset: Int32
    block_idx_offset: Int32
    num_n_blocks: Int32

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        mCuBlockIdxOffsets: Optional[cute.Tensor] = None,
        tile_m: cutlass.Constexpr[Int32] = 128,
        tile_n: cutlass.Constexpr[Int32] = 128,
    ):
        # Fixed-length only: q/k offsets are always 0 and seqlens are the static
        # shapes. mCuTotalMBlocks / mCuBlockIdxOffsets belong to the block-sparse
        # scheduling contract (None for the flashmask fixed-length reuse).
        offset_q = 0
        offset_k = 0
        padded_offset_q = 0
        padded_offset_k = 0
        seqlen_q = seqlen_q_static
        seqlen_k = seqlen_k_static
        m_block_offset = 0 if const_expr(mCuTotalMBlocks is None) else mCuTotalMBlocks[batch_idx]
        num_n_blocks = (seqlen_k + tile_n - 1) // tile_n
        block_idx_offset = (
            mCuBlockIdxOffsets[batch_idx]
            if const_expr(mCuBlockIdxOffsets is not None)
            else m_block_offset * num_n_blocks
        )
        return SeqlenInfoQK(
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            seqlen_q,
            seqlen_k,
            m_block_offset,
            block_idx_offset,
            num_n_blocks,
        )

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        """Seqlen must be the first dimension of mQ"""
        if const_expr(not ragged):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
            return mQ[idx]
        else:
            offset_q = 0
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
            mQ = mQ[idx]
            if const_expr(cute.rank(mQ.shape[0]) == 1):
                return copy_utils.offset_ragged_tensor(
                    mQ, offset_q, self.seqlen_q, ragged_dim=0, ptr_shift=True
                )
            else:  # PackGQA
                assert cute.rank(mQ.shape[0]) == 2
                # Unpack before calling offset_ragged_tensor, then pack
                idx = ((None, None),) + (None,) * (cute.rank(mQ) - 1)
                mQ = mQ[idx]
                mQ = copy_utils.offset_ragged_tensor(
                    mQ, offset_q, self.seqlen_q, ragged_dim=1, ptr_shift=True
                )
                return cute.group_modes(mQ, 0, 2)

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        """Seqlen must be the first dimension of mK"""
        if const_expr(not ragged):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
            return mK[idx]
        else:
            offset_k = 0
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
            mK = mK[idx]
            return copy_utils.offset_ragged_tensor(
                mK, offset_k, self.seqlen_k, ragged_dim=0, ptr_shift=True
            )
