# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

# Ported from upstream FA4 flash_attn/cute/tensormap_utils.py at Dao-AILab/flash-attention
# c055bdc6444a1a1984082a64b7fd62d6cb42a93d. Fork-local difference:
# nvidia-cutlass-dsl 4.5.2 has no cute.arch.inline_ptx, so trap and prefetch.tensormap
# are emitted with llvm.inline_asm (_inline_ptx below).

"""Per-CTA TMA descriptor copies that bound ragged loads by a sequence end.

A load warp copies a TMA atom's descriptor into its CTA's 128-byte workspace slot, rewrites the
rows extent to the sequence end, and acquires the rewrite before loading through it: rows at or
past the end then arrive zero-filled from a bounded footprint (#3071).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.utils import TensorMapManager, TensorMapUpdateMode


@dsl_user_op
def _trap(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [], "trap;", "", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _prefetch_tensormap(addr, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None, [Int64(addr).ir_value(loc=loc, ip=ip)], "prefetch.tensormap [$0];", "l",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )


@cute.jit
def cta_tensormap_slot(workspace: cute.Tensor):
    """(manager, gmem_ptr, desc_ptr) of this CTA's slot in a (num_ctas, 16) Int64 workspace.

    desc_ptr is the generic pointer TMA copies take as tma_desc_ptr. Traps when the workspace
    has fewer slots than the grid has CTAs.
    """
    manager = TensorMapManager(TensorMapUpdateMode.GMEM, 128)
    bidx, bidy, bidz = cute.arch.block_idx()
    gdx, gdy, _ = cute.arch.grid_dim()
    cta_linear_idx = bidx + bidy * gdx + bidz * gdx * gdy
    if cta_linear_idx >= cute.size(workspace, mode=[0]):
        # An undersized workspace would silently corrupt memory: fail loudly instead.
        _trap()
    gmem_ptr = manager.get_tensormap_ptr(workspace[(cta_linear_idx, None)].iterator)
    desc_ptr = manager.get_tensormap_ptr(gmem_ptr, cute.AddressSpace.generic)
    return manager, gmem_ptr, desc_ptr


def bound_rows(mX: cute.Tensor, rows_end) -> cute.Tensor:
    """mX with its rows mode (mode 1) cut at rows_end; descriptor extents must stay >= 1."""
    shape = list(mX.shape)
    shape[1] = cutlass.max(rows_end, 1)
    return cute.make_tensor(mX.iterator, cute.make_layout(tuple(shape), stride=mX.stride))


@cute.jit
def acquire_tensormap(manager: TensorMapManager, gmem_ptr: cute.Pointer):
    """Make a rewritten descriptor visible to this thread's TMA loads and prefetch it."""
    manager.fence_tensormap_update(gmem_ptr)
    _prefetch_tensormap(gmem_ptr.toint())
