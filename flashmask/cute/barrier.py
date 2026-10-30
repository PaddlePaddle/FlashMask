# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    """``ld.global.acquire.gpu``: synchronizes with a gpu-or-wider-scope release of the
    word by any thread on this GPU, including other kernel grids (PTX ISA 8.5). Data the
    releaser acquired from peers or the NIC reaches the caller by causality transitivity
    (8.9.5), so the flag writer, not the data writer, must be on this GPU. The ordering
    is generic-proxy only; async-proxy (TMA) readers need ``fence.proxy.async`` (8.6).
    """
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    state = llvm.inline_asm(
        T.i32(),
        [lock_ptr_i64],
        "ld.global.acquire.gpu.b32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Int32(state)


@cute.jit
def wait_flag_eq(flag_ptr: cute.Pointer, elem_idx: int | Int32, target: Int32) -> None:
    """Busy-spin with acquire loads until ``flag_ptr[elem_idx] == target``."""
    ptr = flag_ptr + elem_idx
    value = ld_acquire(ptr)
    while value != target:
        value = ld_acquire(ptr)


@cute.jit
def wait_write_ptr_ge(wptr: cute.Pointer, elem_idx: int | Int32, target: Int32) -> None:
    """Busy-spin until wptr[elem_idx] >= target. Caller elects a single thread.

    write_ptr is a monotonically advanced (atomicMax) row index on the comm side; once
    it reaches `target` the corresponding remote KV rows are all-gathered and visible to
    the calling thread's generic-proxy reads. The compare is on non-negative row counts
    (the INT_MAX whole-AG sentinel is also positive), so a plain b32 acquire load suffices.
    """
    flag_ptr = wptr + elem_idx
    read_val = ld_acquire(flag_ptr)
    while read_val < target:
        read_val = ld_acquire(flag_ptr)


@dsl_user_op
def red_relaxed(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def red_release(
    lock_ptr: cute.Pointer, val: cutlass.Constexpr[Int32], *, loc=None, ip=None
) -> None:
    lock_ptr_i64 = lock_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [lock_ptr_i64, Int32(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def wait_eq(lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: Int32) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        read_val = Int32(0)
        while read_val != val:
            read_val = ld_acquire(flag_ptr)


@cute.jit
def arrive_inc(
    lock_ptr: cute.Pointer, thread_idx: int | Int32, flag_offset: int, val: cutlass.Constexpr[Int32]
) -> None:
    flag_ptr = lock_ptr + flag_offset
    if thread_idx == 0:
        red_release(flag_ptr, val)
        # red_relaxed(flag_ptr, val)
