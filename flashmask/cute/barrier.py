# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

import functools
import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.ffi import ffi
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm


# Polls between %globaltimer reads; same interval as kSpinCheckInterval of the comm
# runtime's SpinWatchdog (flashmask/overlap/csrc/distributed/nccl_gin_backend.cuh).
SPIN_CHECK_INTERVAL = 1024
_NS_PER_SECOND = 1000 * 1000 * 1000
_INT64_MAX = (1 << 63) - 1
_UINT64_MAX = (1 << 64) - 1


@functools.cache
def spin_timeout_ns_from_env() -> int:
    """Host side: the overlap gates' spin budget in ns, read once per process.

    Same contract as the comm runtime's ``spin_timeout_ns_from_env``
    (``nccl_gin_backend.cu``): ``FLASHMASK_SPIN_TIMEOUT_S`` in whole seconds, default
    1800 (Paddle's collective timeout), 0 disables the time-based trap. The kernel
    compares in Int64, so budgets past INT64_MAX ns (292 years) clamp to it, which
    never trips, like the C++ uint64 budget.
    """
    text = os.environ.get("FLASHMASK_SPIN_TIMEOUT_S", "").lstrip()
    if text == "":
        return 1800 * _NS_PER_SECOND
    digits = text[1:] if text[0] == "+" else text  # strtoull accepts one leading '+'
    if not (digits.isascii() and digits.isdigit()) or int(digits) > _UINT64_MAX // _NS_PER_SECOND:
        raise ValueError(
            "FLASHMASK_SPIN_TIMEOUT_S must be a non-negative integer number of seconds, "
            f"got: {text}"
        )
    return min(int(digits) * _NS_PER_SECOND, _INT64_MAX)


@dsl_user_op
def globaltimer_ns(*, loc=None, ip=None) -> Int64:
    """``%globaltimer``: GPU-wide nanosecond timer, the clock the comm watchdog uses."""
    return Int64(nvvm.read_ptx_sreg_globaltimer(T.i64(), loc=loc, ip=ip))


@dsl_user_op
def sreg_u32_cold(name: str, *, loc=None, ip=None) -> Int32:
    """Read a 32-bit special register (``%ctaid.x``, ``%tid.x``, ...) where it is used.

    A volatile asm read cannot be CSE'd with the kernel's own early reads, so a
    diagnostic on a cold path does not keep those values live across the hot loops
    (``cute.arch.block_idx`` there raised ptxas spills in the fwd compute warps).
    """
    value = llvm.inline_asm(
        T.i32(),
        [],
        f"mov.u32 $0, {name};",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(value)


def _c_string(symbol: str, text: str, *, loc=None) -> ir.Value:
    """Generic pointer to a NUL-terminated internal constant in the enclosing gpu.module."""
    module = ir.InsertionPoint.current.block.owner
    while module.name != "gpu.module":
        module = module.parent
    data = text + "\0"
    if symbol not in ir.SymbolTable(module.operation):
        with ir.InsertionPoint.at_block_begin(module.regions[0].blocks[0]):
            llvm.GlobalOp(
                ir.Type.parse(f"!llvm.array<{len(data.encode())} x i8>"),
                symbol,
                ir.Attribute.parse("#llvm.linkage<internal>"),
                constant=True,
                value=ir.StringAttr.get(data),
                loc=loc,
            )
    return llvm.AddressOfOp(llvm.PointerType.get(), symbol, loc=loc).result


@dsl_user_op
def assert_fail(message: str, function: str, *, loc=None, ip=None) -> None:
    """Device-side assert failure: the CUDA runtime prints ``message`` and traps the grid.

    Calls the ``__assertfail`` system call that ``assert()`` lowers to, so the kernel
    aborts with cudaErrorAssert like a ``trap``. A ``trap`` instruction written in the
    kernel itself instead changes ptxas register allocation kernel-wide (16 spill
    instructions inside the fwd d128 compute loop vs 3), while the call keeps it local.
    """
    ptr_ty = llvm.PointerType.get()
    assertfail = ffi(
        name="__assertfail", params_types=[ptr_ty, ptr_ty, Int32, ptr_ty, Int64], inline=False
    )
    tag = function.replace(" ", "_")
    assertfail(
        _c_string(f"__fm_assert_msg_{tag}", message, loc=loc),
        _c_string("__fm_assert_file", "flashmask/cute/barrier.py", loc=loc),
        Int32(0),
        _c_string(f"__fm_assert_fn_{tag}", function, loc=loc),
        Int64(1),
    )


@cute.jit
def _spin_watchdog_check(
    start_ns: Int64,
    timeout_ns: Int64,
    ptr: cute.Pointer,
    expected: Int32,
    observed: Int32,
    site: cutlass.Constexpr[str],
) -> Int64:
    """One watchdog check, run every SPIN_CHECK_INTERVAL polls; returns the spin's start.

    Mirrors ``spin_watchdog_check`` of the comm runtime: ``timeout_ns == 0`` disables it,
    the start time is taken lazily on the first check (a wait that ends within one
    interval never reads the timer), and a spin longer than the budget prints the stuck
    word and traps. Device printf has no rank, so the line carries block/thread/address.
    The elapsed time is left out: its 64-bit divide alone moved ptxas spills into the fwd
    BHSD compute loop, and the budget is FLASHMASK_SPIN_TIMEOUT_S anyway.
    """
    new_start_ns = start_ns
    if timeout_ns != 0:
        now_ns = globaltimer_ns()
        if start_ns == 0:
            new_start_ns = now_ns
        elif now_ns - start_ns > timeout_ns:
            cute.printf(
                "[FlashMask spin] " + site + ": FLASHMASK_SPIN_TIMEOUT_S exceeded, "
                "block (%d,%d,%d) thread %d addr 0x%llx expected %d actual %d",
                sreg_u32_cold("%ctaid.x"),
                sreg_u32_cold("%ctaid.y"),
                sreg_u32_cold("%ctaid.z"),
                sreg_u32_cold("%tid.x"),
                ptr.toint(),
                expected,
                observed,
            )
            assert_fail("FlashMask overlap " + site + " spin timeout", site)
    return new_start_ns


@dsl_user_op
def ld_acquire(lock_ptr: cute.Pointer, *, loc=None, ip=None) -> cutlass.Int32:
    """``ld.global.acquire.gpu``: pairs with a gpu-scope release by any grid on this GPU;
    data the releaser acquired from peers or the NIC follows by causality, so the flag
    writer must be on this GPU. Generic proxy only: TMA readers also need ``fence.proxy.async``.
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
def wait_flag_eq(
    flag_ptr: cute.Pointer, elem_idx: int | Int32, target: Int32, timeout_ns: Int64
) -> None:
    """Busy-spin with acquire loads until ``flag_ptr[elem_idx] == target``.

    Traps after ``timeout_ns`` of spinning (0: never); see ``_spin_watchdog_check``.
    """
    ptr = flag_ptr + elem_idx
    value = ld_acquire(ptr)
    polls = Int32(0)
    start_ns = Int64(0)
    while value != target:
        value = ld_acquire(ptr)
        polls += 1
        if polls & (SPIN_CHECK_INTERVAL - 1) == 0:
            start_ns = _spin_watchdog_check(
                start_ns, timeout_ns, ptr, target, value, "work_done gate"
            )


@cute.jit
def wait_write_ptr_ge(
    wptr: cute.Pointer, elem_idx: int | Int32, target: Int32, timeout_ns: Int64
) -> None:
    """Busy-spin until wptr[elem_idx] >= target. Caller elects a single thread.

    write_ptr is a monotonically advanced (atomicMax) row index on the comm side; once
    it reaches `target` the corresponding remote KV rows are all-gathered and visible to
    the calling thread's generic-proxy reads. The compare is on non-negative row counts
    (the INT_MAX whole-AG sentinel is also positive), so a plain b32 acquire load suffices.
    Traps after ``timeout_ns`` of spinning (0: never); see ``_spin_watchdog_check``.
    """
    flag_ptr = wptr + elem_idx
    read_val = ld_acquire(flag_ptr)
    polls = Int32(0)
    start_ns = Int64(0)
    while read_val < target:
        read_val = ld_acquire(flag_ptr)
        polls += 1
        if polls & (SPIN_CHECK_INTERVAL - 1) == 0:
            start_ns = _spin_watchdog_check(
                start_ns, timeout_ns, flag_ptr, target, read_val, "write_ptr gate"
            )


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
