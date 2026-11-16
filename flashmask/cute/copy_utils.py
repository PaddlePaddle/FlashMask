# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

import math
from typing import Optional, Type, Tuple, Callable, Sequence


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cutlass.pipeline
from cutlass.base_dsl.arch import Arch
from cutlass.cute.nvgpu import warp


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_fragment_like(src, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric], num_copy_elems: int, is_async: bool = False, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def make_tmem_copy(
    tmem_copy_atom: cute.CopyAtom, num_wg: int = 1, *, loc=None, ip=None
) -> cute.CopyAtom:
    num_dp, num_bits, num_rep, _ = sm100_utils.get_tmem_copy_properties(tmem_copy_atom)
    assert num_dp == 32
    assert num_bits == 32
    tiler_mn = (cute.make_layout((128 * num_rep * num_wg // 32, 32), stride=(32, 1)),)
    layout_tv = cute.make_layout(
        ((32, 4, num_wg), (num_rep, 32)), stride=((0, 1, 4 * num_rep), (4, 4 * num_rep * num_wg))
    )
    return cute.make_tiled_copy(tmem_copy_atom, layout_tv, tiler_mn)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    num_copy_elems = src.shape[0][0]
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric], num_threads: int, num_copy_elems: int = 1, is_async: bool = False
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


@dsl_user_op
def atomic_add_fp32x4(
    a: Float32, b: Float32, c: Float32, d: Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
) -> None:
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    # cache_hint = cutlass.Int64(0x12F0000000000000)
    llvm.inline_asm(
        None,
        [
            gmem_ptr_i64,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        # [gmem_ptr_i64, Float32(a).ir_value(loc=loc, ip=ip), cache_hint.ir_value()],
        "{\n\t"
        # ".reg .b128 abcd;\n\t"
        # "mov.b128 abcd, {$1, $2, $3, $4};\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        # "mov.b128 abcd, {$1, $2, $3, $4};\n\t"
        "mov.f32 abcd.x, $1;\n\t"
        "mov.f32 abcd.y, $2;\n\t"
        "mov.f32 abcd.z, $3;\n\t"
        "mov.f32 abcd.w, $4;\n\t"
        "red.global.add.v4.f32 [$0], abcd;\n\t"
        # "red.global.add.L2::cache_hint.v4.f32 [$0], abcd, 0x14F0000000000000;\n\t"
        "}\n",
        # "red.global.add.L2::cache_hint.f32 [$0], $1, 0x12F0000000000000;",
        # "red.global.add.L2::cache_hint.f32 [$0], $1, $2;",
        "l,f,f,f,f",
        # "l,f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote_fp32x4(
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .v4 .f32 abcd;\n\t"
        "mov.f32 abcd.x, $2;\n\t"
        "mov.f32 abcd.y, $3;\n\t"
        "mov.f32 abcd.z, $4;\n\t"
        "mov.f32 abcd.w, $5;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [$0], abcd, [$1];\n\t"
        "}\n",
        "r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_bulk_s2cluster(
    smem_src_ptr: cute.Pointer,
    smem_dst_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
):
    smem_src_ptr_i32 = smem_src_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_dst_ptr_i32 = set_block_rank(
        smem_dst_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    mbar_ptr_i32 = set_block_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_dst_ptr_i32,
            smem_src_ptr_i32,
            mbar_ptr_i32,
            Int32(size).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $3, [$2];",
        "r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_bulk_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    tma_bar_ptr: cute.Pointer,
    size: int | Int32,
    *,
    loc=None,
    ip=None,
):
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = tma_bar_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, smem_ptr_i32, mbar_ptr_i32, Int32(size).ir_value()],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$0], $3, [$2];",
        "l,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    # cache_hint = cutlass.Int64(0x14F0000000000000)  # EVICT_LAST
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        # [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value(), cache_hint.ir_value()],
        # "cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [$0], [$1], $2, $3;",
        # "l,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def cpasync_bulk_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    group_rank_src = const_expr(cute.rank(src_tensor) - (1 if not single_stage else 0))
    group_rank_dst = const_expr(cute.rank(dst_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    src = cute.group_modes(src_tensor, 0, group_rank_src)
    dst = cute.group_modes(dst_tensor, 0, group_rank_dst)

    def copy_bulk(src_idx, dst_idx, tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(
                atom,
                src[None, src_idx],
                dst[None, dst_idx],
                mbar_ptr=tma_bar_ptr,
                **new_kwargs,
                **kwargs,
            )

    def copy_bulk_single_stage(tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(atom, src, dst, mbar_ptr=tma_bar_ptr, **new_kwargs, **kwargs)

    return copy_bulk if const_expr(not single_stage) else copy_bulk_single_stage


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    def copy_tma_single_stage(**new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn

def swizzle_int(ptr_int: Int32, b: int, m: int, s: int) -> Int32:
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << (m + s)
    return ptr_int ^ ((ptr_int & yyy_msk) >> s)


def swizzle_ptr(ptr: cute.Pointer):
    swz = ptr.type.swizzle_type
    ptr_int = swizzle_int(ptr.toint(), swz.num_bits, swz.num_base, swz.num_shift)
    return cute.make_ptr(ptr.dtype, ptr_int, ptr.memspace, assumed_align=ptr.alignment)

def as_position_independent_swizzle_tensor(tensor: cute.Tensor) -> cute.Tensor:
    outer = tensor.layout
    width = tensor.element_type.width
    swizzle_type = tensor.iterator.type.swizzle_type
    inner = cute.make_swizzle(swizzle_type.num_bits, swizzle_type.num_base, swizzle_type.num_shift)
    # Need to recast the swizzle from byte (e.g. <3, 4, 3> to element units (e.g. <3, 3, 3> for
    # for 16 bits and <3, 2, 3> for 32 bits)
    new_layout = cute.recast_layout(
        width, 8, cute.make_composed_layout(inner, 0, cute.recast_layout(8, width, outer))
    )
    # recast_ptr to remove the pointer swizzle
    return cute.make_tensor(cute.recast_ptr(tensor.iterator, dtype=tensor.element_type), new_layout)


def partition_D_position_independent(
    thr_copy: cute.ThrCopy, tensor: cute.Tensor
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(as_position_independent_swizzle_tensor(tensor)).layout,
    )
    
def _get_smem_store_atom(
    element_type: Type[cute.Numeric],
    transpose: bool = False,
    major_mode_size: Optional[int] = None,
) -> cute.CopyAtom:
    arch = cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum()
    if const_expr(arch < Arch.sm_90 or element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=(2 if not transpose else 1) * element_type.width,
        )
    else:
        num_matrices = (
            4
            if major_mode_size is None or major_mode_size % 16 == 0
            else (2 if major_mode_size % 8 == 0 else 1)
        )
        return cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
            element_type,
        )

def get_smem_store_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    transpose: bool = False,
    position_independent=False,
    major_mode_size: Optional[int] = None,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sC.element_type
    copy_atom = _get_smem_store_atom(dtype, transpose, major_mode_size=major_mode_size)
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    if const_expr(not position_independent):
        tRS_sC = thr_copy.partition_D(sC)
    else:
        tRS_sC = partition_D_position_independent(thr_copy, sC)

    def copy_fn(src: cute.Tensor, dst_idx: Optional[Int32] = None, **new_kwargs):
        dst_tensor = tRS_sC if const_expr(dst_idx is None) else tRS_sC[None, None, None, dst_idx]
        cvt_copy(tiled_copy, src, dst_tensor, retile=True, **new_kwargs)

    return copy_fn, thr_copy, tRS_sC


BIG_INT = 2**30
MAX_INT = 2**31 - 1
BIG_INT_INV = 2**64 // BIG_INT


def create_ragged_tensor_for_tma(
    T: cute.Tensor,
    ragged_dim: int = 0,
    ptr_shift: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    rank = cute.rank(T)
    if ragged_dim < 0:
        ragged_dim += rank
    if ptr_shift:
        assert rank <= 4, "ptr_shift ragged tensor only supports up to 4 dimensions"
        new_shape = T.shape[:ragged_dim] + (BIG_INT,) + T.shape[ragged_dim + 1 :] + (MAX_INT,)
        new_stride = T.stride + (T.stride[ragged_dim],)
        ptr_offset = (None,) * ragged_dim + (-BIG_INT,) + (None,) * (rank - ragged_dim - 1)
        new_ptr = cute.domain_offset(ptr_offset, T).iterator
        return cute.make_tensor(new_ptr, cute.make_layout(new_shape, stride=new_stride))
    else:
        assert rank <= 3, "non-ptr_shift ragged tensor only supports up to 3 dimensions"
        stride_r = T.stride[ragged_dim]
        new_shape = (
            T.shape[:ragged_dim] + (BIG_INT,) + T.shape[ragged_dim + 1 :] + (MAX_INT, MAX_INT)
        )
        new_stride = (
            T.stride[:ragged_dim]
            + (stride_r,)
            + T.stride[ragged_dim + 1 :]
            + (BIG_INT_INV - stride_r, stride_r)
        )
        return cute.make_tensor(T.iterator, cute.make_layout(new_shape, stride=new_stride))


@dsl_user_op
def offset_ragged_tensor(
    T: cute.Tensor,
    offset: Int32,
    length: Int32,
    ragged_dim: int = 0,
    ptr_shift: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    rank = cute.rank(T)
    if ragged_dim < 0:
        ragged_dim += rank
    big_int = cute.size(T, mode=[ragged_dim])
    offset_val = big_int - length
    if ptr_shift:
        # 1-extra-dim: rank = original_rank + 1
        assert rank >= ragged_dim + 2
        offset_tuple = (None,) * ragged_dim + (offset_val,) + (None,) * (rank - ragged_dim - 2)
        index_tuple = (None,) * (rank - 1) + (offset + length,)
    else:
        # 2-extra-dim: rank = original_rank + 2, last 2 modes are the wraparound dims
        assert rank >= ragged_dim + 3
        offset_tuple = (None,) * ragged_dim + (offset_val,) + (None,) * (rank - ragged_dim - 3)
        index_tuple = (None,) * (rank - 2) + (big_int, offset + length)
    return cute.domain_offset(offset_tuple, T[index_tuple])

