# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

from typing import Optional, Tuple
from dataclasses import dataclass, fields

try:
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import cutlass
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr

from . import utils
from .fast_math import clz
from cutlass.cute import FastDivmodDivisor


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self._is_valid_tile, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@dataclass
class ParamsBase:
    def __extract_mlir_values__(self):
        all_fields = [getattr(self, field.name) for field in fields(self)]
        non_constexpr_fields = [f for f in all_fields if not isinstance(f, cutlass.Constexpr)]
        values, self._values_pos = [], []
        for obj in non_constexpr_fields:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        all_fields = {field.name: getattr(self, field.name) for field in fields(self)}
        constexpr_fields = {n: f for n, f in all_fields.items() if isinstance(f, cutlass.Constexpr)}
        non_constexpr_fields = {
            n: f for n, f in all_fields.items() if not isinstance(f, cutlass.Constexpr)
        }
        for (name, field), n_items in zip(non_constexpr_fields.items(), self._values_pos):
            non_constexpr_fields[name] = cutlass.new_from_mlir_values(field, values[:n_items])
            values = values[n_items:]
        return self.__class__(**non_constexpr_fields, **constexpr_fields)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    # How `num_block` relates to the CTAs of a cluster. Two conventions exist and they
    # are NOT interchangeable:
    #   False (default): num_block counts the blocks of ONE CTA, and blockIdx.x IS that
    #     CTA's block index -- the CTAs of a pair work on DIFFERENT blocks (the SM100 bwd:
    #     each CTA owns its own KV tile; the pair index is n_block // cluster_shape_mn[0]).
    #   True: num_block counts the work tiles of a CTA PAIR, so every CTA of the cluster
    #     must resolve the SAME tile index and the MMA splits that tile's M across the pair
    #     (the SM100 2-CTA fwd). The grid is then num_block * cluster_shape_mn[0].
    cluster_share_tile: cutlass.Constexpr[bool] = False
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        cluster_share_tile: cutlass.Constexpr[bool] = False

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
                args.cluster_shape_mn,
                args.cluster_share_tile,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        blk_coord = cute.arch.block_idx()
        if const_expr(params.cluster_share_tile and params.cluster_shape_mn[0] > 1):
            # 2-CTA UMMA with a shared work tile: every CTA of a cluster resolves the SAME
            # tile and the MMA splits the tile's M rows across the pair. Keeping the tile
            # index shared is what keeps the flashmask block list / valid_block_count
            # semantics per-tile (both CTAs walk the identical n_block sequence, so they
            # never block each other on a skipped block). Only kernels that opt in via
            # cluster_share_tile want this; see TileSchedulerArguments.cluster_share_tile.
            blk_coord = (
                blk_coord[0] // params.cluster_shape_mn[0],
                blk_coord[1],
                blk_coord[2],
            )
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        # TODO: this hard-codes the fact that we only use cluster = (1, 1) or (2, 1)
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        # Shared tile: num_block counts CTA-pair tiles, so the grid needs one block per CTA.
        # Otherwise num_block already counts per-CTA blocks and only has to be rounded up to
        # a whole number of clusters.
        num_blk_x = (
            params.num_block * params.cluster_shape_mn[0]
            if const_expr(params.cluster_share_tile)
            else cute.round_up(params.num_block, params.cluster_shape_mn[0])
        )
        return (
            num_blk_x,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentTileScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return StaticPersistentTileScheduler.Params(
                FastDivmodDivisor(args.num_block), FastDivmodDivisor(args.num_head), total_blocks
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentTileScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return StaticPersistentTileScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

    # @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        hn_idx, block_idx = divmod(self._tile_idx, self.params.num_block_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._tile_idx += cute.arch.grid_dim()[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._tile_idx],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)


class StaticPersistentClusterTileScheduler:
    """Persistent scheduler that hands out work per CTA *pair* instead of per CTA.

    StaticPersistentTileScheduler linearizes num_block * num_head * num_batch and gives
    CTA i the tile i, which breaks a (2, 1) cluster: with an odd num_block the two CTAs
    of a cluster would land in different (head, batch), and the grid size
    min(sm_count, total_blocks) is not necessarily a multiple of the cluster size.
    Here the unit of scheduling is the pair, so both CTAs of a cluster always resolve
    the same (head, batch) and adjacent n_block, exactly like the non-persistent
    round_up(num_block, cluster) grid. For an odd num_block the last pair's second CTA
    gets n_block >= num_block and is handled by the same in-kernel guards as before.

    Only for the `cluster_share_tile == False` convention (the SM100 bwd), where
    num_block counts the blocks of ONE CTA.
    """

    @dataclass
    class Params(ParamsBase):
        # Plain Int32, not FastDivmodDivisor: the two divisions happen once per work tile
        # (not per m iteration), so the fast path buys nothing, and pairs_per_hb is 1
        # whenever seqlen_k <= tile_n * cluster -- a divisor value the FastDivmod magic
        # numbers are not worth trusting for a scheduler that produces TMA coordinates.
        pairs_per_hb: Int32
        num_head: Int32
        total_pairs: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "StaticPersistentClusterTileScheduler.Params":
            assert args.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
            assert not args.cluster_share_tile, (
                "StaticPersistentClusterTileScheduler assumes per-CTA num_block"
            )
            cluster = args.cluster_shape_mn[0]
            pairs_per_hb = cute.ceil_div(args.num_block, cluster)
            return StaticPersistentClusterTileScheduler.Params(
                pairs_per_hb=Int32(pairs_per_hb),
                num_head=Int32(args.num_head),
                total_pairs=Int32(pairs_per_hb * args.num_head * args.num_batch),
                cluster_shape_mn=args.cluster_shape_mn,
            )

    def __init__(self, params: Params, pair_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._pair_idx = pair_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return StaticPersistentClusterTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "StaticPersistentClusterTileScheduler":
        pair_idx = cute.arch.block_idx()[0] // params.cluster_shape_mn[0]
        return StaticPersistentClusterTileScheduler(params, pair_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        cluster = params.cluster_shape_mn[0]
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        # A cluster launch needs grid.x to be a multiple of the cluster size; rounding up
        # can overshoot total_pairs * cluster, and those CTAs just see no valid tile.
        num_blk_x = cute.round_up(cutlass.min(sm_count, params.total_pairs * cluster), cluster)
        return (num_blk_x, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        hb_idx = self._pair_idx // params.pairs_per_hb
        pair_in_hb = self._pair_idx - hb_idx * params.pairs_per_hb
        batch_idx = hb_idx // params.num_head
        head_idx = hb_idx - batch_idx * params.num_head
        if const_expr(params.cluster_shape_mn[0] > 1):
            cta_rank_in_cluster = cute.arch.block_in_cluster_idx()[0]
            block_idx = pair_in_hb * params.cluster_shape_mn[0] + cta_rank_in_cluster
        else:
            block_idx = pair_in_hb
        is_valid = self._pair_idx < params.total_pairs
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._pair_idx += cute.arch.grid_dim()[0] // self.params.cluster_shape_mn[0]

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._pair_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._pair_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return StaticPersistentClusterTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_splits: Int32
        num_block: Int32
        l2_minor: Int32
        num_block_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        is_split_kv: cutlass.Constexpr[bool] = False
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        cluster_share_tile: cutlass.Constexpr[bool] = False

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTScheduler.Params":
            size_one_kv_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            size_one_head = size_one_kv_head
            size_l2 = 50 * 1024 * 1024  # 40 MB for K & V
            # Swizzle is the size of each "section". Round swizzle to a power of 2
            # Need to be careful about the case where only one head will fit
            # swizzle is how many heads can fit in L2
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # Seems faster if swizzle if a power of 2
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # swizzle = 1 if size_l2 < size_one_head else (size_l2 // size_one_head)
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            return SingleTileLPTScheduler.Params(
                total_blocks=args.num_block * args.num_head * args.num_batch,
                num_block=args.num_block,
                l2_minor=Int32(swizzle),
                num_block_divmod=FastDivmodDivisor(args.num_block),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * args.num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                num_splits=args.num_splits,
                is_split_kv=args.is_split_kv,
                cluster_shape_mn=args.cluster_shape_mn,
                cluster_share_tile=args.cluster_share_tile,
            )

    def __init__(self, params: Params, tile_idx: Int32, split_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTScheduler":
        tile_idx, split_idx, _ = cute.arch.block_idx()
        if const_expr(params.cluster_share_tile and params.cluster_shape_mn[0] > 1):
            # See SingleTileScheduler.create: all CTAs of a cluster share the work tile.
            tile_idx = tile_idx // params.cluster_shape_mn[0]
        return SingleTileLPTScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        assert params.cluster_shape_mn[1] == 1, "Only cluster_shape_mn[1] == 1 is supported"
        num_blk_x = (
            params.total_blocks * params.cluster_shape_mn[0]
            if const_expr(params.cluster_share_tile)
            else params.total_blocks
        )
        return (
            num_blk_x,
            params.num_splits,
            Int32(1),
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(self._tile_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        # Longest-processing-time-first
        block = params.num_block - 1 - block
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(self._split_idx)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx, self._split_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx, self._split_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)


class SingleTileLPTBwdScheduler:
    @dataclass
    class Params(ParamsBase):
        total_blocks: Int32
        num_block: Int32
        l2_minor: Int32
        num_head_divmod: FastDivmodDivisor
        l2_minor_divmod: FastDivmodDivisor
        l2_major_divmod: FastDivmodDivisor
        l2_minor_residual_divmod: FastDivmodDivisor
        num_hb_quotient: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
        spt: cutlass.Constexpr[bool] = True

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileLPTBwdScheduler.Params":
            size_l2 = 50 * 1024 * 1024
            size_one_qdo_head = args.seqlen_k * (args.headdim + args.headdim_v) * args.element_size
            # size_one_dqaccum_head = args.seqlen_k * (args.headdim) * 4
            size_one_dqaccum_head = 0
            size_one_head = size_one_qdo_head + size_one_dqaccum_head
            log2_floor = lambda n: 31 - clz(n)
            swizzle = 1 if size_l2 < size_one_head else (1 << log2_floor(size_l2 // size_one_head))
            # swizzle = 8
            # If we're in the last section (called residual), we don't want to divide by
            # swizzle. Instead we want to divide by the remainder.
            num_hb_quotient = (args.num_head * args.num_batch) // swizzle
            num_hb_remainder = (args.num_head * args.num_batch) % swizzle
            num_block = cute.ceil_div(args.num_block, args.cluster_shape_mn[0])
            return SingleTileLPTBwdScheduler.Params(
                total_blocks=(num_block * args.cluster_shape_mn[0])
                * args.num_head
                * args.num_batch,
                num_block=num_block,
                l2_minor=Int32(swizzle),
                num_head_divmod=FastDivmodDivisor(args.num_head),
                l2_minor_divmod=FastDivmodDivisor(swizzle),
                l2_major_divmod=FastDivmodDivisor(swizzle * num_block),
                l2_minor_residual_divmod=FastDivmodDivisor(
                    max(num_hb_remainder, 1)
                ),  # don't divide by 0
                num_hb_quotient=Int32(num_hb_quotient),
                cluster_shape_mn=args.cluster_shape_mn,
                spt=args.lpt,
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return SingleTileLPTBwdScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileLPTBwdScheduler":
        tile_idx = cute.arch.block_idx()[0]
        return SingleTileLPTBwdScheduler(params, tile_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (params.total_blocks, Int32(1), Int32(1))

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        cluster_idx = self._tile_idx // self.params.cluster_shape_mn[0]
        params = self.params
        # Implement LPT scheduling coordinate calculation
        bidhb, l2_mod = divmod(cluster_idx, params.l2_major_divmod)
        # If we're in the last section (called residual), we don't want to divide by
        # swizzle. Instead we want to divide by the remainder.
        block, bidhb_residual = 0, 0
        if bidhb < params.num_hb_quotient:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_divmod)
        else:
            block, bidhb_residual = divmod(l2_mod, params.l2_minor_residual_divmod)
        bidhb_actual = bidhb * params.l2_minor + bidhb_residual
        batch_idx, head_idx = divmod(bidhb_actual, params.num_head_divmod)
        if cutlass.const_expr(params.spt):
            block = params.num_block - 1 - block
        if cutlass.const_expr(params.cluster_shape_mn[0] > 1):
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
            block = block * params.cluster_shape_mn[0] + bidx_in_cluster[0]
        is_valid = self._tile_idx < params.total_blocks
        return WorkTileInfo((Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        # Single tile scheduler - set to invalid tile_idx to indicate no more work
        self._tile_idx = self.params.total_blocks

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)



class DynamicPersistentTileScheduler:
    """Persistent scheduler with dynamic (atomic-ticket) work distribution.

    One CTA per SM; the first tile of a CTA is ``blockIdx.x``, every further tile is drawn
    from a global ticket counter (``atomicAdd + gridDim.x``), so a CTA that finishes a light
    tile early simply takes the next one (no static round-robin imbalance).

    Only the *producer* warp (``is_producer=True``) draws tickets. It publishes each ticket
    through a small smem ring (``sched_tile[ring_stages]`` + full/empty mbarriers); every
    other warp of the CTA reads the same sequence from the ring. ``full`` expects one arrive
    (producer lane 0), ``empty`` one arrive per consumer thread. Each warp role must call
    ``advance_to_next_work`` exactly once per tile, including the last one, which returns the
    invalid sentinel (``tile_idx >= total_blocks``).

    Ticket -> tile: (batch, head)-major so the CTAs in flight share K/V in L2, and inside a
    (batch, head) the m_blocks go from the last to the first (longest-processing-time first
    for causal-like masks). The counter (``counter[0]``) and a finished-CTA count
    (``counter[1]``) are reset by the last CTA to draw its final ticket, so the buffer is
    reusable by the next launch on the stream without a host-side memset.
    """

    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_block_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks: Int32

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "DynamicPersistentTileScheduler.Params":
            total_blocks = args.num_block * args.num_head * args.num_batch
            return DynamicPersistentTileScheduler.Params(
                args.num_block,
                FastDivmodDivisor(args.num_block),
                FastDivmodDivisor(args.num_head),
                total_blocks,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        ring_idx: Int32,
        ring_phase: Int32,
        sched_tile: cute.Tensor,
        sched_mbar: cute.Pointer,
        counter: cute.Tensor,
        ring_stages: int,
        num_consumer_threads: int,
        is_producer: bool,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._ring_idx = ring_idx
        self._ring_phase = ring_phase
        self._sched_tile = sched_tile
        self._sched_mbar = sched_mbar
        self._counter = counter
        self._ring_stages = ring_stages
        self._num_consumer_threads = num_consumer_threads
        self._is_producer = is_producer
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None) -> Params:
        return DynamicPersistentTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params,
        sched_tile: cute.Tensor = None,
        sched_mbar: cute.Pointer = None,
        counter: cute.Tensor = None,
        ring_stages: int = 4,
        num_consumer_threads: int = 0,
        is_producer: bool = False,
        *,
        loc=None,
        ip=None,
    ) -> "DynamicPersistentTileScheduler":
        return DynamicPersistentTileScheduler(
            params,
            Int32(cute.arch.block_idx()[0]),
            Int32(0),
            Int32(0),
            sched_tile,
            sched_mbar,
            counter,
            ring_stages,
            num_consumer_threads,
            is_producer,
            loc=loc,
            ip=ip,
        )

    # called by host
    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        sm_count = cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
        return (cutlass.min(sm_count, params.total_blocks), Int32(1), Int32(1))

    @staticmethod
    def init_barriers(sched_mbar: cute.Pointer, ring_stages: int, num_consumer_threads: int):
        """Called by one warp before the CTA-wide barrier-init sync."""
        for i in range(ring_stages):  # plain helper (not @cute.jit): Python-unrolled
            cute.arch.mbarrier_init(sched_mbar + i, 1)
            cute.arch.mbarrier_init(sched_mbar + ring_stages + i, num_consumer_threads)

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        bh_idx, block_rev = divmod(self._tile_idx, self.params.num_block_divmod)
        batch_idx, head_idx = divmod(bh_idx, self.params.num_head_divmod)
        block_idx = self.params.num_block - 1 - block_rev
        is_valid = self._tile_idx < self.params.total_blocks
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    @cute.jit
    def _producer_next(self) -> Int32:
        """Producer warp: draw the next ticket and publish it in the ring."""
        full = self._sched_mbar + self._ring_idx
        empty = self._sched_mbar + self._ring_stages + self._ring_idx
        # Slot free once every consumer thread has read the ticket published there
        # ring_stages tiles ago (phase starts at 1 so the first lap passes).
        cute.arch.mbarrier_wait(empty, self._ring_phase ^ 1)
        lane = cute.arch.lane_idx()
        ticket = Int32(0)
        if lane == 0:
            ticket = (
                cute.arch.atomic_add(self._counter.iterator, Int32(1), sem="relaxed", scope="gpu")
                + cute.arch.grid_dim()[0]
            )
            self._sched_tile[self._ring_idx] = ticket
            if ticket >= self.params.total_blocks:
                # This CTA drew its last ticket. The last CTA to get here resets the
                # counters for the next launch: every other CTA has already drawn its
                # final ticket, so no further atomic touches counter[0].
                done = cute.arch.atomic_add(
                    self._counter.iterator + 1, Int32(1), sem="acq_rel", scope="gpu"
                )
                if done == cute.arch.grid_dim()[0] - 1:
                    self._counter[0] = Int32(0)
                    self._counter[1] = Int32(0)
            # mbarrier.arrive has release semantics: the ticket store is visible to the
            # consumers that observe the phase flip.
            cute.arch.mbarrier_arrive(full)
        ticket = cute.arch.shuffle_sync(ticket, 0)
        return ticket

    @cute.jit
    def _consumer_next(self) -> Int32:
        full = self._sched_mbar + self._ring_idx
        empty = self._sched_mbar + self._ring_stages + self._ring_idx
        cute.arch.mbarrier_wait(full, self._ring_phase)
        ticket = self._sched_tile[self._ring_idx]
        cute.arch.mbarrier_arrive(empty)
        return ticket

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self._is_producer):
            ticket = self._producer_next()
        else:
            ticket = self._consumer_next()
        self._tile_idx = ticket
        # Plain arithmetic (no dynamic `if`): attributes of self cannot be carried out of an
        # scf.if region.
        next_idx = (self._ring_idx + 1) % self._ring_stages
        self._ring_phase = self._ring_phase ^ Int32(next_idx == 0)
        self._ring_idx = next_idx

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.params, self._tile_idx, self._ring_idx, self._ring_phase,
            self._sched_tile, self._sched_mbar, self._counter,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.params, self._tile_idx, self._ring_idx, self._ring_phase,
                self._sched_tile, self._sched_mbar, self._counter,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return DynamicPersistentTileScheduler(
            *obj_list,
            self._ring_stages,
            self._num_consumer_threads,
            self._is_producer,
            loc=self._loc,
        )
