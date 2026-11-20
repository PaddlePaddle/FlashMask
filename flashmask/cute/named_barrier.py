# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

import enum


class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()


class NamedBarrierFwdSm100SwapAB(enum.IntEnum):
    """Barrier IDs of the SM100 SwapAB decode kernel; each *WG2 ID is its *WG1 ID + 1.

    IDs 1-15 plus sync_threads on 0 use the full hardware budget of 16 named barriers.
    """

    TmemPtr = 1
    MaxFull = 2
    MaxEmpty = 3
    SumFullWG1 = 4
    SumFullWG2 = 5
    SumEmptyWG1 = 6
    SumEmptyWG2 = 7
    MaxDone = 8
    ODone = 9
    SumDone = 10
    QKIssued = 11
    PVIssued = 12
    MaxTurnWG1 = 13
    MaxTurnWG2 = 14
    # Separate from MaxDone: reusing one ID lets a fast correction warp arrive twice
    # before all warps have contributed to the first generation.
    StatsReady = 15


class NamedBarrierBwd(enum.IntEnum):
    Epilogue = enum.auto()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PdS = enum.auto()
    dQFullWG0 = enum.auto()
    dQFullWG1 = enum.auto()
    dQEmptyWG0 = enum.auto()
    dQEmptyWG1 = enum.auto()
    FlashMask = enum.auto()


class NamedBarrierBwdSm100(enum.IntEnum):
    EpilogueWG1 = enum.auto()
    EpilogueWG2 = enum.auto()
    Compute = enum.auto()
    dQaccReduce = enum.auto()
    # Persistent only: end-of-work-tile fence between the load warp and the
    # compute / reduce warps. sdK aliases sK and sdV aliases sV, so the next
    # tile's K/V (and flashmask) loads must not start before the current tile's
    # dKV epilogue and dQ reduction are done with those buffers.
    TileBoundary = enum.auto()
