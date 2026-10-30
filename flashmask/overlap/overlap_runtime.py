# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FM-4 Overlap Python runtime: ctypes front-end over ``libfm4_overlap.so``.
Importing never loads the .so; ``_load()`` raises only on first use if missing.
"""

import ctypes
import importlib.util
import os
from typing import NamedTuple

_UID_NBYTES = None
_LIB = None


def _find_so(here):
    for f in os.listdir(here):
        if f.startswith("libfm4_overlap") and f.endswith(".so"):
            return os.path.join(here, f)
    return None


def _nccl_lib_path():
    """libnccl.so.2 of the running interpreter's wheel. None if not wheel-installed."""
    override = os.environ.get("FLASHMASK_NCCL_LIB")
    if override:
        return override
    spec = importlib.util.find_spec("nvidia.nccl")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = next(iter(spec.submodule_search_locations))
    for libdir in ("lib", "lib64"):
        candidate = os.path.join(root, libdir, "libnccl.so.2")
        if os.path.exists(candidate):
            return candidate
    return None


def _pin_nccl():
    """Pin the soname so the bridge reuses this libnccl instead of searching.

    The bridge inlines the GIN device API and must bind the exact libnccl it was
    built against; its DT_RUNPATH ranks below LD_LIBRARY_PATH, which often lists
    a stray system libnccl. RTLD_LOCAL suffices: reuse is keyed on the soname.
    """
    path = _nccl_lib_path()
    if path:
        ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)


def _load():
    """Load the bridge .so and bind argtypes/restype. Cached after first call."""
    global _LIB, _UID_NBYTES
    if _LIB is not None:
        return _LIB

    here = os.path.dirname(os.path.abspath(__file__))
    so_path = _find_so(here)
    if so_path is None:
        raise RuntimeError(
            "FM4 overlap extension (libfm4_overlap.so) not found next to "
            f"{__file__}. Built only when the 'ovl' component is selected and "
            "NCCL with GIN support is available."
        )

    _pin_nccl()
    lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)

    lib.fm4_overlap_unique_id_size.argtypes = []
    lib.fm4_overlap_unique_id_size.restype = ctypes.c_size_t
    _UID_NBYTES = int(lib.fm4_overlap_unique_id_size())
    if _UID_NBYTES <= 0:
        raise RuntimeError("FM4 overlap bridge returned an invalid NCCL unique ID size")

    lib.fm4_overlap_get_unique_id.argtypes = [ctypes.c_char_p]
    lib.fm4_overlap_get_unique_id.restype = ctypes.c_int

    # (b, s, h, d, rank, nranks) ints, uid bytes, mask_head int, kv_shared int.
    lib.fm4_overlap_init.argtypes = (
        [ctypes.c_int] * 6 + [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    )
    lib.fm4_overlap_init.restype = ctypes.c_int

    for name in (
        "fm4_overlap_k_data",
        "fm4_overlap_v_data",
        "fm4_overlap_work_done",
        "fm4_overlap_fwd_write_ptr",
    ):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = ctypes.c_uint64
    for name in (
        "fm4_overlap_s_local",
        "fm4_overlap_nranks",
        "fm4_overlap_rank",
        "fm4_overlap_gpus_per_node",
        "fm4_overlap_use_bhsd_layout",
        "fm4_overlap_use_hierarchical",
        "fm4_overlap_comm_rpb",
        "fm4_overlap_num_segments",
        "fm4_overlap_segment_seqlen",
    ):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = ctypes.c_int
    for name in (
        "fm4_overlap_segment_k_data",
        "fm4_overlap_segment_v_data",
        "fm4_overlap_dk_send",
        "fm4_overlap_dv_send",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_int]
        fn.restype = ctypes.c_uint64

    lib.fm4_overlap_update_kv.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]
    lib.fm4_overlap_update_kv.restype = None

    lib.fm4_overlap_run_ag.argtypes = [ctypes.c_int]
    lib.fm4_overlap_run_ag.restype = None

    # (lt_start_ptr, ut_end_ptr, compute_stream) uint64, fwd int.
    lib.fm4_overlap_compute_chunk_mask.argtypes = [ctypes.c_uint64] * 3 + [ctypes.c_int]
    lib.fm4_overlap_compute_chunk_mask.restype = None

    for name in (
        "fm4_overlap_wait_sr_buffer_empty",
        "fm4_overlap_wait_reset_stream_coordinator",
        "fm4_overlap_wait_reduce_done",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_uint64]
        fn.restype = None
    lib.fm4_overlap_reset_ag_counter.argtypes = [ctypes.c_uint64]
    lib.fm4_overlap_reset_ag_counter.restype = None
    lib.fm4_overlap_prepare_dkv_buffer.argtypes = []
    lib.fm4_overlap_prepare_dkv_buffer.restype = None

    lib.fm4_overlap_start_bwd_segment.argtypes = [ctypes.c_int, ctypes.c_uint64]
    lib.fm4_overlap_start_bwd_segment.restype = None
    lib.fm4_overlap_wait_dkv_buffer.argtypes = [ctypes.c_int, ctypes.c_uint64]
    lib.fm4_overlap_wait_dkv_buffer.restype = None
    lib.fm4_overlap_run_rs.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.fm4_overlap_run_rs.restype = None

    lib.fm4_overlap_wait_wptr_init.argtypes = []
    lib.fm4_overlap_wait_wptr_init.restype = None

    _LIB = lib
    return lib


# B/H/D are Python-side (from the K/V shape); stashed to rebuild the gathered view.
_KV_SHAPE = None   # (B, H, D); S_total = s_local() * nranks()

# Cache one NCCL UID per Python process/group generation.
_UID = None
_UID_KEY = None


def is_available():
    """True if the bridge .so can be loaded with NCCL GIN support."""
    try:
        _load()
        return True
    except Exception:
        return False


def bootstrap_unique_id(rank, group=None, *, be=None, dist=None):
    """Generate and broadcast one NCCL UID for the current group generation.

    The id comes from the vendored bridge rather than from the framework, so
    overlap builds its own communicator and needs nothing from the framework
    beyond a byte broadcast.
    """
    global _UID, _UID_KEY
    if dist is None:
        from flashmask.distributed import get_dist

        dist = get_dist(be.name if be is not None else "paddle")

    world_size = dist.world_size(group)
    uid_key = (id(group), world_size) if group is not None else (None, world_size)
    if _UID is not None and _UID_KEY == uid_key:
        return _UID

    lib = _load()
    buf = (ctypes.c_uint8 * _UID_NBYTES)()
    if rank == 0 and lib.fm4_overlap_get_unique_id(ctypes.cast(buf, ctypes.c_char_p)) != 1:
        raise RuntimeError("fm4_overlap_get_unique_id failed")

    src_global = dist.global_ranks(group)[0]
    _UID = dist.broadcast_bytes(buf, src_global, group=group, be=be)
    _UID_KEY = uid_key
    return _UID


def init_overlap(k, v, rank, nranks, uid_bytes, mask_head=1, kv_shared=False):
    """Create or reconfigure the C++ singleton from k/v shape + topology (only
    k.shape is read; the local-KV copy into the SRBuffer happens in update_kv).
    kv_shared: K and V alias one storage, so only K is transported."""
    global _KV_SHAPE
    lib = _load()
    b, s_local, h, d = (int(x) for x in k.shape)
    _KV_SHAPE = (b, h, d)
    rc = lib.fm4_overlap_init(b, s_local, h, d, int(rank), int(nranks), uid_bytes,
                              int(mask_head), int(kv_shared))
    if rc != 1:
        raise RuntimeError("fm4_overlap_init failed")


def ensure_initialized(k, v, group, mask_head=1, kv_shared=False):
    """Bootstrap the unique id once, then forward shape/topology to the C++ singleton
    every step. init_singleton_instance reconfigures (and reallocs the SRBuffer) only
    when they actually change, so unconditional forwarding is cheap and correct."""
    from flashmask.backend import get_backend
    from flashmask.distributed import get_dist

    be = get_backend(k)
    dist = get_dist(be.name)
    # A paddle Group exposes .rank / .world_size as attributes, a torch
    # ProcessGroup only through dist.get_rank(group); go through the adapter.
    rank = dist.rank(group)
    uid = bootstrap_unique_id(rank, group=group, be=be, dist=dist)
    init_overlap(k, v, rank, dist.world_size(group), uid, mask_head=mask_head,
                 kv_shared=kv_shared)


def use_bhsd_layout():
    """Return the communicator's effective SRBuffer layout after initialization."""
    return bool(_load().fm4_overlap_use_bhsd_layout())


def use_hierarchical():
    """Return whether the communicator's effective topology is hierarchical."""
    return bool(_load().fm4_overlap_use_hierarchical())


def _hier_map_chunk(logical_pos, my_pe, total_n_pes, gpus_per_node):
    """Rank whose K/V sits at hierarchical logical position ``logical_pos``; port of
    ``hier::hier_map_chunk(...).target_rank`` (hierarchical_rank_map.cuh)."""
    my_slot = my_pe % gpus_per_node
    my_node = my_pe // gpus_per_node
    num_nodes = total_n_pes // gpus_per_node
    if logical_pos < num_nodes:
        return my_slot + ((my_node + logical_pos) % num_nodes) * gpus_per_node
    adj = logical_pos - num_nodes
    slot = adj // num_nodes + 1
    sub = adj % num_nodes
    base = (my_slot + slot) % gpus_per_node
    return base + ((my_node + sub) % num_nodes) * gpus_per_node


_SR_ORDER_CACHE = {}


def _sr_chunk_order(rank, nranks, gpus_per_node, hierarchical, fwd):
    """Global KV-chunk index held at each SR position of this rank."""
    key = (rank, nranks, gpus_per_node, hierarchical, fwd)
    order = _SR_ORDER_CACHE.get(key)
    if order is None:
        if hierarchical:
            # SR position j holds logical position nranks-1-j (fwd) or j (bwd); hier_seqlen_id.
            positions = [nranks - 1 - j if fwd else j for j in range(nranks)]
            order = tuple(_hier_map_chunk(p, rank, nranks, gpus_per_node) for p in positions)
        else:
            # Circular shift: fwd puts the local chunk last, bwd puts it first.
            shift = rank + 1 if fwd else rank
            order = tuple((shift + j) % nranks for j in range(nranks))
        _SR_ORDER_CACHE[key] = order
    return order


def sr_order_mask(mask, fwd):
    """Reorder a full mask from group-rank KV-chunk order into this rank's SR-buffer order."""
    if _KV_SHAPE is None:
        raise RuntimeError("ensure_initialized must be called before sr_order_mask")
    lib = _load()
    nranks = int(lib.fm4_overlap_nranks())
    s_local = int(lib.fm4_overlap_s_local())
    assert mask.shape[2] == s_local * nranks, (
        f"mask key length {mask.shape[2]} != S_local {s_local} * group size {nranks}"
    )
    order = _sr_chunk_order(
        int(lib.fm4_overlap_rank()), nranks, int(lib.fm4_overlap_gpus_per_node()),
        bool(lib.fm4_overlap_use_hierarchical()), fwd,
    )
    if order == tuple(range(nranks)):
        return mask
    from flashmask.backend import get_backend

    be = get_backend(mask)
    chunks = be.split(mask, nranks, axis=2)
    return be.concat([chunks[c] for c in order], axis=2)


def _data_ptr(t):
    return int(t.data_ptr())


def _s_total():
    """Total gathered seqlen after the all-gather: S_local * nranks."""
    lib = _load()
    return lib.fm4_overlap_s_local() * lib.fm4_overlap_nranks()


def update_kv(k, v, fwd=True):
    """Copy local K/V into the SRBuffer (cudaMemcpyAsync on comm_stream)."""
    _load().fm4_overlap_update_kv(_data_ptr(k), _data_ptr(v), 1 if fwd else 0)


def run_ag(fwd=True):
    """Launch the AG remote-get kernel; the forward one advances fm4_overlap_fwd_write_ptr."""
    _load().fm4_overlap_run_ag(1 if fwd else 0)


def wait_sr_buffer_empty(compute_stream):
    """Stream-order compute_stream behind the prior step draining the SRBuffer, so the
    next update_kv waits for the prior AG's consumers without blocking overlap."""
    _load().fm4_overlap_wait_sr_buffer_empty(int(compute_stream))


def wait_reset_stream_coordinator(compute_stream):
    """Hold compute_stream until the comm kernel has occupied its SMs (deadlock guard if
    compute grabbed every SM first)."""
    _load().fm4_overlap_wait_reset_stream_coordinator(int(compute_stream))


def reset_ag_counter(compute_stream):
    """Reset the persistent AG scheduler counter before each gather."""
    _load().fm4_overlap_reset_ag_counter(int(compute_stream))


def wait_wptr_init():
    """comm_stream waits for the counter reset (wptr_init event) to land before the
    remote-get kernel reads block_cnt_semaphore. Pairs with reset_ag_counter."""
    _load().fm4_overlap_wait_wptr_init()


def _sparse_chunk_mask_cols(startend_row_indices):
    """Slice (lt_start, ut_end) columns into contiguous (B, H_mask, S_total) int32."""
    num_vecs = startend_row_indices.shape[-1]
    # 4-vec is rejected: lt_end / ut_start are not plumbed through the bridge, so the skip
    # bitmap would over-report fully-masked chunks and silently drop their contribution.
    assert num_vecs == 2, f"overlap mask must be 2-vec (non-causal), got {num_vecs}"
    lt_start = startend_row_indices[..., 0].contiguous()
    ut_end = startend_row_indices[..., 1].contiguous()
    return lt_start, ut_end


def compute_chunk_mask_sparse(startend_row_indices, compute_stream, fwd=True):
    """Drive copy_chunk_mask so the AG kernel skips fully-masked KV chunks. The check
    runs async on compute_stream reading the two tensors, so they MUST outlive run_ag's
    launch -- returned for keepalive."""
    lt_start, ut_end = _sparse_chunk_mask_cols(startend_row_indices)
    _load().fm4_overlap_compute_chunk_mask(
        _data_ptr(lt_start), _data_ptr(ut_end), int(compute_stream), 1 if fwd else 0
    )
    return lt_start, ut_end


def current_stream_handle(be):
    """Current compute-stream handle as a plain int (the cuda_stream pointer)."""
    return int(be.cuda_stream())


class SrKvView(NamedTuple):
    """Cross-jit-safe handle to the gathered SRBuffer K/V."""
    k_addr: int
    v_addr: int
    shape: tuple  # (B, S_total, H, D)


class ForwardAgArgs(NamedTuple):
    """Forward SR view plus the persistent device frontier used by the compute gate."""
    view: SrKvView
    write_ptr_addr: int  # device int*; invalidated by communicator reconfiguration
    kv_chunk_size: int   # S_local


class BackwardAgArgs(NamedTuple):
    """Split-AG metadata; retain ``mask_keepalive`` until all dependent launches are enqueued."""
    num_segments: int
    segment_seqlen: int
    work_done_addr: int  # device int* base of the per-work done bitmap
    comm_rpb: int
    shape: tuple  # logical (B, S_segment, H, D) view of one segment
    mask_keepalive: object  # lifetime ownership only; not read by the CuTe kernel

    def kv_view(self, segment_idx):
        lib = _load()
        return SrKvView(
            k_addr=lib.fm4_overlap_segment_k_data(int(segment_idx)),
            v_addr=lib.fm4_overlap_segment_v_data(int(segment_idx)),
            shape=self.shape,
        )

    def dkv_send_addrs(self, segment_idx):
        lib = _load()
        return (
            lib.fm4_overlap_dk_send(int(segment_idx)),
            lib.fm4_overlap_dv_send(int(segment_idx)),
        )


def sr_kv_view_args():
    """SrKvView for the gathered SRBuffer K/V (post all-gather). SRBuffer is bf16."""
    if _KV_SHAPE is None:
        raise RuntimeError("init_overlap must be called before sr_kv_view_args")
    lib = _load()
    b, h, d = _KV_SHAPE
    return SrKvView(
        k_addr=lib.fm4_overlap_k_data(),
        v_addr=lib.fm4_overlap_v_data(),
        shape=(b, _s_total(), h, d),
    )


def _start_ag(k, v, startend_row_indices, compute_stream, *, fwd):
    wait_sr_buffer_empty(compute_stream)
    mask_keepalive = compute_chunk_mask_sparse(
        startend_row_indices, compute_stream, fwd=fwd
    )
    update_kv(k, v, fwd=fwd)
    reset_ag_counter(compute_stream)
    wait_wptr_init()
    run_ag(fwd=fwd)
    wait_reset_stream_coordinator(compute_stream)
    return sr_kv_view_args(), mask_keepalive


def start_forward_ag(k, v, startend_row_indices, compute_stream):
    """Launch forward sparse AG asynchronously for cumulative-frontier gating."""
    view, _ = _start_ag(k, v, startend_row_indices, compute_stream, fwd=True)
    write_ptr_addr = int(_load().fm4_overlap_fwd_write_ptr())
    return ForwardAgArgs(view, write_ptr_addr, int(k.shape[1]))


def start_backward_ag(k, v, startend_row_indices, compute_stream):
    """Prepare multi-stage backward AG/RS and launch split AG segment 0."""
    lib = _load()
    lib.fm4_overlap_prepare_dkv_buffer()
    mask_keepalive = compute_chunk_mask_sparse(
        startend_row_indices, compute_stream, fwd=False
    )
    wait_sr_buffer_empty(compute_stream)
    update_kv(k, v, fwd=False)

    num_segments = int(lib.fm4_overlap_num_segments())
    if num_segments <= 1:
        raise RuntimeError("FM-4 backward overlap requires multi-stage RS")
    segment_seqlen = int(lib.fm4_overlap_segment_seqlen())
    comm_rpb = int(lib.fm4_overlap_comm_rpb())
    work_done_addr = int(lib.fm4_overlap_work_done())
    if segment_seqlen <= 0 or comm_rpb <= 0 or work_done_addr == 0:
        raise RuntimeError("invalid FM-4 backward overlap metadata")

    b, h, d = _KV_SHAPE
    lib.fm4_overlap_start_bwd_segment(0, int(compute_stream))
    return BackwardAgArgs(
        num_segments=num_segments,
        segment_seqlen=segment_seqlen,
        work_done_addr=work_done_addr,
        comm_rpb=comm_rpb,
        shape=(b, segment_seqlen, h, d),
        mask_keepalive=mask_keepalive,
    )


def start_backward_segment(segment_idx, compute_stream):
    _load().fm4_overlap_start_bwd_segment(int(segment_idx), int(compute_stream))


def wait_dkv_buffer(segment_idx, compute_stream):
    _load().fm4_overlap_wait_dkv_buffer(int(segment_idx), int(compute_stream))


def run_backward_rs(dk, dv, segment_idx, compute_stream):
    _load().fm4_overlap_run_rs(
        _data_ptr(dk), _data_ptr(dv), int(segment_idx), int(compute_stream)
    )


def wait_backward_rs(compute_stream):
    _load().fm4_overlap_wait_reduce_done(int(compute_stream))
