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

    # (b, s, h, d, rank, nranks) ints, uid bytes, kv_shared int -> config generation.
    lib.fm4_overlap_init.argtypes = [ctypes.c_int] * 6 + [ctypes.c_char_p, ctypes.c_int]
    lib.fm4_overlap_init.restype = ctypes.c_uint64

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

    # (mask_ptr) uint64, num_heads int, (batch_stride, head_stride) int64, (k_ptr, v_ptr,
    # compute_stream) uint64, fwd int.
    lib.fm4_overlap_start_ag.argtypes = [
        ctypes.c_uint64, ctypes.c_int, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int,
    ]
    lib.fm4_overlap_start_ag.restype = None
    lib.fm4_overlap_symmetric_buffer_bytes.argtypes = []
    lib.fm4_overlap_symmetric_buffer_bytes.restype = ctypes.c_uint64

    lib.fm4_overlap_wait_reduce_done.argtypes = [ctypes.c_uint64]
    lib.fm4_overlap_wait_reduce_done.restype = None
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

    _LIB = lib
    return lib


# Communicator values of the current config generation (_CommLayout); see init_overlap.
_LAYOUT = None

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


def init_overlap(k, v, rank, nranks, uid_bytes, kv_shared=False):
    """Create or reconfigure the C++ singleton from k/v shape + topology (only
    k.shape is read; the local-KV copy into the SRBuffer happens in start_*_ag).
    kv_shared: K and V alias one storage, so only K is transported."""
    global _LAYOUT
    lib = _load()
    b, s_local, h, d = (int(x) for x in k.shape)
    generation = int(lib.fm4_overlap_init(b, s_local, h, d, int(rank), int(nranks), uid_bytes,
                                          int(kv_shared)))
    if generation == 0:
        raise RuntimeError("fm4_overlap_init failed")
    # The generation moves on every rebuild, reconfigure (B/S/H/D) or kv_shared flip, so
    # an unchanged one guarantees every queried size and address is still current.
    if _LAYOUT is None or _LAYOUT.generation != generation:
        _LAYOUT = _query_layout(lib, generation, b, h, d)


def ensure_initialized(k, v, group, kv_shared=False):
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
    init_overlap(k, v, rank, dist.world_size(group), uid, kv_shared=kv_shared)


def _layout():
    if _LAYOUT is None:
        raise RuntimeError("ensure_initialized must be called first")
    return _LAYOUT


def use_bhsd_layout():
    """Return the communicator's effective SRBuffer layout after initialization."""
    return _layout().bhsd


def use_hierarchical():
    """Return whether the communicator's effective topology is hierarchical."""
    return _layout().hierarchical


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
    layout = _layout()
    nranks, s_local = layout.nranks, layout.s_local
    assert mask.shape[2] == s_local * nranks, (
        f"mask key length {mask.shape[2]} != S_local {s_local} * group size {nranks}"
    )
    order = _sr_chunk_order(
        layout.rank, nranks, layout.gpus_per_node, layout.hierarchical, fwd,
    )
    if order == tuple(range(nranks)):
        return mask
    from flashmask.backend import get_backend

    be = get_backend(mask)
    if not layout.hierarchical:
        # SR position j holds chunk (order[0] + j) % nranks: one roll by order[0] chunks.
        return be.roll(mask, -order[0] * s_local, axis=2)
    chunks = be.split(mask, nranks, axis=2)
    return be.concat([chunks[c] for c in order], axis=2)


def _data_ptr(t):
    return int(t.data_ptr())


def symmetric_buffer_bytes():
    """Total bytes of the communicator's registered symmetric (NCCL window) buffers."""
    return int(_load().fm4_overlap_symmetric_buffer_bytes())


def _chunk_mask_pairs(startend_row_indices):
    """The (B, H_mask, S_total, 2) [lt_start, ut_end] mask as the chunk-mask kernel reads
    it in place: contiguous, 16-byte aligned (S_total is even, so the batch / head strides
    are multiples of 4 int32). Returns (mask, batch_stride, head_stride)."""
    # 4-vec is rejected: lt_end / ut_start are not plumbed through the bridge, so the skip
    # bitmap would over-report fully-masked chunks and silently drop their contribution.
    num_vecs = startend_row_indices.shape[-1]
    assert num_vecs == 2, f"overlap mask must be 2-vec (non-causal), got {num_vecs}"
    from flashmask.backend import get_backend

    be = get_backend(startend_row_indices)
    mask = be.contiguous(startend_row_indices)
    assert be.data_ptr(mask) % 16 == 0, "overlap mask must be 16-byte aligned"
    batch_stride, head_stride = be.strides(mask)[:2]
    return mask, batch_stride, head_stride


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
    segment_addrs: tuple  # per segment: (k_addr, v_addr, dk_send_addr, dv_send_addr)
    mask_keepalive: object  # lifetime ownership only; not read by the CuTe kernel

    def kv_view(self, segment_idx):
        k_addr, v_addr, _, _ = self.segment_addrs[segment_idx]
        return SrKvView(k_addr=k_addr, v_addr=v_addr, shape=self.shape)

    def dkv_send_addrs(self, segment_idx):
        return self.segment_addrs[segment_idx][2:]


class _CommLayout(NamedTuple):
    """Communicator values of one config generation (the fm4_overlap_init return)."""
    generation: int
    nranks: int
    s_local: int
    rank: int
    gpus_per_node: int
    hierarchical: bool
    bhsd: bool
    fwd: ForwardAgArgs
    bwd: BackwardAgArgs  # mask_keepalive is None; start_backward_ag fills it per call


def _query_layout(lib, generation, b, h, d):
    """Read every per-generation value once. b/h/d are the K shape the config was built
    from (the config generation changes with them); the SRBuffer is bf16."""
    nranks = int(lib.fm4_overlap_nranks())
    s_local = int(lib.fm4_overlap_s_local())
    num_segments = int(lib.fm4_overlap_num_segments())
    segment_seqlen = int(lib.fm4_overlap_segment_seqlen())
    comm_rpb = int(lib.fm4_overlap_comm_rpb())
    work_done_addr = int(lib.fm4_overlap_work_done())
    if segment_seqlen <= 0 or comm_rpb <= 0 or work_done_addr == 0:
        raise RuntimeError("invalid FM-4 backward overlap metadata")
    fwd = ForwardAgArgs(
        view=SrKvView(
            k_addr=int(lib.fm4_overlap_k_data()),
            v_addr=int(lib.fm4_overlap_v_data()),
            shape=(b, s_local * nranks, h, d),
        ),
        write_ptr_addr=int(lib.fm4_overlap_fwd_write_ptr()),
        kv_chunk_size=s_local,
    )
    bwd = BackwardAgArgs(
        num_segments=num_segments,
        segment_seqlen=segment_seqlen,
        work_done_addr=work_done_addr,
        comm_rpb=comm_rpb,
        shape=(b, segment_seqlen, h, d),
        segment_addrs=tuple(
            (
                int(lib.fm4_overlap_segment_k_data(i)),
                int(lib.fm4_overlap_segment_v_data(i)),
                int(lib.fm4_overlap_dk_send(i)),
                int(lib.fm4_overlap_dv_send(i)),
            )
            for i in range(num_segments)
        ),
        mask_keepalive=None,
    )
    return _CommLayout(
        generation=generation,
        nranks=nranks,
        s_local=s_local,
        rank=int(lib.fm4_overlap_rank()),
        gpus_per_node=int(lib.fm4_overlap_gpus_per_node()),
        hierarchical=bool(lib.fm4_overlap_use_hierarchical()),
        bhsd=bool(lib.fm4_overlap_use_bhsd_layout()),
        fwd=fwd,
        bwd=bwd,
    )


def _start_ag(k, v, startend_row_indices, compute_stream, fwd):
    """fm4_overlap_start_ag: chunk-skip mask, local K/V staging, the (segment-0) AG launch
    and the stream coordinator. The check reads the mask in place (or a contiguous copy),
    returned for keepalive."""
    mask, batch_stride, head_stride = _chunk_mask_pairs(startend_row_indices)
    _load().fm4_overlap_start_ag(
        _data_ptr(mask), int(mask.shape[1]), batch_stride, head_stride,
        _data_ptr(k), _data_ptr(v), int(compute_stream), 1 if fwd else 0,
    )
    return mask


def start_forward_ag(k, v, startend_row_indices, compute_stream):
    """Launch forward sparse AG asynchronously for cumulative-frontier gating."""
    _start_ag(k, v, startend_row_indices, compute_stream, fwd=True)
    return _layout().fwd


def start_backward_ag(k, v, startend_row_indices, compute_stream):
    """Prepare multi-stage backward AG/RS and launch split AG segment 0."""
    mask = _start_ag(k, v, startend_row_indices, compute_stream, fwd=False)
    return _layout().bwd._replace(mask_keepalive=mask)


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
