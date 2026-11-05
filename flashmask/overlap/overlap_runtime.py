# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FM-4 Overlap Python runtime: ctypes front-end over ``libfm4_overlap.so``.
Importing never loads the .so; ``_load()`` raises only on first use if missing.

Stream contract: all overlap calls of a process run on the current compute stream; the
SR buffer reuse guard does not wait for other streams. CUDA graph capture is rejected.
"""

import ctypes
import importlib.util
import os
from typing import NamedTuple

from ..backend import get_backend
from ..distributed import get_dist

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
    lib.fm4_overlap_num_chunks_per_segment.argtypes = [ctypes.c_int] * 5
    lib.fm4_overlap_num_chunks_per_segment.restype = ctypes.c_int

    # mask_ptr, num_heads, batch_stride, head_stride, k_ptr, v_ptr, compute_stream, fwd.
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


# _CommLayout of the current config generation; set by init_overlap.
_LAYOUT = None

# One NCCL UID per group; _UID_KEY holds the group object so its id cannot be recycled.
_UID = None
_UID_KEY = None


def is_available():
    """True if the bridge .so can be loaded with NCCL GIN support."""
    try:
        _load()
        return True
    except Exception:
        return False


# Mirrors the C++ limits (overlap_comm.cu, remote_get_kernel.cuh) so check_inputs can fail
# before any comm instead of terminating inside the noexcept bridge.
MAX_RANKS = 64
MAX_S_TOTAL = 2097152
SUPPORTED_CHUNKS_PER_SEGMENT = (1, 2, 4, 8)


def check_inputs(q, k, v, startend_row_indices, causal, nranks, be):
    """Reject an unsupported overlap call before any collective, so every rank raises."""
    if be.is_stream_capturing():
        # Init synchronizes the device, and the comm streams never rejoin a capture origin.
        raise ValueError("overlap does not support CUDA graph capture")
    if startend_row_indices is None:
        raise ValueError("overlap requires startend_row_indices (group-rank KV-chunk order)")
    if causal:
        # Causal column 1 is lt_end, but the comm-side chunk mask reads it as ut_end.
        raise ValueError(
            "overlap does not support causal=True; express causality in a non-causal "
            "2-column [lt_start, ut_end] mask"
        )
    num_vecs = startend_row_indices.shape[-1]
    if num_vecs != 2:
        # lt_end / ut_start are not passed to the bridge, so chunks would be wrongly skipped.
        raise ValueError(
            f"overlap requires a 2-column [lt_start, ut_end] mask, got {num_vecs} columns"
        )
    if startend_row_indices.shape[0] != k.shape[0]:
        # The SM100 forward indexes mask metadata by the real batch index; no broadcast.
        raise ValueError(
            f"overlap requires mask batch == batch size {k.shape[0]}, got "
            f"{startend_row_indices.shape[0]} (mask batch broadcasting is not supported)"
        )
    if not (q.dtype == k.dtype == v.dtype == be.bfloat16):
        raise ValueError(
            f"overlap requires bfloat16 q/k/v (the SRBuffer is bf16), got "
            f"{q.dtype}/{k.dtype}/{v.dtype}"
        )
    if v.shape[-1] != q.shape[-1]:
        raise ValueError(
            "overlap requires head_dim_v == head_dim (the SRBuffer V view is built from "
            f"K's head_dim), got head_dim={q.shape[-1]}, head_dim_v={v.shape[-1]}"
        )
    b, s_local, h, d = (int(x) for x in k.shape)
    if s_local <= 0 or s_local % 4 or s_local * h * d % 512:
        # check_s_local in overlap_comm.cu: 4-row chunk-mask loads, 512-element dK/dV reduce tasks
        raise ValueError(
            "overlap requires S_local % 4 == 0 and S_local * H_kv * head_dim % 512 == 0, got "
            f"K shape {tuple(k.shape)}"
        )
    if nranks > MAX_RANKS:
        raise ValueError(f"overlap supports at most {MAX_RANKS} ranks per group, got {nranks}")
    if nranks & (nranks - 1):
        raise ValueError(f"overlap requires a power-of-two group size, got {nranks}")
    if s_local * nranks > MAX_S_TOTAL:
        raise ValueError(
            f"overlap requires S_local * group size <= {MAX_S_TOTAL}, got "
            f"{s_local} * {nranks} = {s_local * nranks}"
        )
    num_chunks = int(_load().fm4_overlap_num_chunks_per_segment(b, s_local, h, d, int(nranks)))
    if num_chunks not in SUPPORTED_CHUNKS_PER_SEGMENT or nranks % num_chunks:
        raise ValueError(
            f"overlap backward picks {num_chunks} chunks per segment for K shape "
            f"{tuple(k.shape)} and group size {nranks}; supported: "
            f"{SUPPORTED_CHUNKS_PER_SEGMENT}, dividing the group size"
        )


def bootstrap_unique_id(rank, group, *, be, dist):
    """Generate and broadcast one NCCL UID for the current group generation.

    The id comes from the vendored bridge rather than from the framework, so
    overlap builds its own communicator and needs nothing from the framework
    beyond a byte broadcast.

    One overlap group per process. Switching groups tears the old communicator down
    collectively on the old group, so all of its ranks must switch at the same call.
    """
    global _UID, _UID_KEY
    world_size = dist.world_size(group)
    if _UID is not None and _UID_KEY[0] is group and _UID_KEY[1] == world_size:
        return _UID

    lib = _load()
    buf = (ctypes.c_uint8 * _UID_NBYTES)()
    if rank == 0 and lib.fm4_overlap_get_unique_id(ctypes.cast(buf, ctypes.c_char_p)) != 1:
        raise RuntimeError("fm4_overlap_get_unique_id failed")

    src_global = dist.global_ranks(group)[0]
    _UID = dist.broadcast_bytes(buf, src_global, group=group, be=be)
    _UID_KEY = (group, world_size)
    return _UID


def init_overlap(k, rank, nranks, uid_bytes, kv_shared=False):
    """Create or reconfigure the C++ singleton from the K shape and topology.
    kv_shared: K and V alias one storage, so only K is transported."""
    global _LAYOUT
    lib = _load()
    b, s_local, h, d = (int(x) for x in k.shape)
    # A failed init terminates inside the noexcept bridge; it never returns 0.
    generation = int(lib.fm4_overlap_init(b, s_local, h, d, int(rank), int(nranks), uid_bytes,
                                          int(kv_shared)))
    # The generation changes on every rebuild or reconfigure, so an unchanged one means
    # the cached layout is still valid.
    if _LAYOUT is None or _LAYOUT.generation != generation:
        _LAYOUT = _query_layout(lib, generation, b, h, d)


def ensure_initialized(k, group, kv_shared=False):
    """Bootstrap the unique id once, then forward shape/topology to the C++ singleton
    every step. init_singleton_instance reconfigures (and reallocs the SRBuffer) only
    when they actually change, so unconditional forwarding is cheap and correct."""
    be = get_backend(k)
    dist = get_dist(be.name)
    # A paddle Group exposes .rank / .world_size as attributes, a torch
    # ProcessGroup only through dist.get_rank(group); go through the adapter.
    rank = dist.rank(group)
    uid = bootstrap_unique_id(rank, group, be=be, dist=dist)
    init_overlap(k, rank, dist.world_size(group), uid, kv_shared=kv_shared)


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
    """Rank holding hierarchical logical position ``logical_pos`` (port of hier_map_chunk)."""
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
            # SR position j holds logical position nranks-1-j (fwd) or j (bwd).
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
    be = get_backend(mask)
    if not layout.hierarchical:
        # SR position j holds chunk (order[0] + j) % nranks: one roll by order[0] chunks.
        return be.roll(mask, -order[0] * s_local, axis=2)
    chunks = be.split(mask, nranks, axis=2)
    return be.concat([chunks[c] for c in order], axis=2)


def _data_ptr(t):
    return int(t.data_ptr())


def symmetric_buffer_bytes():
    """Total bytes of the registered symmetric (NCCL window) buffers; needs init first."""
    _layout()
    return int(_load().fm4_overlap_symmetric_buffer_bytes())


def _chunk_mask_pairs(startend_row_indices):
    """Return (mask, batch_stride, head_stride), the mask contiguous and 16-byte aligned
    as the chunk-mask kernel reads it in place."""
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
    write_ptr_addr: int  # device int*; invalid after reconfiguration
    local_kv_seqlen: int   # S_local


class BackwardAgArgs(NamedTuple):
    """Split-AG metadata of one config generation."""
    num_segments: int
    segment_seqlen: int
    work_done_addr: int  # device int* per-work done bitmap
    local_kv_seqlen: int   # S_local; the work grid restarts at every chunk
    comm_rpb: int
    shape: tuple  # (B, S_segment, H, D) of one segment
    segment_addrs: tuple  # per segment: (k_addr, v_addr, dk_send_addr, dv_send_addr)

    def kv_view(self, segment_idx):
        k_addr, v_addr, _, _ = self.segment_addrs[segment_idx]
        return SrKvView(k_addr=k_addr, v_addr=v_addr, shape=self.shape)

    def dkv_send_addrs(self, segment_idx):
        return self.segment_addrs[segment_idx][2:]


class _CommLayout(NamedTuple):
    """Communicator values of one config generation."""
    generation: int
    nranks: int
    s_local: int
    rank: int
    gpus_per_node: int
    hierarchical: bool
    bhsd: bool
    fwd: ForwardAgArgs
    bwd: BackwardAgArgs


def _query_layout(lib, generation, b, h, d):
    """Query every value of one config generation; b, h, d are the K shape it was built from."""
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
        local_kv_seqlen=s_local,
    )
    bwd = BackwardAgArgs(
        num_segments=num_segments,
        segment_seqlen=segment_seqlen,
        work_done_addr=work_done_addr,
        comm_rpb=comm_rpb,
        local_kv_seqlen=s_local,
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
    """Build the chunk-skip mask, stage local K/V and launch the (segment-0) AG. A temporary
    mask is safe to drop: its reader runs on compute_stream (stream-ordered allocator)."""
    mask, batch_stride, head_stride = _chunk_mask_pairs(startend_row_indices)
    _load().fm4_overlap_start_ag(
        _data_ptr(mask), int(mask.shape[1]), batch_stride, head_stride,
        _data_ptr(k), _data_ptr(v), int(compute_stream), 1 if fwd else 0,
    )


def forward_ag_args():
    """Forward SR view and frontier of the current config generation; launches nothing."""
    return _layout().fwd


def backward_ag_args():
    """Split-AG metadata of the current config generation; launches nothing."""
    return _layout().bwd


def start_forward_ag(k, v, startend_row_indices, compute_stream):
    """Launch forward sparse AG asynchronously for cumulative-frontier gating."""
    _start_ag(k, v, startend_row_indices, compute_stream, fwd=True)


def start_backward_ag(k, v, startend_row_indices, compute_stream):
    """Book the backward AG/RS round and launch AG segment 0; all ranks must then run
    every later segment."""
    _start_ag(k, v, startend_row_indices, compute_stream, fwd=False)


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
