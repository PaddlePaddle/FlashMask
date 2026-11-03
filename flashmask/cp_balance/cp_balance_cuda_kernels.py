# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Framework-neutral ctypes front-end over ``libflashmask_cpbalance.so``.

The cp_balance load-balancing kernels used to be Paddle custom ops; they are now
a raw pointer-ABI shared library (see csrc/cp_balance_bridge.cu) driven through
``flashmask.backend`` so paddle and torch tensors both work. The public function
names/signatures are unchanged, so cp_balance.py needs no edits.
"""

import ctypes
import glob
import os

import numpy as np

from ..backend import get_backend

_LIB = None


def _load_lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    here = os.path.dirname(os.path.abspath(__file__))
    matches = glob.glob(os.path.join(here, "libflashmask_cpbalance*.so"))
    if not matches:
        raise RuntimeError(
            "libflashmask_cpbalance*.so not found next to cp_balance; build FlashMask "
            "with the 'cpb' component (FLASHMASK_BUILD=all or =cpb)."
        )
    lib = ctypes.CDLL(matches[0], mode=ctypes.RTLD_LOCAL)

    u64 = ctypes.c_uint64
    i = ctypes.c_int
    lib.fm_cpb_scan_max_min.argtypes = [u64, i, i, i, u64, u64, u64]
    lib.fm_cpb_scan_max_min.restype = None
    lib.fm_cpb_reduce_workload.argtypes = [u64] * 8 + [u64, i, i, i, i, i, i, i, u64]
    lib.fm_cpb_reduce_workload.restype = None
    lib.fm_cpb_indices_to_chunks.argtypes = [u64, u64, u64, i, i, i, u64]
    lib.fm_cpb_indices_to_chunks.restype = None
    lib.fm_cpb_indices_rerank.argtypes = [u64, u64, u64, i, i, i, i, i, i, u64]
    lib.fm_cpb_indices_rerank.restype = None
    lib.fm_cpb_ipo_solve.argtypes = [u64, i, i, u64]
    lib.fm_cpb_ipo_solve.restype = ctypes.c_int

    _LIB = lib
    return _LIB


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def scanMaxMinChunkedKernel(input_tensor, Bc, B, H, S):
    be = get_backend(input_tensor)
    x = be.contiguous(input_tensor)
    dims = list(x.shape)
    total_batch = _numel(dims[:-1])
    num_sequences = int(dims[-1])
    # pad the number of blocks to a multiple of 4 (matches the kernel's own math)
    nblk = ((num_sequences + Bc - 1) // Bc + 3) & 0xFFFFFFFC
    maxo = be.empty([total_batch, nblk], "int32")
    mino = be.empty([total_batch, nblk], "int32")
    _load_lib().fm_cpb_scan_max_min(
        be.data_ptr(x), int(total_batch), num_sequences, int(Bc),
        be.data_ptr(maxo), be.data_ptr(mino), be.cuda_stream(),
    )
    return maxo, mino


def reduce_workload(start_row_maxmin_indice_list, B, H, Tr, Tc, Br, S):
    (
        LTStartMax, LTStartMin, LTEndMax, LTEndMin,
        UTStartMax, UTStartMin, UTEndMax, UTEndMin,
    ) = start_row_maxmin_indice_list

    be = get_backend(LTStartMax)
    # keep the contiguous views alive until after the launch
    conts = [be.contiguous(t) if t is not None else None for t in start_row_maxmin_indice_list]
    ptrs = [be.data_ptr(t) if t is not None else 0 for t in conts]

    # use the actual padded column stride from scanMaxMin (its last dim), not the
    # caller's unpadded Tc, so the flat index bh*Tc+tc is correct.
    Tc_stride = int(list(LTStartMax.shape)[1])
    workload = be.empty([B, H, Tr, 1], "int32")

    _load_lib().fm_cpb_reduce_workload(
        ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4], ptrs[5], ptrs[6], ptrs[7],
        be.data_ptr(workload), int(B * H), int(Tr), Tc_stride, int(S), int(Br),
        0,    # is_causal (matches the former paddle op call)
        128,  # m_block_size (matches the former paddle op call)
        be.cuda_stream(),
    )
    return workload


def indices_to_chunks_cuda(startend_row_indices, bucket_idx, chunksize=2048):
    be = get_backend(startend_row_indices)
    x = be.contiguous(startend_row_indices)
    buckets = be.contiguous(bucket_idx)
    out = be.empty_like(x, "int32")
    num_rows = _numel(list(x.shape))
    num_buckets = _numel(list(buckets.shape))
    _load_lib().fm_cpb_indices_to_chunks(
        be.data_ptr(x), be.data_ptr(buckets), be.data_ptr(out),
        int(num_rows), int(num_buckets), int(chunksize), be.cuda_stream(),
    )
    return out


def indices_rerank_cuda(startend_row_indices, indices, balance_chunk_size=2048):
    be = get_backend(startend_row_indices)
    x = be.contiguous(startend_row_indices)
    idx = be.contiguous(indices)
    B, H, S, D = [int(v) for v in list(x.shape)]
    num_chunks = (S + balance_chunk_size - 1) // balance_chunk_size
    out = be.empty([B, H, num_chunks * balance_chunk_size, D], "int32")
    _load_lib().fm_cpb_indices_rerank(
        be.data_ptr(x), be.data_ptr(idx), be.data_ptr(out),
        B, H, S, D, int(num_chunks), int(balance_chunk_size), be.cuda_stream(),
    )
    return out


def cp_balance_ipo_solve(weights_np, M):
    """CPU IPO/LPT solver. weights_np: 1-D int32 numpy, M: int. Returns
    (assign_matrix [M, N//M] int32, max_load)."""
    w = np.ascontiguousarray(weights_np, dtype=np.int32)
    N = int(w.size)
    K = N // M
    assign = np.empty((M, K), dtype=np.int32)
    max_load = _load_lib().fm_cpb_ipo_solve(
        w.ctypes.data, N, int(M), assign.ctypes.data
    )
    return assign, int(max_load)
