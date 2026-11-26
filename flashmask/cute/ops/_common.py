# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""Framework-agnostic helpers shared by the fwd / bwd / combine orchestration.

Nothing here imports paddle or torch. Whatever needs a framework takes a
`flashmask.backend` adapter, conventionally named ``be``.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from ...distributed import get_dist
from ..cute_dsl_utils import make_fake_tensor

# Import eagerly: a call-time import fails once a host renames the package in sys.modules.
# Wheels built without 'ovl' lack flashmask.overlap; only overlap calls report that.
try:
    from ...overlap import overlap_runtime as _overlap_runtime
    _overlap_import_error = None
except ImportError as exc:
    _overlap_runtime = None
    _overlap_import_error = exc


def _get_overlap_runtime():
    if _overlap_runtime is None:
        raise RuntimeError(
            "flashmask.overlap.overlap_runtime failed to import; the 'ovl' build "
            f"component may not have been built: {_overlap_import_error}"
        ) from _overlap_import_error
    return _overlap_runtime


def _dist_world_size(be, group):
    """World size of ``group``, via the collective adapter."""
    return get_dist(be.name).world_size(group)


def maybe_contiguous(x, be):
    return x.contiguous() if x is not None and not be.is_last_dim_contiguous(x) else x


# The head_dim / head_dim_v pairs above 256 that have kernels (SM90 and SM100), and the
# only shapes the kv_shared merge is implemented for.
_LARGE_HDIMS = ((512, 512), (576, 512))


def _same_storage(a, b, what="kv_shared"):
    """Whether two tensors start at the same device address.

    Used to detect the kv-shared call convention (the same buffer passed as both k and
    v). Either answers, or raises: a missing data_ptr() must NOT degrade to "not shared",
    because the kv_shared path returns a different gradient layout (dv all-zero, dk
    carrying dK + dV) and silently taking the other path would change the caller's
    gradients with no signal. The kv_shared condition evaluates this last, so a raise
    only reaches calls that match every other kv-shared criterion. ``what`` names the
    caller's path in the error message.
    """
    try:
        return a.data_ptr() == b.data_ptr()
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "cannot tell whether k and v alias: Tensor.data_ptr() is unavailable on "
            f"this build ({type(exc).__name__}: {exc}). This call matches the "
            f"{what} convention in every other respect, and kv_shared "
            "changes the gradient layout (dv comes back all-zero, dk carries "
            "dK + dV), so it will not be guessed. Pass distinct k / v buffers to use "
            "the plain path."
        ) from exc


def _kv_alias_candidate(k, v, be, what):
    """Whether v is k[..., :dv] over the same storage (the kv-shared call convention).

    The arch / head_dim gates stay with the caller. _same_storage is evaluated last
    because it raises rather than guess.
    """
    return (
        k.dtype == v.dtype
        and list(k.shape[:-1]) == list(v.shape[:-1])
        and v.shape[-1] <= k.shape[-1]
        and tuple(be.strides(k)) == tuple(be.strides(v))
        and _same_storage(k, v, what)
    )


def _get_compute_capability(be):
    return be.device_capability()[0]


def _is_valid_flash_dims(query, key, value, compute_capability):
    # SM90 and SM100 (the only archs with kernels) support the same head-dim combos,
    # so compute_capability does not narrow the set.
    q_headdim, k_headdim, v_headdim = query.shape[-1], key.shape[-1], value.shape[-1]
    return (
        (q_headdim <= 128 and k_headdim <= 128 and v_headdim <= 128)
        or (q_headdim, k_headdim, v_headdim)
        in ((192, 192, 128), (256, 256, 256), (512, 512, 512), (576, 576, 512))
    )


_SUPPORTED_HEAD_DIMS_DESC = (
    "q/k/v all <= 128, or 192/192/128, or 256/256/256, or 512/512/512, or 576/576/512"
)


def _check_cutedsl_supported(query, key, value, be):
    """Gate a call on the only two things that decide whether this package can
    serve it: the device compute capability and the (q, k, v) head-dim combo.

    No framework flag is consulted and there is no fallback. An unsupported call
    raises instead of quietly routing to another attention implementation, since
    that would change numerics, gradient layout and mask semantics under the
    caller with no signal. Both frontends call this, so paddle and torch report
    an unsupported configuration with the same text.
    """
    compute_capability = _get_compute_capability(be)
    assert compute_capability in (9, 10), (
        "flashmask kernels exist for SM90 (Hopper) and SM100 (Blackwell) only; "
        f"this device reports compute capability {compute_capability}.x. "
        "There is no fallback path."
    )
    assert _is_valid_flash_dims(query, key, value, compute_capability), (
        "unsupported head_dim combination "
        f"(q={query.shape[-1]}, k={key.shape[-1]}, v={value.shape[-1]}) "
        f"on SM{compute_capability}0. Supported: "
        + _SUPPORTED_HEAD_DIMS_DESC
    )


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    """Number of KV splits for the forward, always >= 1."""
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    # Clamp to >= 1: once total_mblocks > num_SMs (batched decode: large bs with
    # num_m_blocks == 1) the floor division is 0, which is not a split count.
    # total_mblocks == 0 (empty batch or seqlen_q) must not divide by zero.
    return max(1, min(num_SMs // max(total_mblocks, 1), max_splits, num_n_blocks))


# Opt-in shared memory per CTA on SM100.
_SM100_SMEM_PER_CTA = 232448


def sm100_split_kv_supported(head_dim, head_dim_v, elem_bytes):
    """Whether the dense SM100 forward launches with num_splits > 1 at this head dim.

    Large head_dim_v is rejected by the kernel and head_dim above the big-d threshold has
    no split. Below that the limit is SMEM: split writes fp32 O partials, so the sO
    epilogue buffer doubles. It shares sQ's space under overlap_sO_sQ (see
    FlashAttentionForwardSm100), else it adds to sQ. The estimate mirrors the kernel's
    Q / KV / O buffers plus a fixed allowance for barriers and scratch, so it follows
    the kernel's own stage and tile choices.
    """
    d = -(-head_dim // 16) * 16
    dv = -(-head_dim_v // 16) * 16
    if dv >= 192 or d > 256:
        return False
    q_bytes = 2 * 128 * d * elem_bytes
    o_bytes = 2 * 128 * dv * 4
    if d == 192 and dv == 128:
        kv_bytes = 128 * (2 * d + dv) * elem_bytes  # uneven_kv_smem: [large, small, large]
    else:
        kv_bytes = 3 * 128 * max(d, dv) * elem_bytes
    overlap_o_q = dv >= 128 or (d == 192 and dv >= 64)
    qo_bytes = max(q_bytes, o_bytes) if overlap_o_q else q_bytes + o_bytes
    return qo_bytes + kv_bytes + 3072 <= _SM100_SMEM_PER_CTA

@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False

def _tile_size_bwd_sm90(
    head_dim, head_dim_v, causal, local, sparse_block_size_q=None,
    flashmask=False, deterministic=False, kv_shared=False,
):
    """Return BwdConfig for SM90.

    Configs based on C++ FA3 hopper/flash_bwd_launch_template.h,
    benchmarked on H100 SXM.
    """
    if head_dim <= 64:
        # C++ FA3: 128, 128, 64, ..., 2, 2, true, false, false, 2, 1, 2, 2
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        # C++ FA3: 64, 128, 96, dQ_swapAB=False
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        # C++ FA3: causal/local: 64, 128; non-causal: 80, 128 with dQ_swapAB
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        # flashmask stages per-n_block startend_row_indices in extra smem; the
        # non-causal m=80 tile has no headroom for it, so fall back to m=64.
        if flashmask:
            m_block_size = 64
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    elif head_dim > 256:
        # hdim > 256 (FlashAttentionBackwardSm90.dQ_chunked): the d256 flag set. Q, K, dO
        # and the dK/dV accumulators together take most of the smem, so every pipeline is
        # single-stage, and the dQ accumulator is chunked and atomicAdd'ed to gmem
        # (non-deterministic backward).
        assert (head_dim, head_dim_v) in _LARGE_HDIMS, (
            f"SM90 bwd head_dim > 256 supports only 512/512 and 576/512 "
            f"(got {head_dim}/{head_dim_v})"
        )
        # The 512 layout puts head_dim on the M atoms of dK^T / dQ^T and splits them
        # over the two WGs, which needs head_dim to be a multiple of 128. 576 is not, so
        # it splits the WGs along tile_n for dK/dV and along tile_m for dQ instead.
        split_hdim_atoms = head_dim % 128 == 0
        # kv_shared merges dV into dK's accumulator and drops sV, and keeping dv at
        # head_dim_v leaves dO at its unpadded width, which buys a 64-row KV tile.
        # n=48 is out: it gives each WG a dK^T/dV^T width that is not a multiple of 16,
        # which computes a wrong dK and breaks the GQA dK accum postprocess.
        n_block_size = 64 if kv_shared else 32
        return BwdConfig(
            m_block_size=64, n_block_size=n_block_size,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=True,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=1 if split_hdim_atoms else 2,
            AtomLayoutMdQ=1 if split_hdim_atoms else 2,
        )
    else:
        # hdim 256: mirror C++ FA3 run_mha_bwd_hdim256 (Arch>=90); dKV/dQ MMAs
        # use swapAB. The dQ-accumulation path drives the smem budget and must
        # match flash_bwd_sm90's dQacc_use_TMA (== deterministic for d256):
        #   - non-deterministic: dQ is atomicAdd'd straight to gmem, so the dQ smem
        #     staging buffer is dropped. That headroom buys a 2-stage Q pipeline and
        #     the larger dense N=80 tile (fewer KV blocks; flashmask keeps N=64 to
        #     match its block-list tiling).
        #   - deterministic: ordered accumulation forces the TMA-staging path, which
        #     re-adds that buffer. 2-stage Q / N=80 would then overflow Hopper's smem
        #     limit (CUDA_ERROR_INVALID_VALUE at launch), so use a 1-stage Q pipeline
        #     and N=64.
        if deterministic:
            n_block_size, num_stages_Q = 64, 1
        else:
            n_block_size, num_stages_Q = (64 if flashmask else 80), 2
        return BwdConfig(
            m_block_size=64, n_block_size=n_block_size,
            num_stages_Q=num_stages_Q, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=True,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )


def _make_fake_bwd_tensors(dtype, has_gqa, deterministic=False, cluster_size=1):
    """FA4-style fake bwd tensors with all dims as cute.sym_int and stride
    divisibility hints (so 128-bit alignment is guaranteed at compile time)."""
    sym = cute.sym_int
    div = 128 // dtype.width  # 8 for bf16/fp16
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    mQ = make_fake_tensor(dtype, (b, seqlen_q, h_q, d), divisibility=div)
    mO = make_fake_tensor(dtype, (b, seqlen_q, h_q, d_v), divisibility=div)
    mdO = make_fake_tensor(dtype, (b, seqlen_q, h_q, d_v), divisibility=div)
    mK = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d), divisibility=div)
    mV = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d_v), divisibility=div)
    mdQ = make_fake_tensor(dtype, (b, seqlen_q, h_q, d), divisibility=div)
    mdK = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d), divisibility=div)
    mdV = make_fake_tensor(dtype, (b, seqlen_k, h_kv, d_v), divisibility=div)
    mLSE = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q), divisibility=1)
    mLSElog2 = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mPdPsum = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
    mdQaccum = make_fake_tensor(cutlass.Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        mdKaccum = make_fake_tensor(cutlass.Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
        mdVaccum = make_fake_tensor(cutlass.Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
    if not deterministic:
        mdQ_semaphore = None
        mdK_semaphore, mdV_semaphore = None, None
    else:
        num_m_blocks = sym()
        mdQ_semaphore = make_fake_tensor(
            cutlass.Int32, (b, h_q, num_m_blocks, cluster_size), divisibility=1
        )
        if not has_gqa:
            mdK_semaphore, mdV_semaphore = None, None
        else:
            num_n_blocks = sym()
            mdK_semaphore = make_fake_tensor(
                cutlass.Int32, (b, h_kv, num_n_blocks, 2), divisibility=1
            )
            mdV_semaphore = make_fake_tensor(
                cutlass.Int32, (b, h_kv, num_n_blocks, 2), divisibility=1
            )
    return (
        mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum,
        mdQaccum, mdKaccum, mdVaccum, mdQ_semaphore, mdK_semaphore, mdV_semaphore,
    )

