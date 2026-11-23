# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FlashMask functional test (torch).

Runs ``flashmask_attention`` against an eager fp32 reference and allows it the
error a naive bf16 implementation makes -- concretely
``|out - ref| <= 2 * |naive_bf16 - ref| + atol`` -- so the bound scales with the
configuration instead of being a hand-picked constant. Covers forward, dq/dk/dv,
and dsink across the mask library, split/shared KV, and head dims up to 576.
The ``test_decode_swapab*`` / ``test_flashmask_attention_decode_dispatch*`` tests
cover the SM100 SwapAB decode branch (seqlen_q = 1, forward only).

FlashMask only ships the FA-v4 CuTe DSL kernels (SM90 / SM100), so there is no
flash-attention version axis here.
"""

import contextlib
import math
from functools import partial
from unittest import mock

import pytest
import torch

import flashmask.cute.ops.fwd as fwd_ops
from flashmask.cute.flash_fwd_decode_sm100 import swapab_occupancy_fits, swapab_tmem_cols
from flashmask.cute.ops.fwd import _flash_attn_fwd
from flashmask.frontend.torch import flashmask_attention

from tests.torch.attention_ref import attention_ref
from tests.torch.generate_startend_row_indices import (
    startend_row_indices_to_attn_bias,
    generate_none_mask,
    generate_sliding_window_mask,
    generate_causal_document_mask,
    generate_document_mask,
    generate_share_question_mask,
    generate_global_sliding_window_mask,
    generate_causal_blockwise_mask,
    generate_prefix_lm_document_mask,
    generate_prefix_lm_causal_mask,
    generate_qk_sparse_mask,
    generate_random_eviction_mask,
    generate_empty_mask,
)

# KV shared: k and v are ONE buffer (the MLA convention, v = k[..., :dv]). Only the
# big-headdim backward merges dK/dV, and only for these (d, dv) pairs -- anywhere
# else "shared" would just re-run the split path with a different allocation.
KV_SHARED_D_DV = ((512, 512), (576, 512))


def kv_shared_detected(k, v):
    """Whether the backward will take its kv-shared path for these two tensors.

    Mirrors the predicate in the frontend minus the (d, dv) gate, so a test can
    ASSERT which path it exercised: if torch ever materialises ``k[..., :dv]`` as
    a copy, the merge would silently not run and the test would pass while
    covering nothing.
    """
    try:
        same_storage = k.data_ptr() == v.data_ptr()
    except (AttributeError, RuntimeError):
        return False
    return (
        same_storage
        and k.dtype == v.dtype
        and list(k.shape[:-1]) == list(v.shape[:-1])
        and v.shape[-1] <= k.shape[-1]
        and tuple(k.stride()) == tuple(v.stride())
    )

# batch_size, seqlen_q, seqlen_k, nheads, nheads_kv
shape_cases = (
    [
        (2840, 32, 32, 16, 4),
        (1, 300, 300, 16, 16),
        (2, 16383, 16385, 4, 1),
    ]
)

_shape_cases_before = len(shape_cases)
shape_cases = list(dict.fromkeys(shape_cases))  # 保序去重
print(f"{'='*60}")
print(f"[test_flashmask] Shape Cases Summary:")
print(f"  - Original Count: {_shape_cases_before}")
print(f"  - Unique Count:   {len(shape_cases)}")
print(f"  - Removed:        {_shape_cases_before - len(shape_cases)}")
print(f"{'='*60}")

d_dv_cases = [
    (80, 80),
    (128, 128),
    (192, 128),
    (256, 256),
    (512, 512),
    (576, 512),
]

# sink dtype as a string tag (resolved to a torch dtype in the test): pytest
# cannot stringify a torch dtype, so a raw dtype shows up in the id as an opaque
# "sink_dtype4" instead of "fp16".
_SINK_TAGS = ["bf16", "fp16", "fp32"]
_SINK_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# Generate all combinations for second param
def generate_shapes():
    # sink dtype is round-robined over the generated cases: exactly uniform (the case
    # count is a multiple of len(_SINK_TAGS)), deterministic, adds no case, and shows
    # up in the pytest id so a failure tells you the dtype without reading the log.
    for i, (batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices) in (
        enumerate(_generate_shape_params())
    ):
        yield (
            batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices,
            _SINK_TAGS[i % len(_SINK_TAGS)],
        )


def _generate_shape_params():
    for batch_size, seqlen_q, seqlen_k, nheads, nheads_kv in shape_cases:
        if nheads_kv == 1:
          nheads_startend_row_indices_values = [1]
        else:
          nheads_startend_row_indices_values = [1, nheads_kv]
        for nheads_startend_row_indices in nheads_startend_row_indices_values:
            yield (
                batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("kv_mode", ["split", "shared"])
@pytest.mark.parametrize(
    "d, dv",
    d_dv_cases,
    ids=[f"d{c[0]}-dv{c[1]}" for c in d_dv_cases]
)
@pytest.mark.parametrize(
    "batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, nheads_startend_row_indices, sink_dtype",
    list(generate_shapes()),
)
@pytest.mark.parametrize(
    "gen_startend_row_indices",
    [
        partial(generate_none_mask, causal=False), # full
        partial(generate_none_mask, causal=True), # causal
        partial(generate_sliding_window_mask), # sliding window
        partial(generate_causal_document_mask), # causal document mask
        partial(generate_document_mask), # document mask
        partial(generate_share_question_mask), # share question mask
        partial(generate_global_sliding_window_mask), # global sliding window
        partial(generate_causal_blockwise_mask), # causal blockwise mask
        partial(generate_prefix_lm_document_mask), # prefix lm document mask
        partial(generate_prefix_lm_causal_mask), # prefix lm causal mask
        partial(generate_qk_sparse_mask), # qk-sparse mask
        partial(generate_random_eviction_mask), # random eviction mask
        partial(generate_empty_mask),
    ],
)
def test_flashmask(
    batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, d, dv, nheads_startend_row_indices, sink_dtype, kv_mode, dtype, gen_startend_row_indices, softcap=0.0
):
    torch.manual_seed(2026)
    assert nheads % nheads_kv == 0
    # tag -> torch dtype (see _SINK_TAGS for why it arrives as a string)
    sink_dtype = _SINK_DTYPE[sink_dtype]

    # SM90 kernels only support head dims up to 256.
    if torch.cuda.get_device_capability()[0] == 9 and max(d, dv) > 256:
        pytest.skip(f"Skipping d{d}-dv{dv} on SM90: head dim > 256 not supported")

    # Only the big-headdim backward merges dK/dV (see KV_SHARED_D_DV); anywhere
    # else "shared" would just re-run the split path with a different allocation.
    kv_shared = kv_mode == "shared"
    if kv_shared and (d, dv) not in KV_SHARED_D_DV:
        pytest.skip(f"Skipping kv shared for d{d}-dv{dv}: merge not implemented")

    startend_row_indices, causal = gen_startend_row_indices(batch_size, seqlen_q, seqlen_k, nheads_startend_row_indices)

    # d192/dv192 has no causal kernel for seqlen_q != seqlen_k (the cutedsl gate
    # allows 192/192/128, not 192/192/192).
    if (d == 192 and dv == 192) and seqlen_q != seqlen_k and causal:
        pytest.skip("Skipping d192-dv192 causal when seqlen_q != seqlen_k")

    use_sink = not (d == 192 and dv == 192)

    q_ref = torch.randn([batch_size, seqlen_q, nheads, d], dtype=dtype, device="cuda")
    k_ref = torch.randn([batch_size, seqlen_k, nheads_kv, d], dtype=dtype, device="cuda")

    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)

    def make_qkv(q_src, k_src, v_src):
        """One (q, k, v) triplet. In shared mode v is a stride-1 view onto k.

        The view is taken AFTER requires_grad is set, so k is the leaf and
        autograd folds dV back into k.grad the same way the kernel's merged
        accumulator does.
        """
        q = q_src.detach().clone()
        k = k_src.detach().clone()
        q.requires_grad_(True)
        k.requires_grad_(True)
        if kv_shared:
            v = k[..., :dv]
        else:
            v = v_src.detach().clone()
            v.requires_grad_(True)
        return q, k, v

    if kv_shared:
        v_ref = k_ref[..., :dv]
        assert kv_shared_detected(k_ref, v_ref), (
            "v is not a stride-1 view onto k -- torch materialised the slice, "
            "so the kernel's kv-shared path would not be exercised"
        )
    else:
        v_ref = torch.randn([batch_size, seqlen_k, nheads_kv, dv], dtype=dtype, device="cuda")
        v_ref.requires_grad_(True)

    q_bf16, k_bf16, v_bf16 = make_qkv(q_ref, k_ref, v_ref)
    q, k, v = make_qkv(q_ref, k_ref, v_ref)

    if kv_shared:
        assert kv_shared_detected(k, v)

    attn_bias = startend_row_indices_to_attn_bias(startend_row_indices, seqlen_q, nheads, dtype, causal)

    if use_sink:
        # sink_dtype comes from the shape parametrize (round-robined in generate_shapes),
        # so it is independent of the q/k/v dtype and costs no extra case.
        print(f"learnable_sink dtype: {sink_dtype}")
        sink_ref = torch.randn([nheads], dtype=torch.float32, device="cuda").to(sink_dtype)
        sink_bf16 = sink_ref.detach().clone()
        sink = sink_ref.detach().clone()

        sink_bf16.requires_grad_(True)
        sink_ref.requires_grad_(True)
        sink.requires_grad_(True)
    else:
        sink_ref = None
        sink_bf16 = None
        sink = None

    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        causal=causal,
        attn_bias=attn_bias,
        learnable_sink=sink_ref,
    )

    out_bf16, attn_bf16 = attention_ref(
        q_bf16,
        k_bf16,
        v_bf16,
        causal=causal,
        attn_bias=attn_bias,
        upcast=False,
        reorder_ops=True,
        learnable_sink=sink_bf16,
    )

    # # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert softcap == 0.0
    rtol = 2 if softcap == 0.0 else 3

    print(f"Torch naive bf16 Output max diff: {(out_bf16 - out_ref).abs().max().item()}")
    print(f"Torch naive bf16 Output mean diff: {(out_bf16 - out_ref).abs().mean().item()}")

    out, lse = flashmask_attention(
        q,
        k,
        v,
        startend_row_indices,
        causal=causal,
        return_lse=True,
        learnable_sink=sink,
    )
    print(f"flashmask Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"flashmask Output mean diff: {(out - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.

    assert (out - out_ref).abs().max().item() <= rtol * (out_bf16 - out_ref).abs().max().item() + fwd_atol

    g = torch.randn(out.shape, dtype=out.dtype, device="cuda")
    out.backward(g)
    out_ref.backward(g)
    out_bf16.backward(g)

    print(f"flashmask dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"flashmask dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"flashmask dQ mean diff: {(q.grad - q_ref.grad).abs().mean().item()}")
    print(f"flashmask dK mean diff: {(k.grad - k_ref.grad).abs().mean().item()}")

    print(f"Torch naive bf16 dQ max diff: {(q_bf16.grad - q_ref.grad).abs().max().item()}")
    print(f"Torch naive bf16 dK max diff: {(k_bf16.grad - k_ref.grad).abs().max().item()}")
    print(f"Torch naive bf16 dQ mean diff: {(q_bf16.grad - q_ref.grad).abs().mean().item()}")
    print(f"Torch naive bf16 dK mean diff: {(k_bf16.grad - k_ref.grad).abs().mean().item()}")

    if not kv_shared:
        print(f"flashmask dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
        print(f"flashmask dV mean diff: {(v.grad - v_ref.grad).abs().mean().item()}")
        print(f"Torch naive bf16 dV max diff: {(v_bf16.grad - v_ref.grad).abs().max().item()}")
        print(f"Torch naive bf16 dV mean diff: {(v_bf16.grad - v_ref.grad).abs().mean().item()}")

    dq_atol = 2 * (q_ref.grad + 0.3 - 0.3 - q_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (q.grad - q_ref.grad).abs().max().item() <= rtol * (q_bf16.grad - q_ref.grad).abs().max().item() + dq_atol
    # In shared mode k is the only leaf, so k.grad already carries dK + dV: the
    # kernel merges them into one accumulator and returns an all-zero dv, and
    # autograd's slice-grad scatters that zero dv back into the same buffer. The
    # dK check below is therefore the dKV check; v.grad does not exist.
    dk_atol = 2 * (k_ref.grad + 0.3 - 0.3 - k_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
    assert (k.grad - k_ref.grad).abs().max().item() <= rtol * (k_bf16.grad - k_ref.grad).abs().max().item() + dk_atol
    if not kv_shared:
        dv_atol = 2 * (v_ref.grad + 0.3 - 0.3 - v_ref.grad).abs().max().item() + (0 if softcap == 0 else 3e-4)
        assert (v.grad - v_ref.grad).abs().max().item() <= rtol * (v_bf16.grad - v_ref.grad).abs().max().item() + dv_atol

    if use_sink:
        assert sink.grad.dtype == sink_dtype, (
            f"dsink dtype {sink.grad.dtype} != sink dtype {sink_dtype}"
        )
        assert bool(torch.isfinite(sink.grad.to(torch.float32)).all().item()), (
            f"dsink has inf/nan (sink dtype={sink_dtype}): {sink.grad}"
        )
        print(f"flashmask dSink max diff: {(sink.grad - sink_ref.grad).abs().max().item()}")
        print(f"flashmask dSink mean diff: {(sink.grad - sink_ref.grad).abs().mean().item()}")
        print(f"Torch naive bf16 dSink max diff: {(sink_bf16.grad - sink_ref.grad).abs().max().item()}")
        print(f"Torch naive bf16 dSink mean diff: {(sink_bf16.grad - sink_ref.grad).abs().mean().item()}")

        delta = (out_ref * g).sum(-1)                                      # (b, sq, h)
        delta = torch.permute(delta, [0, 2, 1])                            # (b, h, sq)
        p_sink = 1.0 - attn_ref.sum(-1)

        err_scale = (p_sink * delta.abs()).sum(dim=[0, 2]).to(torch.float32)   # (h,)

        # Compare in fp32: sink may be fp16/bf16/fp32, err_scale comes from bf16 attn_ref
        dsink_diff = (
            sink.grad.to(torch.float32) - sink_ref.grad.to(torch.float32)
        ).abs()
        dsink_tol = 1e-2 + rtol * 2**-8 * err_scale
        assert bool((dsink_diff <= dsink_tol).all().item()), (
            f"dsink mismatch (sink dtype={sink_dtype}): "
            f"max_diff={dsink_diff.max().item():.6f}, "
            f"max_tol={dsink_tol.max().item():.6f}, "
            f"err_scale_max={err_scale.max().item():.4f}"
        )

    torch.cuda.empty_cache()


@pytest.mark.parametrize("seqlen", [128, 129])
@pytest.mark.parametrize("kv_mode", ["split", "shared"])
@pytest.mark.parametrize("d, dv", KV_SHARED_D_DV, ids=[f"d{d}-dv{dv}" for d, dv in KV_SHARED_D_DV])
def test_flashmask_bigd_permuted_bshd(d, dv, kv_mode, seqlen):
    """Dense-permuted BSHD must match contiguous forward and gradients bitwise."""
    if torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("head dim > 256 not supported on SM90")
    batch, nheads, nheads_kv, dtype = 1, 4, 2, torch.bfloat16
    torch.manual_seed(2026)
    q_src = torch.randn([batch, seqlen, nheads, d], dtype=dtype, device="cuda")
    k_src = torch.randn([batch, seqlen, nheads_kv, d], dtype=dtype, device="cuda")
    v_src = torch.randn([batch, seqlen, nheads_kv, dv], dtype=dtype, device="cuda")
    g_src = torch.randn([batch, seqlen, nheads, dv], dtype=dtype, device="cuda")

    def run(permuted):
        def layout(x):
            if not permuted:
                return x.clone()
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
            assert x.stride(-1) == 1 and (x.shape[2] == 1 or x.stride(2) != x.shape[-1])
            return x

        q = layout(q_src).requires_grad_(True)
        k = layout(k_src).requires_grad_(True)
        if kv_mode == "shared":  # k.grad carries the merged dK + dV.
            v = k[..., :dv]
            assert kv_shared_detected(k, v)
        else:
            v = layout(v_src).requires_grad_(True)
        out = flashmask_attention(q, k, v, causal=True)
        out.backward(layout(g_src))
        return [out, q.grad, k.grad] + ([] if kv_mode == "shared" else [v.grad])

    prev = (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        contig, permuted = run(False), run(True)
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
    for name, a, b in zip(("out", "dq", "dk", "dv"), contig, permuted):
        assert torch.equal(a, b), f"{name}: max diff {(a.float() - b.float()).abs().max().item()}"
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# SwapAB decode (SM100 / SM103): seqlen_q = 1 with inputs that do not require grad, optional
# FlashMask column mask.
#
# The tests call ``flashmask_attention`` and check that ``_flash_attn_fwd`` took its SwapAB
# branch; split count and occupancy are forced by patching their heuristics. Calls the frontend
# does not accept (a mask head per query head under GQA, ``out=``) go to ``_flash_attn_fwd``.
# Most tests use head_dim 128; the head_dim tests cover the other (head_dim, head_dim_v) pairs.
# The output must stay within ``2 * |naive_lowp - ref| + atol`` of an fp32 reference, where the
# naive low-precision run casts its softmax probabilities to the input dtype as the kernel does.
# ---------------------------------------------------------------------------


def _is_sm100_family():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) in ((10, 0), (10, 3))


_needs_swapab = pytest.mark.skipif(not _is_sm100_family(), reason="SwapAB decode needs SM100/SM103")


def _reference(q, k, v, scale, upcast=True):
    """(out (b, 1, h, d), lse (b, h, 1)); upcast=False mimics a low-precision P."""
    qhpk = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(qhpk, dim=2)
    v = v.repeat_interleave(qhpk, dim=2)
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    lse = torch.logsumexp(scores, dim=-1)  # (b, h, 1)
    probs = torch.softmax(scores, dim=-1)
    if not upcast:
        probs = probs.to(q.dtype).float()
    out = torch.einsum("bhqk,bkhd->bqhd", probs, vf)
    return out, lse


@contextlib.contextmanager
def _expect_swapab(expected=True):
    """Assert whether the _flash_attn_fwd calls inside took the SwapAB branch."""
    taken = []
    real = fwd_ops.swapab_decode_supported

    def spy(*args):
        taken.append(real(*args))
        return taken[-1]

    with mock.patch.object(fwd_ops, "swapab_decode_supported", spy):
        yield
    torch.cuda.synchronize()
    assert any(taken) == expected, f"SwapAB branch taken: {any(taken)}, expected {expected}"


def _attn(q, k, v, mask=None, causal=False, softmax_scale=None, num_splits=0, occupancy=0,
          return_lse=True, out=None):
    """One decode call, asserting it ran SwapAB. num_splits / occupancy > 0 replace the
    kernel's heuristics. Goes through flashmask_attention unless the call needs out= or a mask
    head count other than 1 / num_heads_kv. Returns (out, lse); lse is None unless return_lse."""
    direct = out is not None or (mask is not None and mask.shape[1] not in (1, k.shape[2]))
    with contextlib.ExitStack() as stack:
        stack.enter_context(_expect_swapab())
        if num_splits:
            stack.enter_context(
                mock.patch.object(fwd_ops, "swapab_num_splits", return_value=num_splits)
            )
        if occupancy:
            stack.enter_context(
                mock.patch.object(fwd_ops, "swapab_occupancy", return_value=occupancy)
            )
        if direct:
            res = _flash_attn_fwd(
                q, k, v, softmax_scale=softmax_scale, causal=causal, return_lse=return_lse,
                out=out, startend_row_indices=mask,
            )
        else:
            res = flashmask_attention(
                q, k, v, mask, causal=causal, softmax_scale=softmax_scale, return_lse=return_lse
            )
    if direct:
        return res
    return res if return_lse else (res, None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [
        (1, 1, 8, 1),
        (1, 100, 8, 8),
        (2, 256, 16, 2),
        (3, 257, 32, 8),
        (4, 1000, 28, 4),
        (2, 4096, 32, 1),
        (1, 8192, 64, 8),
        (8, 2049, 8, 2),
        (16, 512, 40, 8),
    ],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
@pytest.mark.parametrize("return_lse", [False, True])
@_needs_swapab
def test_decode_swapab(dtype, batch, seqlen_k, num_heads, num_heads_kv, num_splits, return_lse):
    torch.manual_seed(0)
    head_dim = 128
    device = "cuda"
    q = torch.randn(batch, 1, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)

    out, lse = _attn(q, k, v, num_splits=num_splits, return_lse=return_lse)
    out_ref, lse_ref = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)

    assert out.shape == q.shape and out.dtype == dtype
    assert torch.isfinite(out.float()).all()
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    if return_lse:
        assert lse.shape == (batch, num_heads, 1)
        torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    else:
        assert lse is None


@_needs_swapab
def test_decode_swapab_strided_kv():
    """K/V as a slice of a larger cache: the kernel reads the real strides."""
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 128
    batch, seqlen_k, cap, num_heads, num_heads_kv = 2, 700, 1024, 16, 4
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k_cache = torch.randn(batch, cap, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v_cache = torch.randn(batch, cap, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    k, v = k_cache[:, :seqlen_k], v_cache[:, :seqlen_k]
    out, _ = _attn(q, k, v)
    out_ref, _ = _reference(q, k, v, 1.0 / math.sqrt(head_dim))
    out_naive, _ = _reference(q, k, v, 1.0 / math.sqrt(head_dim), upcast=False)
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"


def _col_mask_keep(mask, causal):
    """(b, h_m, s_k) True where the decode query (mask row 0) may attend the key."""
    row = 0
    num_vecs = mask.shape[3]
    lts = mask[..., 0]
    masked = row >= lts
    if num_vecs == 2:
        bound = mask[..., 1]
        masked = (masked & (row < bound)) if causal else (masked | (row < bound))
    elif num_vecs == 4:
        lte, uts, ute = mask[..., 1], mask[..., 2], mask[..., 3]
        masked = (masked & (row < lte)) | ((row >= uts) & (row < ute))
    return ~masked


def _masked_reference(q, k, v, scale, keep, upcast=True):
    """keep: (b, h, s_k). Fully masked heads give O = 0 and LSE = -inf."""
    qhpk = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(qhpk, dim=2)
    v = v.repeat_interleave(qhpk, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
    scores = scores.masked_fill(~keep[:, :, None, :], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    if not upcast:
        probs = probs.to(q.dtype).float()
    out = torch.einsum("bhqk,bkhd->bqhd", probs, v.float())
    return out, lse


def _random_mask(batch, mask_heads, seqlen_k, num_vecs, device, causal=True, high=2):
    """Random bounds in [0, high] (default 2, so row 0 sees part of the keys), each pair
    ordered. Non-causal num_vecs 2 masks [LTS, inf) and [0, UTE), so keys are visible only
    with UTE <= LTS: that pair is sorted descending."""
    m = torch.randint(0, high + 1, (batch, mask_heads, seqlen_k, num_vecs), device=device)
    if num_vecs >= 2:
        m[..., :2] = m[..., :2].sort(dim=-1, descending=num_vecs == 2 and not causal).values
    if num_vecs == 4:
        m[..., 2:] = m[..., 2:].sort(dim=-1).values
    return m.to(torch.int32).contiguous()


def _run_masked(q, k, v, mask, causal, num_splits=0, occupancy=0):
    """Check one masked decode call against the reference; the query is mask row 0."""
    num_heads = q.shape[2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    out, lse = _attn(q, k, v, mask, causal, num_splits=num_splits, occupancy=occupancy)
    keep = _col_mask_keep(mask, causal)
    keep = keep.repeat_interleave(num_heads // mask.shape[1], dim=1)
    out_ref, lse_ref = _masked_reference(q, k, v, scale, keep)
    out_naive, _ = _masked_reference(q, k, v, scale, keep, upcast=False)
    assert torch.isfinite(out.float()).all()
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(q.dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    return keep


# Layouts flashmask_attention accepts: num_vecs 1 / 2 with causal, 2 / 4 without. causal only
# changes the kernel for num_vecs 2, so (1, False) would add nothing.
_MASK_KINDS = [(1, True), (2, True), (2, False), (4, False)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [
        (2, 300, 16, 2),
        (4, 129, 28, 4),
        (3, 1000, 8, 8),
        (1, 4096, 32, 8),
    ],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
@_needs_swapab
def test_decode_swapab_mask(
    dtype, num_vecs, causal, mask_heads, batch, seqlen_k, num_heads, num_heads_kv, num_splits
):
    torch.manual_seed(0)
    head_dim = 128
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    mask = _random_mask(batch, h_m, seqlen_k, num_vecs, "cuda", causal)
    _run_masked(q, k, v, mask, causal, num_splits)


@pytest.mark.parametrize("mask_heads", ["one", "all"])
@pytest.mark.parametrize("num_splits", [1, 4])
@_needs_swapab
def test_decode_swapab_mask_fully_masked(mask_heads, num_splits):
    """Fully masked heads and fully masked splits: O = 0, LSE = -inf, no NaN."""
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 128
    batch, seqlen_k, num_heads, num_heads_kv = 3, 2048, 16, 2
    h_m = 1 if mask_heads == "one" else num_heads
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    # num_vecs 1: a key is masked iff LTS <= 0 (the query is row 0).
    mask = torch.full((batch, h_m, seqlen_k, 1), seqlen_k, device="cuda", dtype=torch.int32)
    mask[0] = 0  # batch 0: every key masked
    mask[1, :, : seqlen_k * 3 // 4] = 0  # batch 1: the first three splits see only masked keys
    mask[2, 0] = 0  # batch 2: mask head 0 fully masked
    mask[2, :, 7] = 0
    keep = _run_masked(q, k, v, mask, True, num_splits)
    assert not keep[0].any() and keep[1].any()


def _block_sparse_mask(batch, mask_heads, seqlen_k, num_vecs, causal, device):
    """Mask with whole KV units hidden, visible or mixed, so block skipping matters.

    Hidden keys alternate between the two masking intervals where the layout has two.
    """
    s = seqlen_k
    nblk = -(-s // 128)
    state = torch.randint(0, 3, (batch, mask_heads, nblk), device=device)  # 0 hide 1 show 2 mix
    state = state.repeat_interleave(128, dim=-1)[..., :s]
    hidden = (state == 0) | ((state == 2) & (torch.rand(state.shape, device=device) < 0.5))
    odd = (torch.arange(s, device=device) % 2 == 1).expand_as(hidden)
    m = torch.full((batch, mask_heads, s, num_vecs), s, dtype=torch.int32, device=device)
    if num_vecs == 1:
        m[..., 0][hidden] = 0
    elif num_vecs == 2 and causal:
        m[..., 0][hidden] = 0  # [LTS, LTE) = [0, s)
    elif num_vecs == 2:
        m[..., 1] = 0
        m[..., 0][hidden & odd] = 0  # row >= LTS
        m[..., 1][hidden & ~odd] = s  # row < UTE
    else:
        m[..., 0][hidden & odd] = 0  # [LTS, LTE) = [0, s)
        m[..., 2][hidden & ~odd] = 0  # [UTS, UTE) = [0, s)
    return m.contiguous()


@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv", [(3, 4096, 32, 8), (2, 1000, 16, 2)]
)
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@_needs_swapab
def test_decode_swapab_mask_block_sparse(
    num_vecs, causal, mask_heads, batch, seqlen_k, num_heads, num_heads_kv, num_splits
):
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 128
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    _run_masked(q, k, v, mask, causal, num_splits)


_HEAD_DIMS = [(192, 128), (256, 256), (512, 512), (576, 512)]


@pytest.mark.parametrize("head_dim,head_dim_v", _HEAD_DIMS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [(1, 1, 8, 1), (3, 257, 32, 8), (2, 4096, 16, 16), (16, 512, 8, 2)],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
@_needs_swapab
def test_decode_swapab_head_dim(
    head_dim, head_dim_v, dtype, batch, seqlen_k, num_heads, num_heads_kv, num_splits
):
    torch.manual_seed(0)
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    out, lse = _attn(q, k, v, num_splits=num_splits)
    out_ref, lse_ref = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)
    assert out.shape == (batch, 1, num_heads, head_dim_v) and out.dtype == dtype
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("head_dim,head_dim_v", _HEAD_DIMS)
@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@_needs_swapab
def test_decode_swapab_head_dim_mask(
    head_dim, head_dim_v, num_vecs, causal, mask_heads, num_splits
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    _run_masked(q, k, v, mask, causal, num_splits)


@pytest.mark.parametrize("head_dim,head_dim_v", _HEAD_DIMS + [(128, 128)])
@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv",
    [(1, 1, 8, 1), (3, 257, 32, 8), (2, 4096, 16, 16), (16, 512, 8, 2), (4, 1000, 64, 2)],
)
@pytest.mark.parametrize("num_splits", [1, 0, 3])
@pytest.mark.parametrize("occupancy", [1, 2])
@_needs_swapab
def test_decode_swapab_occupancy(
    head_dim, head_dim_v, batch, seqlen_k, num_heads, num_heads_kv, num_splits, occupancy
):
    """Forced occupancy: two CTAs per SM (smaller TMEM allocation, own SMEM share) match the
    reference, and match one CTA per SM bit for bit."""
    qhead_per_kvhead = num_heads // num_heads_kv
    if swapab_tmem_cols(qhead_per_kvhead, head_dim_v) > 512 or not swapab_occupancy_fits(
        qhead_per_kvhead, head_dim_v, occupancy
    ):
        pytest.skip("does not fit TMEM")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    kw = dict(num_splits=num_splits)
    out, lse = _attn(q, k, v, occupancy=occupancy, **kw)
    _assert_within_naive(out, q, k, v, scale)
    torch.testing.assert_close(lse, _reference(q, k, v, scale)[1], atol=1e-3, rtol=1e-3)
    out1, lse1 = _attn(q, k, v, occupancy=1, **kw)
    assert torch.equal(out, out1) and torch.equal(lse, lse1)


@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (256, 256)])
@pytest.mark.parametrize("num_vecs,causal", _MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv", "all"])
@pytest.mark.parametrize("num_splits", [1, 0, 5])
@_needs_swapab
def test_decode_swapab_occupancy_mask(
    head_dim, head_dim_v, num_vecs, causal, mask_heads, num_splits
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    h_m = {"one": 1, "kv": num_heads_kv, "all": num_heads}[mask_heads]
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    mask = _block_sparse_mask(batch, h_m, seqlen_k, num_vecs, causal, "cuda")
    _run_masked(q, k, v, mask, causal, num_splits, occupancy=2)


@_needs_swapab
def test_decode_swapab_occupancy_too_big():
    """16 q heads per KV head at head_dim_v 512 need 288 TMEM columns: two CTAs per SM are
    refused."""
    q = torch.randn(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 256, 1, 512, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="do not fit TMEM"):
        _attn(q, k, k, occupancy=2)


@pytest.mark.parametrize(
    "batch,seqlen_k,num_heads,num_heads_kv", [(2, 4096, 8, 8), (3, 3000, 8, 1), (4, 1000, 16, 1)]
)
@pytest.mark.parametrize("num_splits", [1, 0])
@pytest.mark.parametrize("masked", [False, True])
@_needs_swapab
def test_decode_swapab_mla_shared_kv(batch, seqlen_k, num_heads, num_heads_kv, num_splits, masked):
    """MLA layout: one 576-dim latent cache, V is its first 512 dims (a strided view of K)."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(batch, 1, num_heads, 576, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, 576, device="cuda", dtype=dtype)
    v = k[..., :512]
    if masked:
        mask = _block_sparse_mask(batch, num_heads_kv, seqlen_k, 1, True, "cuda")
        _run_masked(q, k, v, mask, True, num_splits)
        return
    scale = 1.0 / math.sqrt(576)
    out, lse = _attn(q, k, v, num_splits=num_splits)
    out_ref, lse_ref = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)


def _assert_within_naive(out, q, k, v, scale):
    out_ref, _ = _reference(q, k, v, scale)
    out_naive, _ = _reference(q, k, v, scale, upcast=False)
    err = (out.float() - out_ref).abs().max().item()
    naive_err = (out_naive.to(q.dtype).float() - out_ref).abs().max().item()
    assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"


@pytest.mark.parametrize("num_splits", [1, 3])
@pytest.mark.parametrize("strided", [False, True])
@_needs_swapab
def test_decode_swapab_out_buffer(num_splits, strided):
    """out= is written in place, also as a strided view of a wider buffer."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(2, 1, 16, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    if strided:
        big = torch.zeros(2, 1, 16, 256, device="cuda", dtype=dtype)
        out = big[..., :128]
    else:
        out = torch.empty(2, 1, 16, 128, device="cuda", dtype=dtype)
    o, _ = _attn(q, k, v, num_splits=num_splits, return_lse=False, out=out)
    assert o.data_ptr() == out.data_ptr()
    _assert_within_naive(out, q, k, v, 1.0 / math.sqrt(128))
    if strided:
        assert (big[..., 128:] == 0).all()


@pytest.mark.parametrize("num_splits", [1, 3])
@_needs_swapab
def test_decode_swapab_softmax_scale(num_splits):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    q = torch.randn(2, 1, 16, 128, device="cuda", dtype=dtype)
    k = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    v = torch.randn(2, 1000, 2, 128, device="cuda", dtype=dtype)
    out, lse = _attn(q, k, v, softmax_scale=0.3, num_splits=num_splits)
    _assert_within_naive(out, q, k, v, 0.3)
    torch.testing.assert_close(lse, _reference(q, k, v, 0.3)[1], atol=1e-3, rtol=1e-3)


# _MASK_KINDS plus no mask.
_FRONTEND_MASK_KINDS = [(None, False), (None, True)] + _MASK_KINDS


@pytest.mark.parametrize("head_dim,head_dim_v", [(128, 128), (256, 256), (576, 512)])
@pytest.mark.parametrize("num_vecs,causal", _FRONTEND_MASK_KINDS)
@pytest.mark.parametrize("mask_heads", ["one", "kv"])
@_needs_swapab
def test_flashmask_attention_decode_dispatch(head_dim, head_dim_v, num_vecs, causal, mask_heads):
    """seqlen_q = 1 without autograd runs SwapAB with the main kernel's mask semantics."""
    if num_vecs is None and mask_heads == "kv":
        pytest.skip("no mask")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    batch, seqlen_k, num_heads, num_heads_kv = 3, 3000, 16, 4
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seqlen_k, num_heads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, seqlen_k, num_heads_kv, head_dim_v, device="cuda", dtype=dtype)
    mask = None
    if num_vecs is not None:
        h_m = 1 if mask_heads == "one" else num_heads_kv
        # Bounds in {0, 1, 2}: row 0 and row seqlen_k - 1 see different keys.
        mask = _random_mask(batch, h_m, seqlen_k, num_vecs, "cuda", causal, 2)
        mask[0, 0, :, 0] = 0  # sequence 0, mask head 0: every key hidden from row 0
        if num_vecs >= 2:
            mask[0, 0, :, 1] = 1

    with torch.no_grad():
        out, lse = _attn(q, k, v, mask, causal)
    # A q that requires grad keeps _flash_attn_fwd on the main kernel.
    with _expect_swapab(False):
        out_main, lse_main = _flash_attn_fwd(
            q.detach().requires_grad_(), k, v, causal=causal, return_lse=True,
            startend_row_indices=mask, pack_gqa=False,
        )
    assert out.shape == out_main.shape and lse.shape == lse_main.shape

    scale = 1.0 / math.sqrt(head_dim)
    if mask is None:
        keep = torch.ones(batch, num_heads, seqlen_k, dtype=torch.bool, device="cuda")
    else:
        keep = _col_mask_keep(mask, causal)
        keep = keep.repeat_interleave(num_heads // mask.shape[1], dim=1)
    out_ref, lse_ref = _masked_reference(q, k, v, scale, keep)
    out_naive, _ = _masked_reference(q, k, v, scale, keep, upcast=False)
    naive_err = (out_naive.to(dtype).float() - out_ref).abs().max().item()
    for o in (out, out_main):
        err = (o.float() - out_ref).abs().max().item()
        assert err <= 2 * naive_err + 1e-3, f"max err {err} vs naive {naive_err}"
    torch.testing.assert_close(lse, lse_ref, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse, lse_main, atol=1e-3, rtol=1e-3)


@_needs_swapab
def test_flashmask_attention_decode_dispatch_skipped_with_grad():
    """A call that needs gradients keeps the main kernel."""
    torch.manual_seed(0)
    q = torch.randn(2, 1, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
    with _expect_swapab(False):
        out = flashmask_attention(q, k, v)
    out.float().sum().backward()
    assert q.grad is not None
