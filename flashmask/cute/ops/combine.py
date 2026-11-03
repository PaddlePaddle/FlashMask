# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2026 The FlashMask Authors. Modifications for FlashMask.

"""Split-KV combine kernel orchestration.

Framework-agnostic: allocation, dtype and stream access go through a
`flashmask.backend` adapter (``be``).
"""

import math
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ...backend import Tensor, get_backend
from ..flash_fwd_combine import FlashAttentionForwardCombine

__all__ = ["_flash_attn_fwd_combine"]


# --- moved verbatim from the former cute/interface.py ---
def _flash_attn_fwd_combine(
    out_partial: Tensor,
    lse_partial: Tensor,
    out: Tensor,
    lse: Optional[Tensor] = None,
    num_splits_dynamic_ptr: Optional[Tensor] = None,
    semaphore_to_reset: Optional[Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim)
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads)
        out: Output tensor (batch, seqlen, nheads, headdim)
        lse: Output LSE tensor (batch, seqlen, nheads).
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    be = get_backend(out_partial, lse_partial, out)
    # Input validation
    assert out_partial.ndim in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.ndim in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [be.float16, be.bfloat16, be.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == be.float32, "lse_partial must be fp32"
    assert be.is_gpu(out_partial) and be.is_gpu(lse_partial), (
        "tensors must be on CUDA device"
    )
    assert be.strides(out_partial)[-1] == 1, "out_partial must be contiguous in the last dimension"
    assert be.strides(lse_partial)[-2] == 1, "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    # Validate output tensor shapes and types
    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == be.float32, "lse must be fp32"

    # Validate optional tensors
    for t, name in [
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == be.int32, f"{name} must be int32"
            assert be.is_gpu(t), f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    m_block_size = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if m_block_size == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Convert to cute tensors (using kernel-formatted tensors)
    out_partial_tensor = from_dlpack(out_partial.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=4
    )
    lse_partial_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse_partial.ndim - 2
    )
    out_tensor = from_dlpack(out.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=3
    )
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 2)
        if lse is not None
        else None
    )

    optional_tensors = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (num_splits_dynamic_ptr, semaphore_to_reset)
    ]
    num_splits_dynamic_tensor, semaphore_tensor = optional_tensors

    current_stream = cuda.CUstream(be.cuda_stream())

    # Create combine kernel configuration
    dtype = be.to_cute_dtype(out.dtype)
    dtype_partial = be.to_cute_dtype(out_partial.dtype)

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        lse is not None,
    )

    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        fa_combine = FlashAttentionForwardCombine(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
        )

        # Check if implementation is supported
        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=256,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        _flash_attn_fwd_combine.compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
        )

    _flash_attn_fwd_combine.compile_cache[compile_key](
        out_partial_tensor,
        lse_partial_tensor,
        out_tensor,
        lse_tensor,
        num_splits_dynamic_tensor,
        semaphore_tensor,
        current_stream,
    )


_flash_attn_fwd_combine.compile_cache = {}

