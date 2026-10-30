# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Torch-facing public API.

    from flashmask.frontend.torch import flashmask_attention

Deliberately narrower than the paddle signatures. The paddle entry points carry
``dropout`` / ``window_size`` / ``return_seed_offset`` / ``fixed_seed_offset`` /
``rng_name`` / ``training`` / ``name`` / ``block_mask``, which exist only to be
asserted at their defaults -- they were parameters of the
`paddle.nn.functional` op this package used to fall back to. That fallback is
gone, so they are not part of the torch surface. Varlen is not exposed either:
the kernels do not support it.

There is no fallback here either. An unsupported device or head-dim combination
raises from `_check_cutedsl_supported`, with the same text the paddle frontend
produces.
"""

from typing import Optional

import torch

from flashmask.backend import get_backend
from flashmask.cute.ops._common import _check_cutedsl_supported, _dist_world_size
from .autograd import FlashMaskFunc

__all__ = [
    "flashmask_attention",
]


def flashmask_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    startend_row_indices: Optional[torch.Tensor] = None,
    *,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    group=None,
):
    """FlashMask attention.

    Args:
        q, k, v: (batch, seqlen, num_heads, head_dim).
        startend_row_indices: int32 (batch, num_heads_k or 1, seqlen_k, n) mask
            description. ``n`` is 1 or 2 when ``causal``, 2 or 4 otherwise.
            ``None`` runs dense attention.
        causal: apply the causal mask on top of ``startend_row_indices``.
        softmax_scale: defaults to ``1 / sqrt(head_dim)``.
        learnable_sink: (num_heads,) additive sink logit per head.
        return_lse: also return the log-sum-exp, (batch, num_heads, seqlen_q).
        group: context-parallel process group; enables the FM-4 overlap path
            when its world size is greater than 1. Then k/v are the local S_local
            shard and the mask key axis has length S_local * group_size, in
            group-rank chunk order (chunk r is rank r's local K/V). The library
            reorders it internally; pass it unshifted.

    Returns:
        ``out``, or ``(out, lse)`` when ``return_lse``.
    """
    be = get_backend(q, k, v)
    _check_cutedsl_supported(q, k, v, be)

    if startend_row_indices is not None:
        assert startend_row_indices.dtype == torch.int32, (
            f"startend_row_indices.dtype must be torch.int32, but got {startend_row_indices.dtype}"
        )
        assert startend_row_indices.dim() == 4, (
            f"startend_row_indices rank must be 4, but got {list(startend_row_indices.shape)}"
        )
        assert startend_row_indices.shape[0] == k.shape[0], (
            "startend_row_indices.shape[0] must equal batch_size, but got "
            f"{startend_row_indices.shape[0]} and {k.shape[0]}"
        )
        # Overlap: the mask covers the full key length, group-rank chunk order.
        group_size = _dist_world_size(be, group) if group is not None else 1
        seqlen_k = k.shape[1] * group_size
        assert startend_row_indices.shape[2] == seqlen_k, (
            "startend_row_indices.shape[2] must equal seqlen_k (S_local * group_size "
            f"with a CP group), but got {startend_row_indices.shape[2]} and {seqlen_k}"
        )
        assert startend_row_indices.shape[1] in (1, k.shape[2]), (
            "startend_row_indices head_num must equal 1 (broadcast) or head_num_k."
        )
        last = startend_row_indices.shape[-1]
        expected = (1, 2) if causal else (2, 4)
        assert last in expected, (
            f"invalid startend_row_indices last dimension {last}; with causal="
            f"{causal} it must be one of {expected}"
        )

    # apply() passes positionally: torch.autograd.Function does not take keywords.
    out, lse = FlashMaskFunc.apply(
        q, k, v, causal, softmax_scale, learnable_sink, startend_row_indices, group
    )
    return (out, lse) if return_lse else out
