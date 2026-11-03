# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Torch autograd wiring.

Mirrors `flashmask.frontend.paddle.autograd` and calls the same
`flashmask.cute.ops` functions. Four differences from the paddle side are
forced by `torch.autograd.Function`:

- ``apply`` passes arguments positionally, so the public wrappers must not use
  keywords when calling it (``PyLayer.apply`` accepts keywords).
- saved tensors come back via ``ctx.saved_tensors``, not ``ctx.saved_tensor()``.
- ``backward`` returns one gradient slot per forward input, including ``None``
  for the non-tensor ones. Paddle instead expects slots only for inputs that
  wanted a gradient, which is why it has to record ``stop_gradient`` by hand;
  torch hands us ``ctx.needs_input_grad``.
- ``lse`` is marked non-differentiable, so autograd never asks for a gradient
  w.r.t. it.
"""

from typing import Optional

import torch

from ...cute.flashmask_utils import FlashMaskInfoPaddle
from ...cute.ops.bwd import _flash_attn_bwd
from ...cute.ops.fwd import _flash_attn_fwd

__all__ = ["FlashMaskFunc"]


class FlashMaskFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        learnable_sink: Optional[torch.Tensor] = None,
        startend_row_indices: Optional[torch.Tensor] = None,
        group=None,
    ):
        out, lse = _flash_attn_fwd(
            query,
            key,
            value,
            causal=causal,
            softmax_scale=softmax_scale,
            learnable_sink=learnable_sink,
            return_lse=True,
            startend_row_indices=startend_row_indices,
            pack_gqa=False,
            group=group,
        )
        ctx.save_for_backward(query, key, value, startend_row_indices, out, lse, learnable_sink)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.group = group
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        query, key, value, startend_row_indices, out, lse, learnable_sink = ctx.saved_tensors
        if startend_row_indices is not None:
            flashmask_info = FlashMaskInfoPaddle(
                startend_row_indices=startend_row_indices,
                is_causal=ctx.causal,
            )
        else:
            flashmask_info = None
        # learnable_sink is still handed to the backward kernel even when it is
        # frozen: it shifts the softmax normalisation, so dq/dk/dv depend on it.
        dq, dk, dv, dsink = _flash_attn_bwd(
            query,
            key,
            value,
            out,
            dout,
            lse,
            flashmask_info,
            softmax_scale=ctx.softmax_scale,
            causal=ctx.causal,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            learnable_sink=learnable_sink,
            group=ctx.group,
        )
        q_needs, k_needs, v_needs = ctx.needs_input_grad[:3]
        sink_needs = ctx.needs_input_grad[5]
        # Unlike paddle, the slot count is fixed: one per forward input, with
        # None for the ones that did not want a gradient.
        return (
            dq if q_needs else None,
            dk if k_needs else None,
            dv if v_needs else None,
            None,  # causal
            None,  # softmax_scale
            dsink if sink_needs else None,
            None,  # startend_row_indices
            None,  # group
        )
