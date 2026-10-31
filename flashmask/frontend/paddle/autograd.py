# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Paddle autograd wiring.

This stays per framework rather than moving behind the backend adapter:
`paddle.autograd.PyLayer` hands saved tensors back through
``ctx.saved_tensor()`` while `torch.autograd.Function` uses
``ctx.saved_tensors``, and the two disagree on how many gradient slots
``backward`` must return. The bodies only marshal context; the work happens in
`flashmask.cute.ops`.
"""

from typing import Tuple

import paddle

from ...cute.flashmask_utils import FlashMaskInfoPaddle
from ...cute.ops.bwd import _flash_attn_bwd
from ...cute.ops.fwd import _flash_attn_fwd

__all__ = ["FlashMaskFunc"]


class FlashMaskFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        causal: bool = False,
        softmax_scale: float | None = None,
        learnable_sink: paddle.Tensor | None = None,
        startend_row_indices: paddle.Tensor | None = None,
        block_mask: paddle.Tensor | None = None,
        group=None,
    ) -> paddle.Tensor | Tuple[paddle.Tensor, paddle.Tensor]:
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
        # Paddle requires backward to return None at exactly the positions whose
        # forward input had stop_gradient=True, so record
        # which of the differentiable inputs actually want a gradient. Read it
        # here rather than in backward: saved_tensor() does not carry the
        # forward tensor's stop_gradient back.
        ctx.needs_grad = (
            not query.stop_gradient,
            not key.stop_gradient,
            not value.stop_gradient,
            learnable_sink is not None and not learnable_sink.stop_gradient,
        )
        return [out, lse]

    @staticmethod
    def backward(ctx, dout, *args) -> Tuple[paddle.Tensor, ...]:
        query, key, value, startend_row_indices, out, lse, learnable_sink = ctx.saved_tensor()
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
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])["FLAGS_cudnn_deterministic"],
            learnable_sink=learnable_sink,
            group=ctx.group,
        )
        q_needs, k_needs, v_needs, sink_needs = ctx.needs_grad
        grads = (
            dq if q_needs else None,
            dk if k_needs else None,
            dv if v_needs else None,
        )
        # A None sink is not a forward input at all, so it owns no slot; a frozen
        # one owns a slot that must be None.
        if learnable_sink is None:
            return grads
        return grads + (dsink if sink_needs else None,)

