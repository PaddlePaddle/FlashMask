# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Backwards-compatible import path.

This module used to hold the whole implementation. It was split into

- `flashmask.cute.ops`      -- framework-agnostic kernel orchestration
- `flashmask.frontend.paddle` -- paddle autograd + public API

and now only re-exports, so that `from flashmask.cute.interface import ...`
keeps working for the existing tests and benchmarks. New code should import
from `flashmask.frontend.<framework>` (public API) or `flashmask.cute.ops`
(orchestration).

Importing this module pulls in paddle, by definition of what it re-exports.
Torch callers should use `flashmask.frontend.torch` instead.
"""

from .ops._common import (
    BwdConfig,
    _check_cutedsl_supported,
    _get_compute_capability,
    _is_valid_flash_dims,
    _make_fake_bwd_tensors,
    _same_storage,
    _tile_size_bwd_sm90,
    maybe_contiguous,
    num_splits_heuristic,
)
from .ops.bwd import _flash_attn_bwd
from .ops.combine import _flash_attn_fwd_combine
from .ops.fwd import _flash_attn_fwd
from ..frontend.paddle import (
    flashmask_attention,
)
from ..frontend.paddle.autograd import (
    FlashMaskFunc,
)

__all__ = [
    "BwdConfig",
    "FlashMaskFunc",
    "_check_cutedsl_supported",
    "_flash_attn_bwd",
    "_flash_attn_fwd",
    "_flash_attn_fwd_combine",
    "_get_compute_capability",
    "_is_valid_flash_dims",
    "_make_fake_bwd_tensors",
    "_same_storage",
    "_tile_size_bwd_sm90",
    "flashmask_attention",
    "maybe_contiguous",
    "num_splits_heuristic",
]
