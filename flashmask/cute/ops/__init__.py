# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Kernel orchestration, shared by every frontend.

These modules do not import paddle or torch. They take a `flashmask.backend`
adapter for allocation, dtype, stream and device queries, and treat a
distributed ``group`` as an opaque handle.
"""

from ._common import _check_cutedsl_supported
from .bwd import _flash_attn_bwd
from .combine import _flash_attn_fwd_combine
from .fwd import _flash_attn_fwd
from .workspace import FwdWorkspace

__all__ = [
    "FwdWorkspace",
    "_check_cutedsl_supported",
    "_flash_attn_bwd",
    "_flash_attn_fwd",
    "_flash_attn_fwd_combine",
]
