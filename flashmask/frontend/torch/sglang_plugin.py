# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""SGLang plugin registration for the FlashMask attention backend.

Installed under the ``sglang.srt.plugins`` entry-point group. SGLang discovers
it with ``load_plugins()`` early in every process, which registers the backend
with the attention registry and extends the CLI choices so
``--attention-backend flashmask`` validates.
"""

from sglang.srt.arg_groups.choices import add_attention_backend_choices
from sglang.srt.layers.attention.attention_registry import register_attention_backend

__all__ = ["register_plugin"]


def register_plugin():
    from .sglang_backend import FlashMaskAttnBackend

    @register_attention_backend("flashmask")
    def _create_flashmask_backend(runner):
        return FlashMaskAttnBackend(runner)

    add_attention_backend_choices(["flashmask"])
