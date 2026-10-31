# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FlashMask: CuTe DSL attention kernels for SM90 / SM100.

Two ways in. Either name the framework:

    from flashmask.frontend.paddle import flashmask_attention
    from flashmask.frontend.torch import flashmask_attention

or let the input tensors pick:

    import flashmask
    out = flashmask.flashmask_attention(q, k, v, startend_row_indices=idx, causal=True)

Neither framework is a hard dependency: the paddle names below only appear if
paddle imports, and likewise for torch.
"""

from .backend import framework_of

__all__ = ["flashmask_attention"]


def _frontend_for(tensor):
    """Return the frontend module owning ``tensor``, or raise.

    Reads the frontends bound at import time below instead of importing: a host
    that moves this package to another name in sys.modules after loading it
    (paddlefleet_ops does) leaves nothing for an import statement to resolve.
    """
    framework = framework_of(tensor)
    if framework is None:
        raise TypeError(
            f"expected a paddle or torch tensor, got {type(tensor).__name__}. "
            "Note that the owning framework must already be imported."
        )
    frontend = globals()[f"_{framework}_frontend"]
    if frontend is None:
        raise ImportError(
            f"the flashmask {framework} frontend failed to import"
        ) from globals()[f"_{framework}_import_error"]
    return frontend, frontend.flashmask_attention


def flashmask_attention(query, key, value, *args, **kwargs):
    """Dispatch to the frontend matching the input tensor type.

    The remaining arguments are forwarded untouched, so they are those of
    `flashmask.frontend.paddle.flashmask_attention` for paddle tensors and
    `flashmask.frontend.torch.flashmask_attention` for torch tensors. The two
    signatures differ: the torch one drops the keyword arguments that only exist
    to be asserted at their defaults. This function deliberately does not
    translate between them -- pick the frontend directly if you want one fixed
    signature.
    """
    _, fn = _frontend_for(query)
    return fn(query, key, value, *args, **kwargs)


# ============================================================
# Paddle frontend (optional)
# ============================================================
_paddle_available = False
_paddle_frontend = None
_paddle_import_error = None
try:
    from .frontend import paddle as _paddle_frontend  # noqa: F401
    # Binds flashmask.cute.flashmask_attention (see cute.__getattr__). Bound here,
    # after the frontend, because cute itself importing it would be a cycle.
    from .cute import interface as _cute_interface  # noqa: F401

    _paddle_available = True
except ImportError as exc:
    # Kept, not swallowed: a broken install and "paddle is simply not present"
    # look identical from the outside otherwise.
    _paddle_import_error = exc

# ============================================================
# Torch frontend (optional)
# ============================================================
_torch_available = False
_torch_frontend = None
_torch_import_error = None
try:
    from .frontend import torch as _torch_frontend  # noqa: F401

    _torch_available = True
except ImportError as exc:
    _torch_import_error = exc

if not (_paddle_available or _torch_available):
    print("[WARNING] flashmask: neither the paddle nor the torch frontend could "
          f"be imported. paddle: {_paddle_import_error}; torch: {_torch_import_error}")
