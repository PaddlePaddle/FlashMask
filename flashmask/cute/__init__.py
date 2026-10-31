# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Flash Mask CUTE (CUDA Template Engine) implementation."""

__version__ = "4.0.0"

import cutlass.cute as cute

from .cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flashmask_attention",
]


def __getattr__(name):
    """Resolve the public entry point on first access.

    It lives in the paddle frontend, which imports this package -- binding it
    eagerly here is a cycle (`flashmask.cute` -> frontend -> `flashmask.cute`).
    Deferring also keeps `import flashmask.cute.<kernel>` free of any framework
    dependency: the kernel modules themselves import neither paddle nor torch.

    The top-level package imports `interface` right after the paddle frontend,
    which binds it in this namespace. It is read from there rather than imported:
    a host that moves the package to another name in sys.modules after loading it
    (paddlefleet_ops does) leaves nothing for an import statement to resolve.
    """
    if name in __all__:
        interface = globals().get("interface")
        if interface is None:
            raise AttributeError(
                f"{__name__}.{name} needs the paddle frontend, which failed to import"
            )
        return getattr(interface, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
