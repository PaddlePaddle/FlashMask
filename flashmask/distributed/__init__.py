# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Collective dispatch.

``get_dist(name)`` is keyed by framework name rather than by sniffing the group
object: the group may be ``None`` (meaning the default group), in which case
there is nothing to sniff. Callers always have a backend at hand, so they pass
``be.name``.
"""

from .base import REQUIRED_ATTRS, DistBase, check_dist_class

__all__ = ["REQUIRED_ATTRS", "DistBase", "get_dist"]

#: framework name -> (submodule, class name)
_IMPLS = {
    "paddle": ("_paddle", "PaddleDist"),
    "torch": ("_torch", "TorchDist"),
}

_CACHE = {}


def get_dist(name):
    """Collective adapter for a framework name (``be.name``)."""
    if name not in _CACHE:
        if name not in _IMPLS:
            raise ValueError(f"unknown framework {name!r}; expected one of {sorted(_IMPLS)}")
        submodule, cls_name = _IMPLS[name]
        # Bound by flashmask.frontend.<name>; see flashmask.backend._backend_class.
        module = globals().get(submodule)
        if module is None:
            raise RuntimeError(
                f"no {name} collectives: flashmask.frontend.{name} was not imported"
            )
        _CACHE[name] = check_dist_class(getattr(module, cls_name))()
    return _CACHE[name]
