# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Framework dispatch.

``get_backend(*tensors)`` returns the adapter bound to the tensors' device and
refuses a mix of frameworks. Detection only consults frameworks already in
``sys.modules``: a tensor cannot exist without its framework being imported, so
this never drags torch into a paddle-only process, or the reverse.
"""

import sys

from .base import REQUIRED_ATTRS, BackendBase, Tensor, check_backend_class

__all__ = [
    "REQUIRED_ATTRS",
    "BackendBase",
    "Tensor",
    "framework_of",
    "get_backend",
    "get_backend_by_name",
]

#: framework name -> (submodule, class name)
_IMPLS = {
    "paddle": ("_paddle", "PaddleBackend"),
    "torch": ("_torch", "TorchBackend"),
}

_CLASS_CACHE = {}
_INSTANCE_CACHE = {}


def _backend_class(name):
    if name not in _CLASS_CACHE:
        if name not in _IMPLS:
            raise ValueError(f"unknown framework {name!r}; expected one of {sorted(_IMPLS)}")
        submodule, cls_name = _IMPLS[name]
        # Imported by flashmask.frontend.<name>, which binds it in this namespace.
        # Read rather than imported: a host that moves the package to another name
        # in sys.modules after loading it (paddlefleet_ops does) leaves nothing for
        # an import statement to resolve.
        module = globals().get(submodule)
        if module is None:
            raise RuntimeError(
                f"no {name} backend: flashmask.frontend.{name} was not imported"
            )
        _CLASS_CACHE[name] = check_backend_class(getattr(module, cls_name))
    return _CLASS_CACHE[name]


def framework_of(tensor):
    """Name of the framework owning ``tensor``, or None."""
    for name in _IMPLS:
        module = sys.modules.get(name)
        if module is not None and isinstance(tensor, module.Tensor):
            return name
    return None


def _device_of(tensor, framework):
    return tensor.place if framework == "paddle" else tensor.device


def get_backend_by_name(name, device):
    key = (name, str(device))
    if key not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[key] = _backend_class(name)(device)
    return _INSTANCE_CACHE[key]


def get_backend(*tensors):
    """Adapter bound to the device of the first recognised tensor.

    Raises on a mix of frameworks in one call. The kernels take raw pointers, so
    a mixed call would otherwise hand one framework's allocation to the other's
    stream instead of failing.
    """
    chosen = None
    chosen_tensor = None
    for tensor in tensors:
        if tensor is None:
            continue
        framework = framework_of(tensor)
        if framework is None:
            continue
        if chosen is None:
            chosen, chosen_tensor = framework, tensor
        elif framework != chosen:
            raise TypeError(
                f"cannot mix {chosen} and {framework} tensors in one flashmask call"
            )
    if chosen is None:
        raise TypeError(
            "no paddle or torch tensor found in the arguments; cannot pick a backend"
        )
    return get_backend_by_name(chosen, _device_of(chosen_tensor, chosen))
