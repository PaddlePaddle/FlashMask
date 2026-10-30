# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Framework adapter contract.

A backend instance is *bound to one device*, so call sites read like the
original framework-global calls (``be.zeros(shape=..., dtype=be.int32)``)
without threading a device argument through the kernel orchestration layer.
Paddle allocates on a framework-global default device; torch has no equivalent,
so the device has to live somewhere -- it lives here.

Scope is deliberately narrow: allocation, dtype/device queries, the stream
handle, and the eager tensor ops the block-sparsity derivation needs.

Not here, on purpose:

- autograd: ``PyLayer`` and ``autograd.Function`` differ in how saved tensors
  come back and in how many gradient slots ``backward`` returns. Hiding that
  behind one signature only moves the branching around.
- collectives: importing ``torch.distributed`` from here would make a plain
  attention call fail on builds without NCCL. See ``flashmask.distributed``.
- profiler ranges: same reason, and they are frontend concerns.

Every name in ``REQUIRED_ATTRS`` must exist on a concrete backend;
``flashmask.backend`` asserts that at load time so the two implementations
cannot drift apart silently.
"""

from typing import Any

#: ``paddle.Tensor`` / ``torch.Tensor`` in annotations, without importing either.
Tensor = Any

#: Kept explicit rather than derived from this module, so that adding a
#: capability is a conscious two-file change.
REQUIRED_ATTRS = (
    # identity
    "name",
    "tensor_types",
    # dtype constants
    "float16",
    "bfloat16",
    "float32",
    "int32",
    "int64",
    "bool_",
    "learnable_sink_dtypes",
    # dtype helpers
    "norm_dtype",
    "to_cute_dtype",
    # device / stream
    "device",
    "device_capability",
    "multi_processor_count",
    "cuda_stream",
    # introspection
    "strides",
    "data_ptr",
    "contiguous",
    "device_of",
    "is_gpu",
    "is_last_dim_contiguous",
    "requires_grad",
    "to_numpy",
    # factories
    "zeros",
    "empty",
    "full",
    "arange",
    "to_tensor",
    "zeros_like",
    "empty_like",
    "full_like",
    # eager ops
    "astype",
    "broadcast_to",
    "concat",
    "cumsum",
    "minimum",
    "put_along_axis",
    "repeat_interleave",
    "roll",
    "split",
    "sum",
    "where",
)


class BackendBase:
    """Shared behaviour; per-framework subclasses supply the rest."""

    name = None
    tensor_types = ()
    #: set per instance by __init__; declared so the contract check can see it
    device = None

    #: {framework dtype: cutlass numeric type}, filled in by subclasses
    _CUTE_DTYPE_MAP = {}

    def __init__(self, device):
        self.device = device

    def __repr__(self):
        return f"<{type(self).__name__} device={self.device}>"

    def norm_dtype(self, dtype):
        """Accept a framework dtype object, or the string spellings the existing
        call sites use (``dtype="int32"``, ``dtype='bool'``). Paddle takes those
        strings directly; torch does not."""
        if dtype is None:
            return None
        if isinstance(dtype, str):
            key = "bool_" if dtype == "bool" else dtype
            try:
                return getattr(self, key)
            except AttributeError:
                raise TypeError(f"{self.name} backend: unknown dtype string {dtype!r}")
        return dtype

    def to_cute_dtype(self, dtype):
        dtype = self.norm_dtype(dtype)
        try:
            return self._CUTE_DTYPE_MAP[dtype]
        except KeyError:
            raise TypeError(f"{self.name} backend: no cute dtype for {dtype}")

    def is_last_dim_contiguous(self, x):
        return self.strides(x)[-1] == 1


def check_backend_class(cls):
    """Raise if a backend class is missing part of the contract."""
    missing = [a for a in REQUIRED_ATTRS if not hasattr(cls, a)]
    if missing:
        raise TypeError(
            f"{cls.__name__} does not implement the backend contract; "
            f"missing: {', '.join(missing)}"
        )
    return cls
