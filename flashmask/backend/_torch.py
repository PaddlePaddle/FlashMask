# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Torch implementation of the backend contract.

Only the places where torch's spelling differs from paddle's are worth reading:

- ``split``: paddle's ``num_or_sections=int`` means "this many pieces", while
  ``torch.split(t, int)`` means "pieces of this size". ``torch.chunk`` matches.
- ``put_along_axis``: paddle's assign-mode scatter is ``Tensor.scatter``, which
  additionally requires ``index`` to be int64 and ``src`` to share ``self``'s
  dtype. Paddle enforces neither, so both are coerced here.
- ``arange``: paddle reads ``arange(5)`` as ``[0, 5)``; torch's positional
  ``arange(start, end)`` needs the shift done explicitly.
- factories take the device from the bound backend, since torch has no
  framework-global default device to fall back on.
"""

import cutlass
import torch

from .base import BackendBase


class TorchBackend(BackendBase):
    name = "torch"
    tensor_types = (torch.Tensor,)

    float16 = torch.float16
    bfloat16 = torch.bfloat16
    float32 = torch.float32
    int32 = torch.int32
    int64 = torch.int64
    bool_ = torch.bool

    learnable_sink_dtypes = (torch.float16, torch.bfloat16, torch.float32)

    _CUTE_DTYPE_MAP = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    # -- device / stream -----------------------------------------------------

    def device_capability(self):
        return torch.cuda.get_device_capability(self.device)

    def multi_processor_count(self):
        return torch.cuda.get_device_properties(self.device).multi_processor_count

    def cuda_stream(self):
        return torch.cuda.current_stream(self.device).cuda_stream

    # -- introspection -------------------------------------------------------

    def strides(self, x):
        return x.stride()

    def data_ptr(self, x):
        return x.data_ptr()

    def contiguous(self, x):
        return x.contiguous()

    def is_gpu(self, x):
        return x.is_cuda

    def device_of(self, x):
        return x.device

    def requires_grad(self, x):
        return x.requires_grad

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    # -- factories -----------------------------------------------------------

    def zeros(self, shape, dtype=None):
        return torch.zeros(tuple(shape), dtype=self.norm_dtype(dtype), device=self.device)

    def empty(self, shape, dtype=None):
        return torch.empty(tuple(shape), dtype=self.norm_dtype(dtype), device=self.device)

    def full(self, shape, fill_value, dtype=None):
        return torch.full(
            tuple(shape), fill_value, dtype=self.norm_dtype(dtype), device=self.device
        )

    def arange(self, start=0, end=None, step=1, dtype=None):
        if end is None:
            start, end = 0, start
        return torch.arange(
            start, end, step, dtype=self.norm_dtype(dtype), device=self.device
        )

    def to_tensor(self, data, dtype=None):
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device, dtype=self.norm_dtype(dtype))
        return torch.tensor(data, dtype=self.norm_dtype(dtype), device=self.device)

    def zeros_like(self, x, dtype=None):
        return torch.zeros_like(x, dtype=self.norm_dtype(dtype))

    def empty_like(self, x, dtype=None):
        return torch.empty_like(x, dtype=self.norm_dtype(dtype))

    def full_like(self, x, fill_value, dtype=None):
        return torch.full_like(x, fill_value, dtype=self.norm_dtype(dtype))

    # -- eager ops -----------------------------------------------------------

    def astype(self, x, dtype):
        return x.to(self.norm_dtype(dtype))

    def broadcast_to(self, x, shape):
        return torch.broadcast_to(x, tuple(shape))

    def concat(self, xs, axis=0):
        return torch.cat(tuple(xs), dim=axis)

    def cumsum(self, x, axis=None):
        return torch.cumsum(x, dim=axis)

    def minimum(self, x, y):
        return torch.minimum(x, y)

    def put_along_axis(self, arr, indices, values, axis):
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)
        if not isinstance(values, torch.Tensor):
            return arr.scatter(axis, indices, values)
        return arr.scatter(axis, indices, values.to(arr.dtype))

    def repeat_interleave(self, x, repeats, axis=None):
        return torch.repeat_interleave(x, repeats, dim=axis)

    def roll(self, x, shifts, axis):
        return torch.roll(x, shifts, dims=axis)

    def split(self, x, num_or_sections, axis=0):
        if isinstance(num_or_sections, int):
            return list(torch.chunk(x, num_or_sections, dim=axis))
        return list(torch.split(x, tuple(num_or_sections), dim=axis))

    def sum(self, x, axis=None, keepdim=False):
        if axis is None:
            return torch.sum(x)
        return torch.sum(x, dim=axis, keepdim=keepdim)

    def where(self, condition, x, y):
        return torch.where(condition, x, y)
