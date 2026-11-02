# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""Paddle implementation of the backend contract."""

import cutlass
import paddle

from .base import BackendBase


class PaddleBackend(BackendBase):
    name = "paddle"
    tensor_types = (paddle.Tensor,)

    float16 = paddle.float16
    bfloat16 = paddle.bfloat16
    float32 = paddle.float32
    int32 = paddle.int32
    int64 = paddle.int64
    bool_ = paddle.bool

    learnable_sink_dtypes = (paddle.float16, paddle.bfloat16, paddle.float32)

    _CUTE_DTYPE_MAP = {
        paddle.float16: cutlass.Float16,
        paddle.bfloat16: cutlass.BFloat16,
        paddle.float32: cutlass.Float32,
    }

    # -- device / stream -----------------------------------------------------

    def device_capability(self):
        return paddle.device.cuda.get_device_capability()

    def multi_processor_count(self):
        device_id = self.device.gpu_device_id() if hasattr(self.device, "gpu_device_id") else 0
        return paddle.device.cuda.get_device_properties(device_id).multi_processor_count

    def cuda_stream(self):
        return paddle.device.current_stream().stream_base.cuda_stream

    def is_stream_capturing(self):
        # Paddle only exposes the process-wide CUDA graph capture state.
        return bool(paddle.base.core.is_cuda_graph_capturing())

    # -- introspection -------------------------------------------------------

    def strides(self, x):
        return x.strides

    def data_ptr(self, x):
        return x.data_ptr()

    def contiguous(self, x):
        return x.contiguous()

    def is_gpu(self, x):
        return x.place.is_gpu_place()

    def device_of(self, x):
        return x.place

    def requires_grad(self, x):
        # paddle spells "wants a gradient" as the negation of stop_gradient.
        return not x.stop_gradient

    def to_numpy(self, x):
        return x.numpy()

    # -- factories -----------------------------------------------------------

    def zeros(self, shape, dtype=None):
        return paddle.zeros(shape, self.norm_dtype(dtype), device=self.device)

    def empty(self, shape, dtype=None):
        return paddle.empty(shape, self.norm_dtype(dtype), device=self.device)

    def full(self, shape, fill_value, dtype=None):
        return paddle.full(shape, fill_value, self.norm_dtype(dtype), device=self.device)

    def arange(self, start=0, end=None, step=1, dtype=None):
        return paddle.arange(start, end, step, self.norm_dtype(dtype), device=self.device)

    def to_tensor(self, data, dtype=None):
        return paddle.to_tensor(data, self.norm_dtype(dtype), place=self.device)

    def zeros_like(self, x, dtype=None):
        return paddle.zeros_like(x, self.norm_dtype(dtype))

    def empty_like(self, x, dtype=None):
        return paddle.empty_like(x, self.norm_dtype(dtype))

    def full_like(self, x, fill_value, dtype=None):
        return paddle.full_like(x, fill_value, self.norm_dtype(dtype))

    # -- eager ops -----------------------------------------------------------

    def astype(self, x, dtype):
        return x.astype(self.norm_dtype(dtype))

    def broadcast_to(self, x, shape):
        return paddle.broadcast_to(x, shape)

    def concat(self, xs, axis=0):
        return paddle.concat(xs, axis)

    def cumsum(self, x, axis=None):
        return paddle.cumsum(x, axis)

    def minimum(self, x, y):
        return paddle.minimum(x, y)

    def put_along_axis(self, arr, indices, values, axis):
        return paddle.put_along_axis(arr, indices, values, axis)

    def repeat_interleave(self, x, repeats, axis=None):
        return paddle.repeat_interleave(x, repeats, axis)

    def roll(self, x, shifts, axis):
        return paddle.roll(x, shifts, axis)

    def split(self, x, num_or_sections, axis=0):
        return paddle.split(x, num_or_sections, axis)

    def sum(self, x, axis=None, keepdim=False):
        return paddle.sum(x, axis, keepdim=keepdim)

    def where(self, condition, x, y):
        return paddle.where(condition, x, y)
