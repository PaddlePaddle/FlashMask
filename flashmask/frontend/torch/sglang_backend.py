# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""SGLang attention backend for FlashMask's SM100 kernels (task #36, v1).

The backend plugs FlashMask's fixed-length forward kernels into SGLang's
attention-backend contract:

* DECODE (single token) uses the SwapAB decode kernel when the page size is
  128 and the shapes fit its limits, otherwise the main SM100 kernel.
* TARGET_VERIFY (topk == 1) is treated as a fixed-width extend: the draft
  tokens are appended to the KV cache and causal attention is bottom-right
  aligned against ``seq_lens + n_q``.
* EXTEND / MIXED run one fixed-rectangle call per request.
* CUDA-graph capture is supported for DECODE and TARGET_VERIFY. Metadata
  buffers and the workspace are allocated in ``init_cuda_graph_state`` and
  refreshed outside the captured region on every replay.

v1 limitations (rejected with ``NotImplementedError``): masks / tree verify
(topk > 1), MLA, sliding window, cross-attention, logit softcap, FP8 KV,
mixed KV dtypes, DLLM_EXTEND / DRAFT_EXTEND_V2, and eager EXTEND capture.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# Importing this module implies the torch frontend package was imported first,
# which binds flashmask's torch backend adapter before the cute ops load.
from ...cute.ops._common import sm100_split_kv_supported
from ...cute.ops.fwd import _flash_attn_fwd
from ...cute.ops.workspace import FwdWorkspace
from ...cute.seqlen_info import SeqlenMeta

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.layout.paged_view import paged_kv_view
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_mode import ForwardMode

try:  # pragma: no cover - present in every SGLang install
    from sglang.kernels.ops.attention.trtllm_mha_page_table import (
        build_trtllm_mha_page_table,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "flashmask's SGLang backend needs sglang.kernels.ops.attention."
        "trtllm_mha_page_table; is SGLang installed?"
    ) from exc

try:  # pragma: no cover - optional at import time, used in __init__
    from sglang.srt.runtime_context import get_parallel
except ImportError:  # pragma: no cover
    get_parallel = None

__all__ = ["FlashMaskAttnBackend"]

_BIDIRECTIONAL_ATTN_TYPES = ("encoder_only", "decoder_bidirectional")
_MAX_SPLITS = 8


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "on", "yes")


def _attn_type_value(layer) -> str:
    value = getattr(layer, "attn_type", None)
    return str(getattr(value, "value", value))


class FlashMaskAttnBackend(AttentionBackend):
    """Run FlashMask's SM100 forward kernels under SGLang.

    ``__init__`` only captures pool/table references and allocates a reference
    tensor. Eager workspaces and metadata buffers are allocated lazily and
    grow monotonically; CUDA-graph buffers are allocated once in
    ``init_cuda_graph_state``, before any capture.
    """

    needs_cpu_seq_lens = False

    def __init__(self, model_runner):
        super().__init__()

        if _env_flag("SGLANG_ENABLE_POST_CAPTURE_KV_SIZING"):
            raise NotImplementedError(
                "The flashmask SGLang backend does not support "
                "SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1: the VMM path does not "
                "zero-fill KV pages, but the paged kernels require finite "
                "values past seqused_k in a sequence's last used page."
            )
        device = torch.device(model_runner.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise NotImplementedError("flashmask requires a CUDA device")
        major, _ = torch.cuda.get_device_capability(device)
        if major != 10:
            raise NotImplementedError(
                "The flashmask SGLang backend requires SM100 (B300); got "
                f"compute capability {major}"
            )

        model_config = model_runner.model_config
        self.model_config = model_config
        self.device = device
        self.page_size = int(model_runner.page_size)
        if self.page_size < 1 or 4096 % self.page_size != 0:
            raise NotImplementedError(
                "page_size must be a positive divisor of 4096 for the "
                "device-side page-table builder; "
                f"got page_size={self.page_size}"
            )
        if getattr(model_config, "is_encoder_decoder", False):
            raise NotImplementedError(
                "The flashmask SGLang backend does not support encoder-decoder "
                "(cross-attention) models."
            )
        arch = getattr(model_config, "attention_arch", "MHA")
        arch_name = str(getattr(arch, "name", arch)).upper()
        if arch_name != "MHA":
            raise NotImplementedError(
                "The flashmask SGLang backend only supports MHA attention; "
                f"got attention_arch={arch_name}"
            )
        self.sliding_window_size = getattr(model_runner, "sliding_window_size", None)
        if self.sliding_window_size is not None and self.sliding_window_size > -1:
            raise NotImplementedError(
                "The flashmask SGLang backend does not support sliding-window "
                "attention in v1."
            )

        self.dtype = model_config.dtype
        self.kv_cache_dtype = getattr(model_runner, "kv_cache_dtype", None)
        if self.kv_cache_dtype is None:
            self.kv_cache_dtype = self.dtype
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                "The flashmask SGLang backend requires fp16 or bf16 activations; "
                f"got {self.dtype}"
            )
        if self.kv_cache_dtype != self.dtype:
            raise NotImplementedError(
                "The flashmask SGLang backend requires the KV cache dtype to "
                f"equal the activation dtype ({self.dtype}); got "
                f"{self.kv_cache_dtype}"
            )

        max_context_len = getattr(model_config, "context_len", None)
        if max_context_len is None:
            max_context_len = getattr(model_runner, "max_context_len", 8192)
        self.max_context_len = int(max_context_len)
        self.max_num_pages = -(-self.max_context_len // self.page_size)
        self.max_seqlen_k = self.max_num_pages * self.page_size

        tp_size = 1
        if get_parallel is not None:
            tp_size = int(get_parallel().attn_tp_size)
        self.num_heads = int(model_config.get_num_attention_heads(tp_size))
        self.num_heads_kv = int(model_config.get_num_kv_heads(tp_size))
        if self.num_heads_kv < 1 or self.num_heads % self.num_heads_kv != 0:
            raise NotImplementedError(
                f"Invalid head configuration {self.num_heads}q/{self.num_heads_kv}kv"
            )
        self.head_dim = int(model_config.head_dim)
        head_dim_v = getattr(model_config, "v_head_dim", None)
        self.head_dim_v = int(head_dim_v) if head_dim_v else self.head_dim
        self.max_splits = (
            _MAX_SPLITS
            if sm100_split_kv_supported(
                self.head_dim, self.head_dim_v, torch.finfo(self.dtype).bits // 8
            )
            else 1
        )

        # Pool / table references captured at construction, like the FA backend.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = self.req_to_token_pool.req_to_token
        self.is_draft_worker = bool(getattr(model_runner, "is_draft_worker", False))

        self._ref: Optional[torch.Tensor] = None

        # CUDA-graph state (allocated in init_cuda_graph_state, before capture).
        self._graph_workspace: Optional[FwdWorkspace] = None
        self._graph_seqused_k: Optional[torch.Tensor] = None
        self._graph_page_table: Optional[torch.Tensor] = None
        self._graph_max_bs = 0
        self._graph_max_q = 0
        self._graph_captured = False

        # Eager state, grown lazily and never used by a captured graph.
        self._eager_seqused_k: Optional[torch.Tensor] = None
        self._eager_page_table: Optional[torch.Tensor] = None
        self._eager_workspaces = {}

        # Active per-step state, published by init_forward_metadata_out_graph.
        self._active_workspace: Optional[FwdWorkspace] = None
        self._active_seqused_k: Optional[torch.Tensor] = None
        self._active_page_table: Optional[torch.Tensor] = None
        self._active_bs = 0
        self._active_n_q = 1
        self._active_is_verify = False
        self._active_extend = False

    # -- workspace / metadata buffers -----------------------------------------

    def _make_workspace(
        self,
        max_batch: int,
        max_seqlen_q: int,
        max_splits: Optional[int] = None,
    ) -> FwdWorkspace:
        if self._ref is None:
            self._ref = torch.empty(1, device=self.device, dtype=self.dtype)
        return FwdWorkspace(
            self._ref,
            max_batch=max(1, int(max_batch)),
            max_seqlen_q=max(1, int(max_seqlen_q)),
            max_seqlen_k=self.max_seqlen_k,
            num_heads=self.num_heads,
            num_heads_kv=self.num_heads_kv,
            head_dim=self.head_dim,
            head_dim_v=self.head_dim_v,
            max_splits=self.max_splits if max_splits is None else max_splits,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Allocate the graph workspace and fixed metadata buffers (once)."""
        max_bs = max(1, int(max_bs))
        max_num_tokens = max(1, int(max_num_tokens))
        max_q = max(1, max_num_tokens // max_bs)

        if self._graph_workspace is not None:
            if max_bs <= self._graph_max_bs and max_q <= self._graph_max_q:
                return
            if self._graph_captured:
                raise RuntimeError(
                    "Cannot grow the flashmask CUDA-graph workspace after a "
                    "graph has been captured."
                )
            max_bs = max(max_bs, self._graph_max_bs)
            max_q = max(max_q, self._graph_max_q)

        self._graph_max_bs = max_bs
        self._graph_max_q = max_q
        self._graph_workspace = self._make_workspace(max_bs, max_q)
        self._graph_seqused_k = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._graph_page_table = torch.zeros(
            max_bs, self.max_num_pages, dtype=torch.int32, device=self.device
        )

    def _ensure_eager_meta(self, batch_size: int):
        if (
            self._eager_seqused_k is not None
            and batch_size <= self._eager_seqused_k.shape[0]
        ):
            return
        new_bs = max(
            batch_size,
            self._eager_seqused_k.shape[0] if self._eager_seqused_k is not None else 1,
        )
        self._eager_seqused_k = torch.zeros(
            new_bs, dtype=torch.int32, device=self.device
        )
        self._eager_page_table = torch.zeros(
            new_bs, self.max_num_pages, dtype=torch.int32, device=self.device
        )

    def _workspace_for(
        self,
        mode_key: str,
        max_batch: int,
        max_seqlen_q: int,
        max_splits: Optional[int] = None,
    ) -> FwdWorkspace:
        ws = self._eager_workspaces.get(mode_key)
        if (
            ws is not None
            and max_batch <= ws.max_batch
            and max_seqlen_q <= ws.max_seqlen_q
            and (max_splits is None or max_splits == ws.max_splits)
        ):
            return ws
        new_batch = max(max_batch, ws.max_batch if ws is not None else 1)
        new_q = max(max_seqlen_q, ws.max_seqlen_q if ws is not None else 1)
        ws = self._make_workspace(new_batch, new_q, max_splits=max_splits)
        self._eager_workspaces[mode_key] = ws
        return ws

    # -- metadata hooks --------------------------------------------------------

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        mode = forward_batch.forward_mode
        if mode.is_dllm_extend() or mode.is_draft_extend_v2():
            raise NotImplementedError(
                f"The flashmask SGLang backend does not support {mode.name} "
                "batches in v1."
            )

        bs = max(1, int(forward_batch.batch_size))
        num_tokens = int(forward_batch.positions.numel())
        n_q = max(1, num_tokens // bs)
        is_verify = bool(mode.is_target_verify())
        is_decode = bool(mode.is_decode()) or bool(mode.is_idle())
        is_extend = not (is_verify or is_decode)

        if is_verify:
            spec_info = getattr(forward_batch, "spec_info", None)
            custom_mask = getattr(spec_info, "custom_mask", None)
            topk = getattr(spec_info, "topk", None)
            if topk is None:
                topk = getattr(spec_info, "draft_token_num", n_q)
            if custom_mask is not None and int(topk) > 1:
                raise NotImplementedError(
                    "The flashmask SGLang backend does not support tree verify "
                    "(topk > 1) in v1."
                )
        if in_capture and is_extend:
            raise NotImplementedError(
                "The flashmask SGLang backend only supports CUDA graph capture "
                "for DECODE and TARGET_VERIFY in v1."
            )

        use_graph = (
            self._graph_workspace is not None
            and (is_decode or is_verify)
            and bs <= self._graph_max_bs
            and n_q <= self._graph_max_q
        )
        if self._graph_workspace is not None and (is_decode or is_verify) and not use_graph:
            raise RuntimeError(
                f"A {mode.name} batch of {bs} x {n_q} exceeds the flashmask "
                f"CUDA-graph workspace bounds "
                f"({self._graph_max_bs} x {self._graph_max_q})."
            )

        if use_graph:
            workspace = self._graph_workspace
            seqused_buf = self._graph_seqused_k
            page_table_buf = self._graph_page_table
        else:
            self._ensure_eager_meta(bs)
            if is_decode:
                workspace = self._workspace_for("decode", bs, 1)
            elif is_verify:
                workspace = self._workspace_for("verify", bs, n_q)
            else:
                # EXTEND/MIXED run one request at a time; the workspace only
                # ever sees a batch of 1 and a query length <= num_tokens.
                # Split-KV is disabled for extend to keep its partials small.
                workspace = self._workspace_for(
                    "extend", 1, max(1, num_tokens), max_splits=1
                )
            seqused_buf = self._eager_seqused_k
            page_table_buf = self._eager_page_table

        seq_vals = forward_batch.seq_lens[:bs].to(torch.int32)
        if is_verify:
            # The draft tokens are appended to the KV cache; the kernel's
            # bottom-right causal alignment gives each draft row the keys up to
            # seq_lens + n_q.
            seq_vals = seq_vals + n_q
        # Out of place: never mutate the caller's seq_lens (it is an int32 view
        # when the caller already uses int32).
        seq_vals = seq_vals.clamp(max=self.max_seqlen_k)
        seqused = seqused_buf[:bs]
        seqused.copy_(seq_vals)

        # Fixed-width page table (max_num_pages), filled on-device. Stale
        # entries beyond a sequence's seqused_k are never read by the kernels.
        build_trtllm_mha_page_table(
            self.req_to_token,
            forward_batch.req_pool_indices[:bs],
            seqused,
            page_table_buf[:bs],
            self.page_size,
        )

        if workspace.max_splits > 1 and not is_extend:
            # Device-side split-count plan, shared by every layer of this step.
            # EXTEND plans each per-request call separately (its workspace is
            # batch-1 with split-KV disabled).
            workspace.plan(seqused_k=seqused)

        self._active_workspace = workspace
        self._active_seqused_k = seqused_buf
        self._active_page_table = page_table_buf
        self._active_bs = bs
        self._active_n_q = n_q
        self._active_is_verify = is_verify
        self._active_extend = is_extend
        if in_capture:
            self._graph_captured = True

    def init_forward_metadata_in_graph(self, forward_batch):
        """No-op: all metadata is refreshed outside the captured region."""

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        """Nothing to do: verify metadata is rebuilt before every replay."""

    # -- layer / call validation ----------------------------------------------

    def _check_layer(self, layer):
        if getattr(layer, "is_cross_attention", False):
            raise NotImplementedError(
                "The flashmask SGLang backend does not support cross-attention "
                "layers."
            )
        if _attn_type_value(layer) in _BIDIRECTIONAL_ATTN_TYPES:
            raise NotImplementedError(
                "The flashmask SGLang backend does not support "
                f"{_attn_type_value(layer)} layers."
            )
        window = getattr(layer, "sliding_window_size", None)
        if window is not None and window > -1:
            raise NotImplementedError(
                "The flashmask SGLang backend does not support sliding-window "
                "layers."
            )
        logit_cap = getattr(layer, "logit_cap", None)
        if logit_cap:
            raise NotImplementedError(
                "The flashmask SGLang backend does not support softcap "
                "(logit_cap) layers."
            )
        if (layer.tp_q_head_num, layer.tp_k_head_num, layer.tp_v_head_num) != (
            self.num_heads,
            self.num_heads_kv,
            self.num_heads_kv,
        ):
            raise NotImplementedError(
                "Layer head configuration "
                f"({layer.tp_q_head_num}q/{layer.tp_k_head_num}k/"
                f"{layer.tp_v_head_num}v) does not match the backend "
                f"({self.num_heads}q/{self.num_heads_kv}k/"
                f"{self.num_heads_kv}v)."
            )
        if layer.qk_head_dim != self.head_dim or layer.v_head_dim != self.head_dim_v:
            raise NotImplementedError(
                "Layer head dimensions "
                f"({layer.qk_head_dim}/{layer.v_head_dim}) do not match the "
                f"backend ({self.head_dim}/{self.head_dim_v})."
            )

    @staticmethod
    def _check_unsupported_kwargs(kwargs):
        for name, value in kwargs.items():
            if value is None:
                continue
            raise NotImplementedError(
                f"The flashmask SGLang backend does not support the "
                f"'{name}' argument in v1."
            )

    # -- KV cache -------------------------------------------------------------

    def _store_kv(self, layer, forward_batch, k, v):
        loc = KVWriteLoc.for_layer(forward_batch, layer)
        self.kv_layers.layer(layer.layer_id).store(
            layer, loc, k, v, layer.k_scale, layer.v_scale
        )

    def _read_kv(self, layer):
        key_cache, value_cache = self.kv_layers.read(layer.layer_id).kv()
        key_cache = paged_kv_view(
            key_cache, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = paged_kv_view(
            value_cache, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        return key_cache, value_cache

    # -- forward --------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        *,
        return_lse: bool = False,
        **kwargs,
    ):
        self._check_layer(layer)
        self._check_unsupported_kwargs(kwargs)
        if k is not None and save_kv_cache:
            self._store_kv(layer, forward_batch, k, v)

        bs = self._active_bs
        key_cache, value_cache = self._read_kv(layer)
        q4 = q.contiguous().view(bs, 1, layer.tp_q_head_num, layer.qk_head_dim)
        out, lse = _flash_attn_fwd(
            q4,
            key_cache,
            value_cache,
            softmax_scale=layer.scaling,
            causal=True,
            seqlen_meta=SeqlenMeta(
                seqused_k=self._active_seqused_k[:bs],
                page_table=self._active_page_table[:bs],
            ),
            workspace=self._active_workspace,
            return_lse=return_lse,
        )
        out = out.reshape(bs, layer.tp_q_head_num * layer.v_head_dim)
        if return_lse:
            return out, lse.squeeze(-1)
        return out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        *,
        return_lse: bool = False,
        **kwargs,
    ):
        self._check_layer(layer)
        self._check_unsupported_kwargs(kwargs)
        if k is not None and save_kv_cache:
            self._store_kv(layer, forward_batch, k, v)

        key_cache, value_cache = self._read_kv(layer)
        bs = self._active_bs
        causal = not (
            getattr(layer, "is_cross_attention", False)
            or _attn_type_value(layer) in _BIDIRECTIONAL_ATTN_TYPES
        )

        if self._active_is_verify:
            n_q = max(1, q.shape[0] // bs)
            q4 = q.contiguous().view(
                bs, n_q, layer.tp_q_head_num, layer.qk_head_dim
            )
            out, lse = _flash_attn_fwd(
                q4,
                key_cache,
                value_cache,
                softmax_scale=layer.scaling,
                causal=True,
                seqlen_meta=SeqlenMeta(
                    seqused_k=self._active_seqused_k[:bs],
                    page_table=self._active_page_table[:bs],
                ),
                workspace=self._active_workspace,
                return_lse=return_lse,
            )
            out = out.reshape(bs * n_q, layer.tp_q_head_num * layer.v_head_dim)
            if return_lse:
                # kernel [B, H, N] -> [B * N, H]
                lse = lse.transpose(1, 2).reshape(
                    bs * n_q, layer.tp_q_head_num
                )
                return out, lse
            return out

        # EXTEND / MIXED: one fixed-rectangle call per request.
        lens = self._query_lens(forward_batch, q, bs)
        total = int(q.shape[0])
        out = torch.empty(
            (total, layer.tp_q_head_num, layer.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        lse_out = (
            torch.full(
                (total, layer.tp_q_head_num),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            if return_lse
            else None
        )

        offset = 0
        for i, seqlen_q in enumerate(lens):
            seqlen_q = int(seqlen_q)
            if seqlen_q <= 0:
                continue
            q_i = q[offset : offset + seqlen_q].contiguous().view(
                1, seqlen_q, layer.tp_q_head_num, layer.qk_head_dim
            )
            seqused_k = self._active_seqused_k[i : i + 1]
            page_table = self._active_page_table[i : i + 1]
            o_i, lse_i = _flash_attn_fwd(
                q_i,
                key_cache,
                value_cache,
                softmax_scale=layer.scaling,
                causal=causal,
                seqlen_meta=SeqlenMeta(seqused_k=seqused_k, page_table=page_table),
                workspace=self._active_workspace,
                return_lse=return_lse,
            )
            out[offset : offset + seqlen_q] = o_i[0]
            if return_lse:
                # kernel [1, H, L] -> [L, H]
                lse_out[offset : offset + seqlen_q] = lse_i[0].transpose(0, 1)
            offset += seqlen_q

        out = out.reshape(total, layer.tp_q_head_num * layer.v_head_dim)
        if return_lse:
            return out, lse_out
        return out

    def _query_lens(self, forward_batch, q, bs):
        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None and len(extend_seq_lens_cpu) >= bs:
            return [int(x) for x in extend_seq_lens_cpu[:bs]]
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None and extend_seq_lens.numel() >= bs:
            return extend_seq_lens[:bs].tolist()
        if bs == 1:
            return [int(q.shape[0])]
        raise NotImplementedError(
            "The flashmask SGLang backend needs extend_seq_lens_cpu (or "
            "extend_seq_lens) for MIXED batches."
        )
