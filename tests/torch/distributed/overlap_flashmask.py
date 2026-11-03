# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FM-4 context-parallel FlashMask with distributed overlap (DualChunkSwap), torch port.

Mirrors `tests/paddle/distributed/overlap_flashmask.py`: local query + local K/V +
the CP ``group`` are handed to the FA-v4 SM100 in-library kernel, which gathers
remote K/V internally (sparse all-gather) while computing, overlapping comm with
compute, and reduce-scatters dK/dV back to the local share.

The mask is handed to both passes in group-rank chunk order (dual-chunk
preprocessed + ``rearrange_blocks``); the library reorders it internally.
"""

import torch
import torch.distributed as dist

from cp_utils import preprocess_index_dual_chunks

from flashmask.cute.interface import _flash_attn_fwd, _flash_attn_bwd

_REORDER_IDX_CACHE = {}


def _dual_chunk_reorder_idx(cp_size, device):
    key = (cp_size, str(device))
    idx = _REORDER_IDX_CACHE.get(key)
    if idx is None:
        n_blocks = cp_size * 2
        order = []
        for i in range(cp_size):
            order.append(i)
            order.append(n_blocks - 1 - i)
        idx = torch.tensor(order, dtype=torch.int64, device=device)
        _REORDER_IDX_CACHE[key] = idx
    return idx


def rearrange_blocks(input_tensor, cp_size):
    """Interleave the 2*cp mask blocks into per-rank dual-chunk order."""
    B, _, S, _ = input_tensor.shape
    n_blocks = cp_size * 2
    block_size = S // n_blocks
    blocks = input_tensor.reshape([B, -1, n_blocks, block_size, 2])
    reordered = blocks.index_select(2, _dual_chunk_reorder_idx(cp_size, input_tensor.device))
    return reordered.reshape([B, -1, S, 2])


def _deterministic():
    return torch.are_deterministic_algorithms_enabled()


def _forward(query, key, value, startend_row_indices, group, causal):
    rank = dist.get_rank(group=group)
    cp_size = dist.get_world_size(group=group)

    seq_blocksize = query.shape[1] // 2
    startend_row_indices = preprocess_index_dual_chunks(
        startend_row_indices,
        chunk_id_first=rank,
        chunk_id_second=2 * cp_size - rank - 1,
        seq_blocksize=seq_blocksize,
        max_seqlen_q=seq_blocksize,
    )
    startend_row_indices = rearrange_blocks(startend_row_indices, cp_size)

    output, log_sum_exp = _flash_attn_fwd(
        query,
        key,
        value,
        startend_row_indices=startend_row_indices,
        causal=causal,
        return_lse=True,
        pack_gqa=False,
        group=group,
    )
    return output, log_sum_exp, startend_row_indices


def _backward(query, key, value, startend_row_indices, output, log_sum_exp, output_grad, group, causal):
    # dK/dV are already reduce-scattered to the local share.
    query_grad, key_grad, value_grad, _ = _flash_attn_bwd(
        query,
        key,
        value,
        output,
        output_grad,
        log_sum_exp,
        flashmask_info=startend_row_indices,
        causal=causal,
        group=group,
        deterministic=_deterministic(),
    )
    return query_grad, key_grad, value_grad


class OverlappedFlashMaskFM4(torch.autograd.Function):
    """FM-4 forward+backward overlap CP FlashMask (DualChunkSwap), torch."""

    @staticmethod
    def forward(ctx, query, key, value, startend_row_indices, causal, group):
        if causal:
            raise NotImplementedError("FM-4 overlap does not support causal=True.")
        assert group is not None, "overlap requires an explicit process group"
        assert query.shape[1] % 2 == 0, (
            f"Query sequence length {query.shape[1]} must be divisible by 2 (DualChunkSwap)."
        )

        output, log_sum_exp, startend_row_indices = _forward(
            query, key, value, startend_row_indices, group, causal
        )
        ctx.save_for_backward(query, key, value, output, log_sum_exp, startend_row_indices)
        ctx.group = group
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx, output_grad):
        query, key, value, output, log_sum_exp, startend_row_indices = ctx.saved_tensors
        query_grad, key_grad, value_grad = _backward(
            query, key, value, startend_row_indices, output, log_sum_exp,
            output_grad, ctx.group, ctx.causal,
        )
        # one slot per forward input: (query, key, value, startend_row_indices, causal, group)
        return query_grad, key_grad, value_grad, None, None, None


def overlap_flashmask_attention_fm4(query, key, value, startend_row_indices, causal=False, group=None, mode="overlap"):
    """Public API: FM-4 CP FlashMask attention with distributed overlap (torch)."""
    assert mode == "overlap", f"only overlap is implemented, got {mode}"
    return OverlappedFlashMaskFM4.apply(query, key, value, startend_row_indices, causal, group)
