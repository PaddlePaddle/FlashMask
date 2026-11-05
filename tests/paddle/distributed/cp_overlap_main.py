#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The FlashMask Authors. All rights reserved.

"""FM-4 CP overlap accuracy test.

ONE mask per process: a process fixes the mask type and cycles head config
(MQA/MHA/GQA) and batch size in-process, N reps each.

Split axes (separate processes): CP size x mask_type x head dim.
In-process sweep: batch x hk, N reps each. The single DOC_PACKING sums to 32768 so it
fits S_total = s_local * cp = 32768 exactly (no padding/scaling).

--s_locals A,B,... instead runs every config at each S_local in that order inside
ONE process (same on all ranks), exercising the runtime's per-call S handling. Each
S_total gets its own full mask and inputs; doc lengths are DOC_PACKING scaled to
S_total (fixed ratios, remainder into the last doc) so documents still cross owners.

Both paths use DualChunkSwap and the same FA-v4 SM100 kernel, so per-rank outputs
differ only by floating-point reduction order (thresholds in compare.py).

Launch with exactly cp_size GPUs; pass --cp_size to match. See run_cp_overlap.sh.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import paddle
from paddle.distributed import fleet

# launched as a script by run_cp_overlap.sh, so put the repo root on sys.path for
# the framework-neutral helpers in tests/common (siblings resolve on their own)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cp_utils import scatter_balance
from all_gather_flashmask import flashmask_attention_cp
from overlap_flashmask import overlap_flashmask_attention_fm4
from tests.common.startend_row_indices import DOC_PACKING, GENERATORS
from tests.common.compare import check_accuracy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cp_size", type=int, required=True, help="CP size = launched world size")
    p.add_argument("--d", type=int, required=True, help="head dim (split axis)")
    p.add_argument("--s_local", type=int, required=True, help="local seqlen (fixed per process)")
    p.add_argument("--s_locals", type=str, default="",
                   help="comma list of local seqlens run in order in-process; overrides --s_local")
    p.add_argument("--bs", type=str, default="1,2", help="batch sizes cycled in-process")
    p.add_argument("--hq", type=int, default=16, help="query heads")
    p.add_argument("--hks", type=str, default="16,4,1", help="kv-head counts cycled in-process")
    p.add_argument("--masks", type=str, default="document,prefix_lm", help="mask type(s); one per process")
    p.add_argument("--repeats", type=int, default=2, help="fresh-random-input repeats per config")
    p.add_argument("--seed", type=int, default=2024)
    return p.parse_args()


def _int_list(s):
    return [int(x) for x in s.split(",") if x != ""]


opts = parse_args()

assert paddle.device.cuda.get_device_capability()[0] >= 10, "FM-4 overlap requires SM100+."
_fa = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])["FLAGS_flash_attn_version"]
assert _fa == 4, f"Set FLAGS_flash_attn_version=4 (got {_fa})."

strategy = fleet.DistributedStrategy()
strategy.hybrid_configs = {
    "dp_degree": 1, "mp_degree": 1, "pp_degree": 1,
    "sharding_degree": opts.cp_size, "sep_degree": 1, "ep_degree": opts.cp_size,
    "moe_sharding_degree": 1, "cp_degree": opts.cp_size,
    "order": ["sharding", "moe_sharding", "pp", "sep", "cp", "dp", "ep", "mp"],
}
fleet.init(is_collective=True, strategy=strategy)
cp_group = fleet.get_hybrid_communicate_group().get_context_parallel_group()
cp_size = cp_group.world_size
rank = paddle.distributed.get_rank()
assert cp_size == opts.cp_size, f"cp_group={cp_size} != --cp_size {opts.cp_size}"

KV_SHARED = opts.d == 512
# d=512 is kv_shared: dk carries dK+dV (~4x the reduction length), so its reduction-
# order noise reaches ~4 ULP vs <=1 ULP for d<=256. Widen ONLY this run's dk/dv band
# to 6 ULP (others stay tight). setdefault so an explicit FM4_TOL_* still wins.
if KV_SHARED:
    os.environ.setdefault("FM4_TOL_DK", "6,1e-2,2e-4")
    os.environ.setdefault("FM4_TOL_DV", "6,1e-2,2e-4")
BS = _int_list(opts.bs)
S_LOCALS = _int_list(opts.s_locals)
HKS = _int_list(opts.hks)
MASKS = [m for m in opts.masks.split(",") if m]
for _m in MASKS:
    assert _m in ("document", "prefix_lm"), f"bad mask {_m!r}"
if not S_LOCALS:
    S_LOCALS = [opts.s_local]
    assert sum(doc for _, doc in DOC_PACKING) == opts.s_local * cp_size, (
        f"DOC_PACKING sums to {sum(d for _, d in DOC_PACKING)} != S_total {opts.s_local * cp_size} "
        f"(need s_local*cp == 32768)"
    )


def doc_packing(s_total):
    """DOC_PACKING as-is at 32768, else scaled to s_total with the same ratios."""
    full = sum(doc for _, doc in DOC_PACKING)
    if s_total == full:
        return DOC_PACKING
    packing = [(p * s_total // full, d * s_total // full) for p, d in DOC_PACKING]
    p, d = packing[-1]
    packing[-1] = (p, d + s_total - sum(d for _, d in packing))
    assert all(0 < p < d for p, d in packing), packing
    return packing


def make_mask(mask_type, b, s_total):
    arr, _ = GENERATORS[mask_type](b, s_total, s_total, 1, doc_packing(s_total))
    return paddle.to_tensor(arr)


def scat(full):
    return scatter_balance(full, group=cp_group, axis=1, mode="dual_chunk").contiguous()


def run_once(mask_type, hk, b, s_total, seed):
    paddle.seed(seed)
    np.random.seed(seed)
    full_q = paddle.randn([b, s_total, opts.hq, opts.d], dtype=paddle.bfloat16)
    full_k = paddle.randn([b, s_total, hk, opts.d], dtype=paddle.bfloat16)
    full_v = full_k if KV_SHARED else paddle.randn([b, s_total, hk, opts.d], dtype=paddle.bfloat16)
    for t in (full_q, full_k, full_v):
        t.stop_gradient = True

    q_ref, k_ref = scat(full_q), scat(full_k)
    v_ref = k_ref if KV_SHARED else scat(full_v)
    q, k = scat(full_q), scat(full_k)
    v = k if KV_SHARED else scat(full_v)
    for t in (q, k, v, q_ref, k_ref, v_ref):
        t.stop_gradient = False

    mask = make_mask(mask_type, b, s_total)

    out_ref = flashmask_attention_cp(q_ref, k_ref, v_ref, mask.clone(), mode="all-gather")
    out = overlap_flashmask_attention_fm4(q, k, v, mask, group=cp_group, mode="overlap")
    ok = check_accuracy(out, out_ref, "out", rank=rank)

    out.backward()
    out_ref.backward()
    ok = check_accuracy(q.grad, q_ref.grad, "dq", rank=rank) and ok
    ok = check_accuracy(k.grad, k_ref.grad, "dk", rank=rank) and ok
    if not KV_SHARED:
        ok = check_accuracy(v.grad, v_ref.grad, "dv", rank=rank) and ok
    return ok


def all_ranks_ok(local_ok):
    flag = paddle.to_tensor([1 if local_ok else 0], dtype="int32")
    paddle.distributed.all_reduce(flag, op=paddle.distributed.ReduceOp.MIN, group=cp_group)
    return bool(flag.numpy()[0])


def main():
    total = 0
    failed = []
    counter = 0
    # ONE mask per process (mask_type fixed by the launcher); S_local in the given
    # order, then cycle B x hk.
    for mask_type in MASKS:
        for s_local in S_LOCALS:
            for b in BS:
                for hk in HKS:
                    desc = (f"d={opts.d} hq={opts.hq} hk={hk} b={b} S_local={s_local} "
                            f"cp={cp_size} mask={mask_type}")
                    print(f"[r{rank}] >>> {desc}", flush=True)
                    local_ok = True
                    global_ok = True
                    for r in range(opts.repeats):
                        counter += 1
                        this = run_once(mask_type, hk, b, s_local * cp_size, opts.seed + counter * 10)
                        local_ok = local_ok and this
                        global_ok = all_ranks_ok(this) and global_ok
                    total += 1
                    tag = "PASS" if global_ok else "FAIL"
                    note = "" if local_ok else "  <- THIS RANK failed"
                    print(f"[r{rank}][{tag}] {desc}{note}", flush=True)
                    if not global_ok:
                        failed.append(desc)
    if rank == 0:
        print("=" * 60)
        s_desc = ",".join(map(str, S_LOCALS))
        print(f"CP={cp_size} d={opts.d} S_local={s_desc}: {total - len(failed)}/{total} configs passed.")
        for f in failed:
            print(f"  FAILED: {f}")
        print("=" * 60, flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
