#!/bin/bash
# FM-4 CP overlap accuracy test launcher (torch, ONE config per launch).
#
# Single-node torchrun launcher mirroring run_cp_overlap.sh. One config per
# process because the overlap singleton must not carry state across configs.
# Single-node launcher only: hierarchical (multi-node) overlap is not exercised here.
#
# Usage (single mask):
#   CP_SIZE=4 D=128 MASKS=document bash run_cp_overlap.sh
# Run test_cp_overlap_launch.py (pytest) to cover the small default matrix, or
# run_cp_overlap_sweep.sh for the full document+prefix_lm x headdim matrix (it
# launches one process per mask, which is the only safe way to cover both).
#
# Env:
#   CP_SIZE   context-parallel size = launched world size = GPUs used (default 8)
#   D HQ HKS BS S_LOCAL MASKS REPEATS   one mask per launch; B/hk cycled in-process
#   DEVICES   GPUs to use (default: first CP_SIZE, i.e. 0..CP_SIZE-1)
#   PY_BIN    python interpreter (default: python on PATH)
#   PORT      torchrun rendezvous port (default 29500)
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
export NCCL_IB_ADAPTIVE_ROUTING=1
export CUDA_MODULE_LOADING=LAZY

# CUDA forward-compat driver libs: on CUDA>12 hosts the overlap TMA bridge needs
# the compat libcuda to resolve cuTensorMapEncodeTiled, else it dies with 715.
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | awk -F'.' '{print $1}')
if [ "${cuda_version}" != "12" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
fi

CP_SIZE=${CP_SIZE:-8}
PY_BIN=${PY_BIN:-python}
PORT=${PORT:-29500}

# One (headdim, CP, mask) per launch; batch x hk cycled in-process. S_local defaults
# so S_total=32768 (the DOC_PACKING fits exactly, no scaling).
D=${D:-128}
S_LOCAL=${S_LOCAL:-$((32768 / CP_SIZE))}
BS=${BS:-1,2}
HQ=${HQ:-16}
HKS=${HKS:-16,4,1}
# ONE mask per launch by default; run_cp_overlap_sweep.sh covers the full matrix.
# Switching mask content mid-process no longer corrupts dK/dV (fixed in 647d699).
MASKS=${MASKS:-document}
REPEATS=${REPEATS:-2}

# device selection: default the first CP_SIZE GPUs.
if [ -n "${DEVICES:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${DEVICES}"
else
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((CP_SIZE - 1)))"
fi

logdir="log_cp${CP_SIZE}-d${D}-${MASKS}"
echo "[run_cp_overlap] CP=$CP_SIZE d=$D S_local=$S_LOCAL bs=$BS hq=$HQ hks=$HKS masks=$MASKS devices=$CUDA_VISIBLE_DEVICES"

$PY_BIN -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node "$CP_SIZE" \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:$PORT" \
    --log_dir "$logdir" \
    --redirects 3 --tee 3 \
    cp_overlap_main.py \
        --cp_size "$CP_SIZE" --d "$D" --s_local "$S_LOCAL" --bs "$BS" --hq "$HQ" --hks "$HKS" \
        --masks "$MASKS" --repeats "$REPEATS"
