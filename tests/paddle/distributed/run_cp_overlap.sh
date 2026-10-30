#!/bin/bash
# FM-4 context-parallel overlap accuracy test launcher (ONE config per launch).
#
# Launch scaffolding kept from dist_flashmask_dev/run_fm4_cp_test2.sh (the PADDLE_*
# env unset dance, NCCL env, TARGET_RANKS node selection and hostfile-derived
# master are what make rendezvous work on this cluster). One config per process,
# because the overlap singleton must not carry state across configs.
#
# Usage (single mask):
#   CP_SIZE=4 D=128 MASKS=document bash run_cp_overlap.sh
# Run test_cp_overlap_launch.py (pytest) to cover the small default matrix, or
# run_cp_overlap_sweep.sh for the full document+prefix_lm x headdim matrix (it
# launches one process per mask, which is the only safe way to cover both).
#
# Env:
#   CP_SIZE       context-parallel size = launched world size (default 8)
#   D HQ HKS BS S_LOCAL MASKS REPEATS   one mask per launch; B/hk cycled in-process
#   TARGET_RANKS  PADDLE_TRAINER_IDs that participate (default "0"); count = nnodes
#   DEVICES       GPUs per node (default: single-node -> 0..CP_SIZE-1, else all)
#   PY_BIN        python interpreter (default: python on PATH)
#   MASTER PORT HOSTFILE   rendezvous overrides (single node defaults to localhost)

mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
node_rank=$((mpi_rank+offset))
mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
nnode_train=${nnode_set:-${mpi_node}}
master_train=${master:-localhost}

echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
set -x

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
nnodes=$PADDLE_TRAINERS_NUM
rank=$PADDLE_TRAINER_ID

for name in `env | grep -E 'PADDLE|ENDPOINT' | awk -F'=' '{print $1}'`; do
  unset ${name}
done

export NCCL_DEBUG=INFO
unset GLOG_vmodule GLOG_v
export PYTHONUNBUFFERED=1
export FLAGS_use_auto_growth_pinned_allocator=True

export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_NVLS_ENABLE=0
export NCCL_IB_ADAPTIVE_ROUTING=1

export PADDLE_PG_TIMEOUT=150000
export CUDA_MODULE_LOADING=LAZY

# release stale shared memory
find /dev/shm/ -type f -name "paddle_*" -print0 | xargs -0 rm -f

# CUDA forward-compat driver libs: on CUDA>12 hosts the overlap TMA bridge needs
# the compat libcuda to resolve cuTensorMapEncodeTiled, else it dies with 715.
cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | awk -F'.' '{print $1}')
if [ "${cuda_version}" != "12" ]; then
    export LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}
fi

# node selection: only the listed PADDLE_TRAINER_IDs participate; count = nnodes.
TARGET_RANKS=(${TARGET_RANKS:-0})

is_target=false
for i in "${!TARGET_RANKS[@]}"; do
    if [[ $rank -eq ${TARGET_RANKS[$i]} ]]; then
        is_target=true
        new_rank=$i
        break
    fi
done

if [[ "$is_target" = false ]]; then
    echo "[run_cp_overlap] SKIP: rank=$rank not in TARGET_RANKS=(${TARGET_RANKS[*]}); nothing to run on this node."
    exit 0
fi

rank=$new_rank
nnodes=${#TARGET_RANKS[@]}

first_rank=${TARGET_RANKS[0]}
line_num=$((first_rank + 1))

# master endpoint: single node just uses localhost; multi-node derives it from a
# hostfile. Override with MASTER=<ip> / PORT=<port>, or point HOSTFILE elsewhere.
if [ "$nnodes" -eq 1 ]; then
    master=${MASTER:-localhost}
else
    hostfile=${HOSTFILE:-/root/paddlejob/workspace/hostfile}
    master=${MASTER:-$(cat "$hostfile" | head -n "${line_num}" | tail -n 1 | awk '{print $1}')}
fi
port=${PORT:-12348}

export FLAGS_shard_bypass_dygraph_optimizer=1

# FA-v4 in-library SM100 path + accuracy determinism.
export FLAGS_flash_attn_version=4
export FLAGS_use_cinn=True
export FLAGS_cudnn_deterministic=1
export FLAGS_call_stack_level=2

CP_SIZE=${CP_SIZE:-8}
PY_BIN=${PY_BIN:-python}

# One (headdim, CP, mask) per launch. Inside the process the mask is FIXED and
# batch x hk are cycled (validated safe). S_local defaults so S_total=32768 (doc 4
# fits exactly, no scaling).
D=${D:-128}
S_LOCAL=${S_LOCAL:-$((32768 / CP_SIZE))}
BS=${BS:-1,2}
HQ=${HQ:-16}
HKS=${HKS:-16,4,1}
# ONE mask per launch. The overlap singleton carries dK/dV state across calls, so
# switching mask *content* mid-process (e.g. document -> prefix_lm) leaves a stale
# dK/dV buffer and corrupts the second mask's gradients. Default to a single mask;
# use run_cp_overlap_sweep.sh (one process per mask) for the full matrix. Passing a
# comma-joined MASKS still runs, but is unsafe for exactly this reason.
MASKS=${MASKS:-document}
REPEATS=${REPEATS:-2}

cd "$(dirname "${BASH_SOURCE[0]}")"

# device selection: single node uses the first CP_SIZE GPUs; multi-node uses all.
if [ -n "${DEVICES:-}" ]; then
    device_arg="--devices ${DEVICES}"
elif [ "$nnodes" -eq 1 ]; then
    device_arg="--devices $(seq -s, 0 $((CP_SIZE - 1)))"
else
    device_arg=""
fi

logdir="log_cp${CP_SIZE}-d${D}-${MASKS}-r$rank"
echo "[run_cp_overlap] CP=$CP_SIZE d=$D S_local=$S_LOCAL bs=$BS hq=$HQ hks=$HKS masks=$MASKS nnodes=$nnodes rank=$rank"

$PY_BIN -m paddle.distributed.launch \
    --log_dir "$logdir" \
    --master "$master:$port" \
    --nnodes "$nnodes" \
    --rank "$rank" \
    --run_mode=collective \
    $device_arg \
    cp_overlap_main.py \
        --cp_size "$CP_SIZE" --d "$D" --s_local "$S_LOCAL" --bs "$BS" --hq "$HQ" --hks "$HKS" \
        --masks "$MASKS" --repeats "$REPEATS"
