#!/bin/bash
# FM-4 overlap accuracy sweep -- all-in-one initial-interception gate.
#
# Runs, with NO env needed, one process per (CP, mask_type, headdim):
#   CP in {4, 8}  x  mask in {document, prefix_lm}  x  headdim in {128, 256, 512}
# = 12 launches (6 per CP). Each process fixes the mask and cycles batch x hk in
# process (2 reps). S_local = 32768/CP so S_total=32768 (the DOC_PACKING fits exactly).
# CP16 needs two nodes and is intentionally left out of this single-node gate.
#
# One process per mask keeps the launches independent; switching mask content
# mid-process no longer corrupts dK/dV (fixed in 647d699).
#
# Usage (single node, just run it):
#   bash run_cp_overlap_sweep.sh
# Overridable: CP_SIZES="4" HEADDIMS="128 256" MASK_TYPES="document" ...
#
# TARGET_RANKS defaults to this node's PADDLE_TRAINER_ID; the inner launcher skips
# (exit 0) on any other node, so nothing runs there.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

CP_SIZES=${CP_SIZES:-"4 8"}
TARGET_RANKS=${TARGET_RANKS:-${PADDLE_TRAINER_ID:-0}}
MASK_TYPES=${MASK_TYPES:-"document prefix_lm"}
HEADDIMS=${HEADDIMS:-"128 256 512"}
PY_BIN=${PY_BIN:-}
DEVICES=${DEVICES:-}

# kv-head counts per head dim: d=512 -> MQA + shared-KV only; else MHA/GQA/MQA
hks_for() { case "$1" in 512) echo "1" ;; *) echo "16,4,1" ;; esac; }

overall=0
for cp in $CP_SIZES; do
    for m in $MASK_TYPES; do
        for d in $HEADDIMS; do
            echo "======================= CP=$cp mask=$m d=$d ======================="
            if TARGET_RANKS="$TARGET_RANKS" PY_BIN="$PY_BIN" DEVICES="$DEVICES" CP_SIZE="$cp" \
               D="$d" HQ=16 HKS="$(hks_for "$d")" MASKS="$m" bash run_cp_overlap.sh; then
                :
            else
                overall=1
                echo "== FAILED: CP=$cp mask=$m d=$d =="
            fi
        done
    done
done

exit $overall
