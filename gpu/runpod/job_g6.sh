#!/usr/bin/env bash
# Phase 2 step G6 (D-33): leave-one-farm-out models of the reference recipe, 72 epochs each, one
# per held-out farm (gpu/make_lofo_sets.py). Derval first: it is the farm Phase 1 lost most on.
# Each model is exported to ONNX; full-image predictions of its held-out farm are made afterwards.
# HS_SEED (default 0) repeats the runs with another training seed, named g6ref_s<seed>_<farm> so
# they never overwrite seed 0; HS_FARMS limits the farms (default all three, Derval first).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
LOFO=/root/hs_lofo
SEED=${HS_SEED:-0}
SUFFIX=""
[ "$SEED" != 0 ] && SUFFIX="_s$SEED"
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
python "$REPO/gpu/make_lofo_sets.py" --package "$PKG" --out "$LOFO" || exit 3
python -m pip install -q "onnxruntime-gpu==1.22.0" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
    || echo "WARNING: GPU ONNX Runtime not installed, CPU scoring"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS"
for FARM in ${HS_FARMS:-Derval Jalogny Mauron}; do
    RUN="g6ref${SUFFIX}_${FARM}"
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$LOFO" --fold "lofo_$FARM" \
        --epochs 72 --recipe phase1 --seed "$SEED" --workers "$WORKERS" --work "$WORK" \
        --dfine /root/D-FINE || exit 4
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
    python "$REPO/gpu/eval_tiles.py" --onnx "$WORK/$RUN/last.onnx" \
        --ann "$LOFO/monitor_lofo_$FARM.json" --images "$LOFO/images" \
        --out "$OUT/${RUN}_monitor_eval.json" || exit 5
    echo "$RUN done $(date -u +%FT%TZ)"
done
echo "G6 reference session done (seed $SEED)"
