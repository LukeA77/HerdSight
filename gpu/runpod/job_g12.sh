#!/usr/bin/env bash
# Phase 2, dark-coat, tight-group and outline training under leave-one-farm-out (after D-49): the
# reference recipe plus coat darkening inside each animal's oval, animals pasted along their ovals
# as touching groups, and mild whole-scene darkening (train_dfine.py --recipe g12). 72 epochs per
# held-out farm, as G6, so it is compared with the reference's leave-one-farm-out models. Ovals
# come from the C6 rotated boxes (session c6-4 on Kaggle) and are built per training set, so a
# held-out farm's animals are never read (gpu/make_tile_ovals.py). Test-farm images are never
# read. Full-image predictions follow separately (job_fullscore.sh, HS_LOFO_PREFIX=g12).
# HS_SEED (default 0) names repeats g12_s<seed>_<farm>; HS_FARMS limits the farms.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
LOFO=/root/hs_lofo
C6=/root/hs_c6
C6_TAG=c6-4
SEED=${HS_SEED:-0}
SUFFIX=""
[ "$SEED" != 0 ] && SUFFIX="_s$SEED"
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
USER_REF=$(cut -d/ -f1 /tmp/mine.csv | head -1)
[ -n "$USER_REF" ] || { echo "Kaggle account not readable"; exit 3; }
mkdir -p "$C6"
kaggle datasets download "$USER_REF/herdsight-rp-$C6_TAG" -p "$C6" --unzip -q \
    || { echo "C6 label download failed"; exit 3; }
LABELS=$(find "$C6" -name labels.jsonl | head -1)
[ -n "$LABELS" ] || { echo "C6 labels.jsonl not found"; exit 3; }
python "$REPO/gpu/make_lofo_sets.py" --package "$PKG" --out "$LOFO" || exit 3
python -m pip install -q "onnxruntime-gpu==1.22.0" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
    || echo "WARNING: GPU ONNX Runtime not installed, CPU scoring"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS, seed: $SEED"
for FARM in ${HS_FARMS:-Derval Jalogny Mauron}; do
    RUN="g12${SUFFIX}_${FARM}"
    OVALS="$LOFO/ovals_lofo_$FARM.json"
    python "$REPO/gpu/make_tile_ovals.py" --labels "$LABELS" \
        --coco "$LOFO/train_lofo_$FARM.json" --out "$OVALS" || exit 3
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$LOFO" --fold "lofo_$FARM" \
        --epochs 72 --recipe g12 --ovals "$OVALS" --seed "$SEED" --workers "$WORKERS" \
        --work "$WORK" --dfine /root/D-FINE || exit 4
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
    python "$REPO/gpu/eval_tiles.py" --onnx "$WORK/$RUN/last.onnx" \
        --ann "$LOFO/monitor_lofo_$FARM.json" --images "$LOFO/images" \
        --out "$OUT/${RUN}_monitor_eval.json" || exit 5
    echo "$RUN done $(date -u +%FT%TZ)"
done
echo "G12 session done (seed $SEED)"
