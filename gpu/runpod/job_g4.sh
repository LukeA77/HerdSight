#!/usr/bin/env bash
# Phase 2 step G4: D-FINE M on the best recipe so far, all three folds, 72 epochs each. G2 did not
# replace the reference (results/phase2/g2_comparison.json), so the best recipe is the reference.
# Each model is exported to ONNX; full-image predictions and the paired comparison with the
# same-recipe D-FINE S models follow separately, as for G2.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
python -m pip install -q "onnxruntime-gpu==1.22.0" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
    || echo "WARNING: GPU ONNX Runtime not installed, CPU scoring"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS"
for FOLD in fold0 fold1 fold2; do
    RUN="g4m_${FOLD}"
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$PKG" --fold "$FOLD" --size m \
        --epochs 72 --recipe phase1 --workers "$WORKERS" --work "$WORK" \
        --dfine /root/D-FINE || exit 4
    python "$REPO/gpu/eval_tiles.py" --onnx "$WORK/$RUN/last.onnx" \
        --ann "$PKG/monitor_${FOLD}.json" --images "$PKG/images" \
        --out "$OUT/${RUN}_monitor_eval.json" || exit 5
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
    echo "$RUN done $(date -u +%FT%TZ)"
done
echo "G4 session done"
