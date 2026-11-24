#!/usr/bin/env bash
# Phase 2 step G10: the onboard herd detector (S7), the final recipe on D-FINE N, all three folds,
# 72 epochs each. HS_G10_RECIPE names the final recipe: g2 (configs/g2_recipe.yaml, with its hard
# negatives and touching-tile repeats, as in gpu/runpod/job_g2.sh) or phase1 (the reference). It
# has no default, so the recipe is always chosen on purpose after the leave-one-farm-out
# comparison. Each model is exported to ONNX; full-image predictions follow separately
# (job_fullscore.sh with HS_KINDS=g10n), and S7 is scored at home base, as for G4.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
RECIPE="${HS_G10_RECIPE:-}"
case "$RECIPE" in
    g2 | phase1) ;;
    *) echo "HS_G10_RECIPE must be g2 or phase1 (got '$RECIPE')"; exit 2 ;;
esac
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
EXTRA=(--recipe phase1)
if [ "$RECIPE" = g2 ]; then
    HN=$(package_dir "$PKGROOT" herdsight_hardneg_v1) || { echo "hard-negative package not found"; exit 3; }
    python - "$HN/manifest.json" "$REPO/results/phase2/herdsight_hardneg_v1_manifest.json" <<'PY' || exit 3
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
keys = ("name", "summary", "images", "training_exclusions_sha256")
if any(a[k] != b[k] for k in keys):
    sys.exit("hard-negative package on Kaggle differs from the manifest in the repository")
print("hard-negative package matches the repository manifest")
PY
    K=$(python - "$REPO/configs/g2_recipe.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["oversample_touching"]["value"])
PY
)
    EXTRA=(--recipe g2 --oversample-touching "$K" --hardneg "$HN"
           --exclude "$REPO/results/data/training_exclusions.json")
fi
python -m pip install -q "onnxruntime-gpu==1.22.0" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
    || echo "WARNING: GPU ONNX Runtime not installed, CPU scoring"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "recipe: $RECIPE, usable CPUs: $NCPU, data-loading workers: $WORKERS"
for FOLD in fold0 fold1 fold2; do
    RUN="g10n_${FOLD}"
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$PKG" --fold "$FOLD" --size n \
        --epochs 72 "${EXTRA[@]}" --workers "$WORKERS" --work "$WORK" \
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
echo "G10 session done (recipe $RECIPE)"
