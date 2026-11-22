#!/usr/bin/env bash
# Phase 2 step G2: the combined recipe (configs/g2_recipe.yaml) against the reference recipe,
# all three folds, 72 epochs each, on one GPU type (D-35). G2 runs first, so if the session has
# to stop early the candidate is complete. Each model is exported to ONNX; full-image
# predictions and the paired comparison are made at home base (scripts/run_dfine_baselines.py
# and the scoring scripts), as in Phase 1. A quick look on each fold's monitor tiles is written
# here too (gpu/eval_tiles.py); it is not the comparison.
# HS_G2_RUNS (optional) lists which to train, default "g2 ref".
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
HN=$(package_dir "$PKGROOT" herdsight_hardneg_v1) || { echo "hard-negative package not found"; exit 3; }
python - "$HN/manifest.json" "$REPO/results/phase2/herdsight_hardneg_v1_manifest.json" <<'PY' || exit 3
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:])
keys = ("name", "summary", "images", "training_exclusions_sha256")
if any(a[k] != b[k] for k in keys):
    sys.exit("hard-negative package on Kaggle differs from the manifest in the repository")
print("hard-negative package matches the repository manifest")
PY
python -m pip install -q onnxruntime-gpu || echo "WARNING: onnxruntime-gpu not installed, CPU scoring"
EXCLUDE="$REPO/results/data/training_exclusions.json"
K=$(python - "$REPO/configs/g2_recipe.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["oversample_touching"]["value"])
PY
)
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS, touching-tile repeats: $K"
for KIND in ${HS_G2_RUNS:-g2 ref}; do
    for FOLD in fold0 fold1 fold2; do
        RUN="${KIND}_${FOLD}"
        if [ "$KIND" = g2 ]; then
            EXTRA=(--recipe g2 --oversample-touching "$K" --hardneg "$HN" --exclude "$EXCLUDE")
        else
            EXTRA=(--recipe phase1)
        fi
        python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$PKG" --fold "$FOLD" \
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
done
echo "G2 session done"
