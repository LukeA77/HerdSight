#!/usr/bin/env bash
# Phase 2 step G3 (D-34): the three mining models for nested hard-negative mining.
# Each model trains on ONE fold's tiles at half the screening schedule (configs/phase2.yaml,
# mining_epochs); they only find confusing objects and are never scored or used as detectors.
# The model trained on fold a later mines folds b and c on the home-base machine
# (scripts/mine_hard_negatives.py), so the hard negatives used when training fold k's model
# always come from models that never saw fold k.
# Training tiles come from train_<other>.json, which holds fold a's tiles; each model is
# monitored on that other fold's monitor tiles, which it never trains on.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "package not found"; exit 3; }
EXCLUDE="$REPO/results/data/training_exclusions.json"
EPOCHS=$(python - "$REPO/configs/phase2.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["mining_epochs"]["value"])
PY
)
echo "mining epochs: $EPOCHS"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS"
# mining model for fold a: (file holding fold a's tiles, fold a)
for PAIR in "fold1 fold0" "fold2 fold1" "fold0 fold2"; do
    read -r FILE_FOLD ONLY <<< "$PAIR"
    RUN="g3_mine_${ONLY}"
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$PKG" --fold "$FILE_FOLD" \
        --only-fold "$ONLY" --epochs "$EPOCHS" --recipe phase1 --exclude "$EXCLUDE" \
        --workers "$WORKERS" --work "$WORK" --dfine /root/D-FINE || exit 4
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
done
echo "mining models done"
