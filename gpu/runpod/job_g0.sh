#!/usr/bin/env bash
# Phase 2 step G0 (H9): does the authors' full 220-epoch schedule beat the 72-epoch one?
# Fold 0's training data, with one farm-date group (configs/phase2.yaml: g0_holdout_group) held
# out for scoring; never fold 0's own validation images. Same recipe, data and seed for both;
# only the number of epochs differs. Each model is scored on the hold-out tiles by
# gpu/eval_tiles.py; the comparison is written to g0_summary.json.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
# packages and training files on the container's local disk (fast); outputs go to $OUT
PKGROOT=/root/hs_packages
WORK=/root/hs_work
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "package not found"; exit 3; }
python -m pip install -q onnxruntime-gpu || echo "WARNING: onnxruntime-gpu not installed, CPU scoring"
EXCLUDE="$REPO/results/data/training_exclusions.json"
read -r HOLDOUT H9MIN < <(python - "$REPO/configs/phase2.yaml" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
print(c["g0_holdout_group"]["value"], c["hypotheses"]["value"]["H9"]["centre_recall_gain_min"])
PY
)
echo "hold-out group: $HOLDOUT"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS"
for EP in 72 220; do
    RUN="g0_e${EP}"
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$PKG" --fold fold0 \
        --epochs "$EP" --recipe phase1 --exclude "$EXCLUDE" --holdout-group "$HOLDOUT" \
        --workers "$WORKERS" --work "$WORK" --dfine /root/D-FINE || exit 4
    python "$REPO/gpu/eval_tiles.py" --onnx "$WORK/$RUN/last.onnx" \
        --ann "$WORK/$RUN/holdout.json" --images "$PKG/images" --out "$OUT/${RUN}_eval.json" \
        || exit 5
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
done
python - "$OUT" "$H9MIN" <<'PY'
import json, os, sys
out, h9min = sys.argv[1], float(sys.argv[2])
a = json.load(open(os.path.join(out, "g0_e72_eval.json")))
b = json.load(open(os.path.join(out, "g0_e220_eval.json")))
keys = ("centre_matched_recall", "centre_matched_precision", "centre_ap", "upright_ap50",
        "count_mae_per_tile")
s = {k: {"e72": a[k], "e220": b[k], "e220_minus_e72": b[k] - a[k]} for k in keys}
s["H9_rule"] = f"held if e220 centre-matched recall is at least {h9min} above e72 (configs/phase2.yaml)"
s["H9_held"] = s["centre_matched_recall"]["e220_minus_e72"] >= h9min
json.dump(s, open(os.path.join(out, "g0_summary.json"), "w"), indent=1)
print(json.dumps(s, indent=1))
PY
