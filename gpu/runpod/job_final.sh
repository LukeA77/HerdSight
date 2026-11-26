#!/usr/bin/env bash
# Phase 7: the two final models, trained once on every cross-validation tile (all three folds) and
# never on the test farm, which is not in the tile packages.
#   final_s      home-base detector: D-FINE S, reference recipe (D-50), seed 0, 72 epochs (D-22),
#                full-resolution tiles (herdsight_tiles_v1)
#   final_n1920  onboard herd detector: D-FINE N, same recipe, stills reduced to a 1,920 px long
#                side before tiling (herdsight_onboard_tiles_1920_v1, D-51 record)
# The checkpoint is the last epoch (no selection), as in every Phase 2 run. monitor_final.json is
# fold 0's monitor set, which is inside the training data here, so its log is a training check
# only, not a result. The test farm is scored once at home base after the freeze (Phase 7).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
download_tiles "$PKGROOT" || exit 3
python -m pip install -q "onnxruntime-gpu==1.22.0" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
    || echo "WARNING: GPU ONNX Runtime not installed"
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS"
for SPEC in "final_s s herdsight_tiles_v1" "final_n1920 n herdsight_onboard_tiles_1920_v1"; do
    read -r RUN SIZE NAME <<< "$SPEC"
    PKG=$(package_dir "$PKGROOT" "$NAME") || { echo "tile package $NAME not found"; exit 3; }
    python - "$REPO/gpu" "$PKG" <<'PY' || exit 3
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from make_lofo_sets import coco, union
pkg = Path(sys.argv[2])
images, anns, cats = union(pkg)
(pkg / "train_final.json").write_text(json.dumps(coco(sorted(images), images, anns, cats)))
shutil.copy(pkg / "monitor_fold0.json", pkg / "monitor_final.json")
print(f"{pkg.name}: {len(images)} tiles, {sum(map(len, anns.values()))} boxes in train_final")
PY
    python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$PKG" --fold final --size "$SIZE" \
        --epochs 72 --recipe phase1 --seed 0 --workers "$WORKERS" --work "$WORK" \
        --dfine /root/D-FINE || exit 4
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
    echo "$RUN done $(date -u +%FT%TZ)"
done
echo "final models done"
