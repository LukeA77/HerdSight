#!/usr/bin/env bash
# Phase 2 step G7: RTMDet-R, the first oriented family, on all three folds with the authors'
# 36-epoch schedule, then out-of-fold full-image predictions. Needs the C6 eye check to have passed
# (scripts/make_obb_tiles.py refuses otherwise). The stack needs CUDA 11.8 PyTorch, so a host whose
# GPU cannot run it exits with code 10. Test-farm images are never read. Scoring stays at home base.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
export PYTHONPATH="$REPO/python"
export HERDSIGHT_STORAGE=/root/hs_storage
LABELS_TAG="${HS_LABELS_TAG:-c6-4}"
ZIP_MD5=c18911fb11cb58741b9cfa16516ca8cd
RAW="$HERDSIGHT_STORAGE/datasets/raw/icaerus_grazing_cows_v2"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
BUDGET_S=${HS_JOB_SECONDS:-21000}  # leaves time to upload whatever is finished
T0=$SECONDS
mkdir -p "$RAW/unpacked" "$WORK" "$OUT/predictions"

URL="https://zenodo.org/records/11048412/files/Cattle_drone_images_042024.zip?download=1"
RATE=$(curl -sL -r 0-40000000 --max-time 30 -o /dev/null -w '%{speed_download}' "$URL" | cut -d. -f1)
echo "Zenodo speed on this host: ${RATE:-0} bytes/s"
if [ "${RATE:-0}" -lt "${HS_MIN_RATE:-1500000}" ]; then
    echo "host network too slow for the photos; not running the job"
    exit 10
fi

# The pinned stack (rehearsed on CPU first). mmrotate 1.0.0rc1 needs Python 3.9.
TOOL=/root/g7_tool
ENVDIR=/root/g7_env
export UV_PYTHON_INSTALL_DIR=/root/g7_python UV_CACHE_DIR=/root/g7_cache
python -m venv "$TOOL" && "$TOOL/bin/pip" install -q uv==0.5.31 || { echo "uv failed"; exit 3; }
UV="$TOOL/bin/uv"
"$UV" venv --python 3.9 "$ENVDIR" || { echo "Python 3.9 failed"; exit 3; }
PY="$ENVDIR/bin/python"
"$UV" pip install --python "$PY" "torch==2.0.1" "torchvision==0.15.2" \
    --index-url https://download.pytorch.org/whl/cu118 || { echo "torch failed"; exit 3; }
"$UV" pip install --python "$PY" "numpy==1.26.4" "setuptools==69.5.1" \
    "opencv-python-headless==4.9.0.80" "mmengine==0.10.4" pyyaml pillow \
    || { echo "base packages failed"; exit 3; }
"$UV" pip install --python "$PY" "mmcv==2.0.1" --only-binary mmcv \
    --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
    || { echo "mmcv failed"; exit 3; }
"$UV" pip install --python "$PY" "mmdet==3.0.0" "mmrotate==1.0.0rc1" \
    || { echo "mmdet or mmrotate failed"; exit 3; }
"$UV" pip freeze --python "$PY" > "$OUT/g7_packages.txt"

# The authors' box loss runs only on CUDA, so check it here before any long step.
"$PY" - <<'PY' || { echo "this GPU cannot run the stack"; exit 10; }
import torch
from mmrotate.models.losses import RotatedIoULoss
assert torch.cuda.is_available()
a = torch.tensor([[50.0, 50.0, 40.0, 15.0, 0.4]], device="cuda", requires_grad=True)
b = torch.tensor([[52.0, 49.0, 38.0, 16.0, 0.5]], device="cuda")
loss = RotatedIoULoss(mode="linear")(a, b)
loss.backward()
assert torch.isfinite(loss)
print("GPU check passed:", torch.cuda.get_device_name(0), "loss", round(float(loss), 4))
PY

download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
USER_REF=$(kaggle datasets list --mine --csv | tail -n +2 | head -1 | cut -d/ -f1)
[ -n "$USER_REF" ] || { echo "Kaggle account not readable"; exit 3; }
mkdir -p /root/hs_labels
kaggle datasets download "$USER_REF/herdsight-rp-$LABELS_TAG" -p /root/hs_labels --unzip -q \
    || { echo "SAM label download failed"; exit 3; }
python -m pip install -q numpy pyyaml || { echo "label tools failed"; exit 3; }
python "$REPO/scripts/make_obb_tiles.py" --labels /root/hs_labels/labels/labels.jsonl \
    --package "$PKG" --out "$WORK/obb" || { echo "oriented labels refused"; exit 4; }

NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 10 ? 8 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS"
for K in 0 1 2; do
    RUN="g7_fold$K"
    LEFT=$((BUDGET_S - (SECONDS - T0)))
    [ "$LEFT" -gt 60 ] || { echo "time budget used before $RUN"; exit 7; }
    timeout "$LEFT" "$PY" "$REPO/gpu/train_rtmdet.py" --run "$RUN" --labels "$WORK/obb" \
        --images "$PKG/images" --fold "fold$K" --work "$WORK" --workers "$WORKERS" \
        > "$OUT/${RUN}_train.txt" 2>&1 || { echo "failed: $RUN"; tail -20 "$OUT/${RUN}_train.txt"; exit 5; }
    mkdir -p "$OUT/$RUN"
    cp "$WORK/$RUN/last.pth" "$WORK/$RUN/config.py" "$WORK/$RUN/run.json" "$OUT/$RUN/"
    echo "$RUN trained $(date -u +%FT%TZ)"
done

GOT=1
for ATTEMPT in $(seq 1 12); do
    if curl -fL --retry 3 --retry-delay 15 -C - -o "$RAW/photos.zip" "$URL"; then
        GOT=0
        break
    fi
    echo "download attempt $ATTEMPT stopped; resuming"
    sleep 20
done
[ "$GOT" -eq 0 ] || { echo "photo download failed"; exit 3; }
[ "$(md5sum "$RAW/photos.zip" | cut -d' ' -f1)" = "$ZIP_MD5" ] \
    || { echo "photo archive MD5 differs"; exit 3; }
python -m zipfile -e "$RAW/photos.zip" "$RAW/unpacked" || { echo "unzip failed"; exit 3; }
rm -f "$RAW/photos.zip"
echo "photos ready after $((SECONDS - T0)) s"

LEFT=$((BUDGET_S - (SECONDS - T0)))
RC=0
if [ "$LEFT" -le 60 ]; then
    echo "time budget used before predicting"
    RC=7
else
    timeout "$LEFT" "$PY" "$REPO/gpu/predict_rtmdet.py" --models "$WORK" --device cuda || RC=5
fi
cp -r "$HERDSIGHT_STORAGE/runs/baselines/rtmdet_r" "$OUT/predictions/" 2>/dev/null
echo "G7 finished with code $RC"
exit "$RC"
