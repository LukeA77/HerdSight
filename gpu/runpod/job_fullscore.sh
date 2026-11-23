#!/usr/bin/env bash
# Phase 2: full-image, out-of-fold predictions of a session's models on one GPU, so recipes and
# model sizes are compared on the same footing (D-35). Only the cross-validation
# folds are predicted; the test farms are never run. Scoring stays at home base
# (scripts/phase2_compare.py). The photos come from their public record (Zenodo 11048412, CC BY
# 4.0) and are checked against its MD5. HS_MODELS_TAG names the session holding the models and
# HS_KINDS the model name prefixes in it (for example "g2 ref", or "g4m"). HS_LOFO_TAG, if set,
# names a session of leave-one-farm-out models (g6ref_<farm>): each predicts only its held-out
# farm, into runs/baselines/dfine_g6ref_<farm>.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO/python"
export HERDSIGHT_STORAGE=/root/hs_storage
TAG="${HS_MODELS_TAG:-g2-3}"
KINDS="${HS_KINDS:-g2 ref}"
FIRST_KIND="${KINDS%% *}"
LOFO_TAG="${HS_LOFO_TAG:-}"
LOFO_MODELS=/root/hs_models_lofo
FARMS="Derval Jalogny Mauron"
ZIP_MD5=c18911fb11cb58741b9cfa16516ca8cd
RAW="$HERDSIGHT_STORAGE/datasets/raw/icaerus_grazing_cows_v2"
MODELS=/root/hs_models
BUDGET_S=${HS_JOB_SECONDS:-15000}  # leaves time to upload whatever is finished
T0=$SECONDS
mkdir -p "$RAW/unpacked" "$MODELS" "$OUT/predictions"

URL="https://zenodo.org/records/11048412/files/Cattle_drone_images_042024.zip?download=1"
# Some rented hosts reach Zenodo at about 0.1 MB/s (session g2-score-1), which would take days.
# Measure first; a slow host is not worth holding (exit code 10: no wait before shutting down).
RATE=$(curl -sL -r 0-40000000 --max-time 30 -o /dev/null -w '%{speed_download}' "$URL" | cut -d. -f1)
echo "Zenodo speed on this host: ${RATE:-0} bytes/s"
if [ "${RATE:-0}" -lt "${HS_MIN_RATE:-1500000}" ]; then
    echo "host network too slow for the photos; not running the job"
    exit 10
fi
python -m pip install -q "onnxruntime-gpu==1.22.0" nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*" \
    pyyaml pillow numpy || { echo "could not install ONNX Runtime"; exit 3; }
LIBS=$(python - <<'PY'
import os, site
found = set()
for root in site.getsitepackages():
    for d, _, files in os.walk(root):
        if any(f.startswith(("libcublasLt.so.12", "libcudnn.so.9")) for f in files):
            found.add(d)
print(":".join(sorted(found)))
PY
)
export LD_LIBRARY_PATH="${LIBS}:${LD_LIBRARY_PATH:-}"

USER_REF=$(kaggle datasets list --mine --csv | tail -n +2 | head -1 | cut -d/ -f1)
[ -n "$USER_REF" ] || { echo "Kaggle account not readable"; exit 3; }
kaggle datasets download "$USER_REF/herdsight-rp-$TAG" -p "$MODELS" --unzip -q \
    || { echo "model download failed"; exit 3; }
python - "$MODELS" $KINDS <<'PY' || exit 4
import hashlib, json, sys
root = sys.argv[1]
for kind in sys.argv[2:]:
    for f in range(3):
        d = f"{root}/{kind}_fold{f}"
        want = json.load(open(f"{d}/run.json"))["onnx_sha256"]
        if hashlib.sha256(open(f"{d}/last.onnx", "rb").read()).hexdigest() != want:
            sys.exit(f"model hash differs: {d}")
print("model hashes match:", " ".join(sys.argv[2:]))
PY
if [ -n "$LOFO_TAG" ]; then
    mkdir -p "$LOFO_MODELS"
    kaggle datasets download "$USER_REF/herdsight-rp-$LOFO_TAG" -p "$LOFO_MODELS" --unzip -q \
        || { echo "leave-one-farm-out model download failed"; exit 3; }
    python - "$LOFO_MODELS" $FARMS <<'PY' || exit 4
import hashlib, json, sys
root = sys.argv[1]
for farm in sys.argv[2:]:
    d = f"{root}/g6ref_{farm}"
    want = json.load(open(f"{d}/run.json"))["onnx_sha256"]
    if hashlib.sha256(open(f"{d}/last.onnx", "rb").read()).hexdigest() != want:
        sys.exit(f"model hash differs: {d}")
print("leave-one-farm-out model hashes match")
PY
fi
python - "$MODELS/${FIRST_KIND}_fold0/last.onnx" <<'PY' || { echo "ONNX Runtime cannot use the GPU"; exit 6; }
import sys
import numpy as np
import onnxruntime as ort
s = ort.InferenceSession(sys.argv[1], providers=["CUDAExecutionProvider"])
if s.get_providers()[0] != "CUDAExecutionProvider":
    sys.exit("not on CUDA")
s.run(None, {"images": np.zeros((1, 3, 640, 640), np.float32),
             "orig_target_sizes": np.array([[640, 640]], np.int64)})
print("ONNX Runtime is running on the GPU:", s.get_providers())
PY

# The server can close a long transfer part-way (session c6-2, at 96%), which curl does not retry
# by itself; each attempt resumes from what has arrived, and the MD5 below checks the result.
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

RC=0
for KIND in $KINDS; do
    for FOLD in 0 1 2; do
        LEFT=$((BUDGET_S - (SECONDS - T0)))
        if [ "$LEFT" -le 60 ]; then
            echo "time budget used before ${KIND}_fold$FOLD"
            RC=7
            break 2
        fi
        timeout "$LEFT" python "$REPO/scripts/run_baseline.py" --detector dfine \
            --fold "fold$FOLD" --onnx "$MODELS/${KIND}_fold$FOLD/last.onnx" \
            --name "dfine_$KIND" --gpu || { echo "failed: ${KIND}_fold$FOLD"; RC=5; break 2; }
        echo "${KIND}_fold$FOLD done $(date -u +%FT%TZ)"
    done
done
if [ -n "$LOFO_TAG" ] && [ "$RC" -eq 0 ]; then
    for FARM in $FARMS; do
        LEFT=$((BUDGET_S - (SECONDS - T0)))
        if [ "$LEFT" -le 60 ]; then
            echo "time budget used before g6ref_$FARM"
            RC=7
            break
        fi
        timeout "$LEFT" python "$REPO/scripts/run_baseline.py" --detector dfine --fold all \
            --farm "$FARM" --onnx "$LOFO_MODELS/g6ref_$FARM/last.onnx" \
            --name "dfine_g6ref_$FARM" --gpu || { echo "failed: g6ref_$FARM"; RC=5; break; }
        echo "g6ref_$FARM done $(date -u +%FT%TZ)"
    done
fi
for FARM in $FARMS; do
    if [ -d "$HERDSIGHT_STORAGE/runs/baselines/dfine_g6ref_$FARM" ]; then
        cp -r "$HERDSIGHT_STORAGE/runs/baselines/dfine_g6ref_$FARM" "$OUT/predictions/"
        cp "$REPO/results/baselines/dfine_g6ref_${FARM}_all_run.json" "$OUT/predictions/" 2>/dev/null
    fi
done
for KIND in $KINDS; do
    if [ -d "$HERDSIGHT_STORAGE/runs/baselines/dfine_$KIND" ]; then
        cp -r "$HERDSIGHT_STORAGE/runs/baselines/dfine_$KIND" "$OUT/predictions/"
    fi
done
for KIND in $KINDS; do
    cp "$REPO"/results/baselines/dfine_${KIND}_fold*_run.json "$OUT/predictions/" 2>/dev/null
done
exit "$RC"
