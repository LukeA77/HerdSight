#!/usr/bin/env bash
# Phase 2 step G11: predictions on degraded cross-validation images, for the resolution study.
# Inference only, with the G2 models. The photos come from their public record (Zenodo 11048412,
# CC BY 4.0) and are checked against its MD5. Test-farm images are never read. Scoring stays at
# home base (scripts/resolution_score.py).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO/python"
export HERDSIGHT_STORAGE=/root/hs_storage
TAG="${HS_MODELS_TAG:-g2-3}"
KIND=g2
ZIP_MD5=c18911fb11cb58741b9cfa16516ca8cd
RAW="$HERDSIGHT_STORAGE/datasets/raw/icaerus_grazing_cows_v2"
MODELS=/root/hs_models
BUDGET_S=${HS_JOB_SECONDS:-18000}  # covers the slowest accepted download; leaves time to upload
T0=$SECONDS
mkdir -p "$RAW/unpacked" "$MODELS" "$OUT/g11"

URL="https://zenodo.org/records/11048412/files/Cattle_drone_images_042024.zip?download=1"
# A slow host is not worth holding (exit code 10: no wait before shutting down).
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
python - "$MODELS" "$KIND" <<'PY' || exit 4
import hashlib, json, sys
root, kind = sys.argv[1:3]
for f in range(3):
    d = f"{root}/{kind}_fold{f}"
    want = json.load(open(f"{d}/run.json"))["onnx_sha256"]
    if hashlib.sha256(open(f"{d}/last.onnx", "rb").read()).hexdigest() != want:
        sys.exit(f"model hash differs: {d}")
print("model hashes match")
PY
python - "$MODELS/${KIND}_fold0/last.onnx" <<'PY' || { echo "ONNX Runtime cannot use the GPU"; exit 6; }
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

# The server can close a long transfer part-way (session c6-2), which curl does not retry by
# itself; each attempt resumes from what has arrived, and the MD5 below checks the result.
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
    timeout "$LEFT" python "$REPO/scripts/resolution_predict.py" --models "$MODELS" --kind "$KIND" \
        --gpu || RC=5
fi
cp -r "$HERDSIGHT_STORAGE/runs/phase2/g11/"* "$OUT/g11/" 2>/dev/null
echo "G11 finished with code $RC"
exit "$RC"
