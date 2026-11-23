#!/usr/bin/env bash
# Phase 2 step C6: oriented-box labels from SAM ViT-B for every cross-validation animal, on one GPU.
# The photos come from their public record (Zenodo 11048412, CC BY 4.0) and are checked against its
# MD5. Test-farm images are never read. Acceptance, the centre rule and the review of 100 labels
# happen at home base, before any oriented model is trained.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$REPO/python"
export HERDSIGHT_STORAGE=/root/hs_storage
ZIP_MD5=c18911fb11cb58741b9cfa16516ca8cd
RAW="$HERDSIGHT_STORAGE/datasets/raw/icaerus_grazing_cows_v2"
BUDGET_S=${HS_JOB_SECONDS:-21600}  # leaves time to upload whatever is finished
T0=$SECONDS
mkdir -p "$RAW/unpacked" "$OUT/labels"

URL="https://zenodo.org/records/11048412/files/Cattle_drone_images_042024.zip?download=1"
# A slow host is not worth holding (exit code 10: no wait before shutting down).
RATE=$(curl -sL -r 0-40000000 --max-time 30 -o /dev/null -w '%{speed_download}' "$URL" | cut -d. -f1)
echo "Zenodo speed on this host: ${RATE:-0} bytes/s"
if [ "${RATE:-0}" -lt "${HS_MIN_RATE:-600000}" ]; then
    echo "host network too slow for the photos; not running the job"
    exit 10
fi
python -m pip install -q "transformers==5.23.0" pyyaml pillow numpy \
    "opencv-python-headless==5.1.0.95" || { echo "could not install packages"; exit 3; }
python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))" \
    || { echo "CUDA not available"; exit 10; }
# Check the labelling code on a drawn animal first, so a library problem stops the job in minutes
# rather than after the long photo download.
python - <<'PY' || { echo "SAM check failed"; exit 4; }
import math
import numpy as np
from PIL import Image, ImageDraw
from herdsight.labels.sam_obb import SamObbLabeller
from herdsight.utils.config import load_config
cfg = load_config("sam_labels")
labeller = SamObbLabeller(cfg.require("sam_model"), cfg.require("sam_revision"), device="cuda")
im = Image.new("RGB", (400, 400), (70, 120, 50))
a, cx, cy, hw, hh = math.radians(30), 200, 200, 70, 25
c, s = math.cos(a), math.sin(a)
pts = [(cx + u * c - v * s, cy + u * s + v * c) for u, v in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
ImageDraw.Draw(im).polygon(pts, fill=(25, 20, 18))
xy = np.array(pts)
label = labeller.label(im, np.array([*xy.min(0), *xy.max(0)]))
assert label.obb is not None, "no mask"
got = math.degrees(float(label.obb[4]))
assert abs((got - 30 + 90) % 180 - 90) <= 10, f"angle {got:.1f} degrees, expected 30"
print(f"SAM check passed: angle {got:.1f} degrees, expected 30")
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

LEFT=$((BUDGET_S - (SECONDS - T0)))
RC=0
if [ "$LEFT" -le 60 ]; then
    echo "time budget used before labelling"
    RC=7
else
    timeout "$LEFT" python "$REPO/scripts/make_sam_labels.py" --device cuda || RC=5
fi
cp "$HERDSIGHT_STORAGE/runs/labels/sam_obb_v1/"* "$OUT/labels/" 2>/dev/null
echo "C6 labelling finished with code $RC"
exit "$RC"
