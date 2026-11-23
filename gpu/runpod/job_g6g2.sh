#!/usr/bin/env bash
# Phase 2, first new-farm experiment (D-44): the G2 appearance and crowding recipe under
# leave-one-farm-out, 72 epochs per held-out farm, then full-image predictions of each held-out
# farm. Hard negatives are left out, because they were mined by models trained on the farm that is
# now held out. Test-farm images are never read. Scoring stays at home base.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO/gpu/runpod/lib.sh"
export PYTHONPATH="$REPO/python"
export HERDSIGHT_STORAGE=/root/hs_storage
FARMS="Derval Jalogny Mauron"
ZIP_MD5=c18911fb11cb58741b9cfa16516ca8cd
RAW="$HERDSIGHT_STORAGE/datasets/raw/icaerus_grazing_cows_v2"
PKGROOT=/root/hs_packages
WORK=/root/hs_work
LOFO=/root/hs_lofo
BUDGET_S=${HS_JOB_SECONDS:-24000}  # leaves time to upload whatever is finished
T0=$SECONDS
mkdir -p "$RAW/unpacked" "$WORK" "$OUT/predictions"

URL="https://zenodo.org/records/11048412/files/Cattle_drone_images_042024.zip?download=1"
RATE=$(curl -sL -r 0-40000000 --max-time 30 -o /dev/null -w '%{speed_download}' "$URL" | cut -d. -f1)
echo "Zenodo speed on this host: ${RATE:-0} bytes/s"
if [ "${RATE:-0}" -lt "${HS_MIN_RATE:-1500000}" ]; then
    echo "host network too slow for the photos; not running the job"
    exit 10
fi
download_tiles "$PKGROOT" || exit 3
PKG=$(package_dir "$PKGROOT" herdsight_tiles_v1) || { echo "tile package not found"; exit 3; }
python "$REPO/gpu/make_lofo_sets.py" --package "$PKG" --out "$LOFO" || exit 3
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
EXCLUDE="$REPO/results/data/training_exclusions.json"
K=$(python - "$REPO/configs/g2_recipe.yaml" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["oversample_touching"]["value"])
PY
)
NCPU=$(usable_cpus)
WORKERS=$(( NCPU > 14 ? 12 : (NCPU > 3 ? NCPU - 2 : 1) ))
echo "usable CPUs: $NCPU, data-loading workers: $WORKERS, touching-tile repeats: $K"
for FARM in $FARMS; do
    RUN="g6g2_${FARM}"
    LEFT=$((BUDGET_S - (SECONDS - T0)))
    [ "$LEFT" -gt 60 ] || { echo "time budget used before $RUN"; exit 7; }
    timeout "$LEFT" python "$REPO/gpu/train_dfine.py" --run "$RUN" --package "$LOFO" \
        --fold "lofo_$FARM" --epochs 72 --recipe g2 --oversample-touching "$K" \
        --exclude "$EXCLUDE" --workers "$WORKERS" --work "$WORK" --dfine /root/D-FINE || exit 4
    mkdir -p "$OUT/$RUN"
    for f in run.json log.txt train_stdout.txt "$RUN.yml" last.onnx; do
        cp "$WORK/$RUN/$f" "$OUT/$RUN/"
    done
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

RC=0
for FARM in $FARMS; do
    LEFT=$((BUDGET_S - (SECONDS - T0)))
    if [ "$LEFT" -le 60 ]; then
        echo "time budget used before predicting $FARM"
        RC=7
        break
    fi
    timeout "$LEFT" python "$REPO/scripts/run_baseline.py" --detector dfine --fold all \
        --farm "$FARM" --onnx "$WORK/g6g2_$FARM/last.onnx" --name "dfine_g6g2_$FARM" --gpu \
        || { echo "failed: predictions for $FARM"; RC=5; break; }
    echo "predictions for $FARM done $(date -u +%FT%TZ)"
done
for FARM in $FARMS; do
    if [ -d "$HERDSIGHT_STORAGE/runs/baselines/dfine_g6g2_$FARM" ]; then
        cp -r "$HERDSIGHT_STORAGE/runs/baselines/dfine_g6g2_$FARM" "$OUT/predictions/"
        cp "$REPO/results/baselines/dfine_g6g2_${FARM}_all_run.json" "$OUT/predictions/" 2>/dev/null
    fi
done
echo "G6 G2-recipe session finished with code $RC"
exit "$RC"
