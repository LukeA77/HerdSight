#!/usr/bin/env bash
# Test session (about 20 minutes): checks every step a real Phase 2 session needs, and measures
# the GPU's speed on the Phase 1 recipe so the cost estimates can be corrected before real runs.
#   1. machine report (GPU, CUDA, CPU, memory, disk);
#   2. download the training tiles from the owner's private Kaggle datasets;
#   3. time two minutes of the ICAERUS download from Zenodo (full sessions download it whole);
#   4. train D-FINE S on fold0 for 3 epochs with the Phase 1 recipe (a speed benchmark only; the
#      model is not kept) and export it to ONNX;
#   5. write benchmark.json. The Kaggle T4 reference is the Phase 1 run time from its run.json.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK=/workspace/work
PKG=/workspace/packages
mkdir -p "$WORK" "$PKG"

echo "== 1. machine"
nvidia-smi
nproc
free -g
df -h /workspace
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

echo "== 2. training tiles from Kaggle"
T=$(date +%s)
kaggle datasets list --mine --csv | cut -d, -f1 | grep -i herdsight | grep -v herdsight-rp- \
    | while read -r ref; do
        echo "downloading $ref"
        kaggle datasets download "$ref" -p "$PKG" --unzip -q
    done
TILES_S=$(( $(date +%s) - T ))
du -sh "$PKG"

echo "== 3. Zenodo download speed (2 minutes)"
ZEN=$(curl -s -L -o /dev/null --max-time 120 -w '%{size_download} %{time_total}' \
    "https://zenodo.org/records/11048412/files/Cattle_drone_images_042024.zip?download=1")
echo "zenodo bytes and seconds: $ZEN"

echo "== 4. D-FINE S, fold0, 3 epochs"
T=$(date +%s)
HS_WORK="$WORK" HS_PACKAGES="$PKG" HS_RUNS=dfine_s_baseline_fold0 HS_EPOCHS=3 \
    python "$REPO/gpu/train_dfine_baselines_kaggle.py"
TRAIN_RC=$?
TRAIN_S=$(( $(date +%s) - T ))
RUN="$WORK/dfine_s_baseline_fold0"
for f in run.json log.txt train_stdout.txt dfine_s_baseline_fold0.yml; do
    [ -f "$RUN/$f" ] && cp "$RUN/$f" "$OUT/"
done

echo "== 5. benchmark"
python - "$TILES_S" "$ZEN" "$TRAIN_S" "$TRAIN_RC" "$RUN" <<'PY'
import json, os, sys
from datetime import datetime
tiles_s, zen, train_s, rc, run = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
out = {"tiles_download_s": int(tiles_s), "train_exit_code": rc, "train_wall_s": train_s}
if len(zen) == 2 and float(zen[1]) > 0:
    out["zenodo_mb_per_s"] = round(float(zen[0]) / float(zen[1]) / 1e6, 1)
    out["zenodo_full_16gb_minutes_estimate"] = round(16.6e9 / (float(zen[0]) / float(zen[1])) / 60, 1)
# seconds per epoch from the training log's epoch timestamps when present, else wall time / 3
log = os.path.join(run, "log.txt")
epochs = []
if os.path.exists(log):
    epochs = [json.loads(x) for x in open(log) if x.strip()]
out["epochs_logged"] = len(epochs)
import re
stdout = os.path.join(run, "train_stdout.txt")
s_per_it = []
if os.path.exists(stdout):
    for line in open(stdout, errors="ignore"):
        m = re.match(r"Epoch: \[(\d+)/\d+\] Total time: \S+ \(([\d.]+) s / it\)", line)
        if m and int(m.group(1)) > 0:  # epoch 0 includes start-up
            s_per_it.append(float(m.group(2)))
T4_S_PER_IT = 0.8204  # Phase 1 Kaggle T4, dfine_s_baseline_fold0, mean of epochs 1-71
out["s_per_it_epochs_1plus"] = s_per_it
if s_per_it:
    mean = sum(s_per_it) / len(s_per_it)
    out["speed_vs_kaggle_t4"] = round(T4_S_PER_IT / mean, 2)
out["note"] = ("Same recipe, data and batch as the Phase 1 T4 run (0.8204 s per iteration, "
               "epochs 1-71, 3 h 07 min for 72 epochs); speed_vs_kaggle_t4 = T4 time / this time")
json.dump(out, open(os.path.join(os.environ["OUT"], "benchmark.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
PY
exit "$TRAIN_RC"
