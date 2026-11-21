#!/usr/bin/env bash
# One self-contained training session on a rented RunPod GPU (D-35).
#
# The pod's start command clones this repository at a fixed commit and runs this file (see
# gpu/runpod/README.md). It then:
#   1. runs the job named by HS_JOB (gpu/runpod/job_<name>.sh), writing everything to $OUT;
#   2. uploads $OUT to a NEW private Kaggle dataset, herdsight-rp-<HS_TAG>;
#   3. checks the upload, then deletes the pod; if the upload cannot be confirmed it only STOPS
#      the pod (GPU billing ends, the pod's disk is kept so nothing is lost).
# It never exits back to the container: a container that exits is restarted by RunPod, which
# would run the job again.
#
# Environment (set on the pod):
#   HS_JOB            job name, e.g. smoke
#   HS_TAG            short session label, used in the Kaggle dataset name (lowercase, digits, -)
#   HS_COMMIT         the commit this session runs (checked out by the start command)
#   HS_PRICE_PER_HR   the hourly price quoted when the pod was created (for the run ledger)
#   KAGGLE_API_TOKEN  from the RunPod secret kaggle_api_token
#   RUNPOD_POD_ID     set by RunPod

set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export OUT=/workspace/out
mkdir -p "$OUT"
exec > >(tee -a "$OUT/session_log.txt") 2>&1

T0=$(date +%s)
echo "session start $(date -u +%FT%TZ) job=${HS_JOB} tag=${HS_TAG} commit=${HS_COMMIT}"

stop_pod() {
    echo "stopping pod (outputs kept on the pod's disk)"
    runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
}
delete_pod() {
    echo "deleting pod"
    runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" || stop_pod
}
park() {  # never return to the container
    sleep infinity
}

python -m pip install -q -U kaggle >/dev/null 2>&1 || echo "WARNING: kaggle install failed"

JOB="$REPO/gpu/runpod/job_${HS_JOB}.sh"
if [ -f "$JOB" ]; then
    bash "$JOB"
    JOB_RC=$?
else
    echo "no such job: $JOB"
    JOB_RC=2
fi
echo "job exit code: $JOB_RC"

T1=$(date +%s)
python - "$T0" "$T1" "$JOB_RC" <<'PY'
import json, os, subprocess, sys
t0, t1, rc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
price = float(os.environ.get("HS_PRICE_PER_HR", "0") or 0)
hours = (t1 - t0) / 3600
row = {"tag": os.environ["HS_TAG"], "job": os.environ["HS_JOB"], "commit": os.environ["HS_COMMIT"],
       "gpu": gpu, "price_per_hr": price, "job_hours": round(hours, 3),
       "job_cost_usd_estimate": round(hours * price, 3), "job_exit_code": rc,
       "note": "time from session start to job end; pod start-up and upload add a few minutes"}
json.dump(row, open(os.path.join(os.environ["OUT"], "session.json"), "w"), indent=1)
print(json.dumps(row, indent=1))
PY

# --- upload to a new private Kaggle dataset -------------------------------------------------
kaggle datasets list --mine --csv > /tmp/mine_up.csv 2>&1
USER_REF=$(grep -m1 '/' /tmp/mine_up.csv | cut -d, -f1 | cut -d/ -f1)
if [ -z "$USER_REF" ]; then
    echo "could not read the Kaggle username (Kaggle's reply below; no key is printed):"
    head -3 /tmp/mine_up.csv
    stop_pod
    park
fi
DS="$USER_REF/herdsight-rp-${HS_TAG}"
cat > "$OUT/dataset-metadata.json" <<META
{"title": "herdsight-rp-${HS_TAG}", "id": "${DS}", "licenses": [{"name": "other"}]}
META
echo "uploading to private Kaggle dataset $DS"
if kaggle datasets create -p "$OUT" --dir-mode zip -q; then
    for _ in $(seq 1 30); do
        STATUS=$(kaggle datasets status "$DS" 2>/dev/null | tr -d '[:space:]')
        echo "dataset status: $STATUS"
        [ "$STATUS" = "ready" ] && break
        sleep 20
    done
    if [ "${STATUS:-}" = "ready" ] && kaggle datasets files "$DS" | grep -q session.json; then
        echo "upload confirmed"
        delete_pod
        park
    fi
fi
echo "upload NOT confirmed"
stop_pod
park
