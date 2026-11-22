# Shared steps for gpu/runpod/job_*.sh (sourced, not run).

# Download every HerdSight tile package from the owner's private Kaggle datasets into $1.
download_tiles() {
    local dest="$1"
    mkdir -p "$dest"
    # Kaggle lists 20 datasets per page; read every page (the session result datasets,
    # herdsight-rp-*, soon fill the first one)
    : > /tmp/mine.csv
    local page=1 rows
    while [ "$page" -le 50 ]; do
        if ! kaggle datasets list --mine --csv --page "$page" > /tmp/page.csv 2>&1; then
            echo "Kaggle refused the request (is the kaggle_api_token secret valid?):"
            head -5 /tmp/page.csv
            return 3
        fi
        rows=$(tail -n +2 /tmp/page.csv | grep -c . || true)
        [ "$rows" -eq 0 ] && break
        tail -n +2 /tmp/page.csv >> /tmp/mine.csv
        page=$((page + 1))
    done
    local refs
    refs=$(cut -d, -f1 /tmp/mine.csv | grep -i herdsight | grep -v herdsight-rp- || true)
    if [ -z "$refs" ]; then
        echo "No HerdSight datasets found on the Kaggle account."
        head -5 /tmp/mine.csv
        return 3
    fi
    for ref in $refs; do
        echo "downloading $ref"
        kaggle datasets download "$ref" -p "$dest" --unzip -q || { echo "failed: $ref"; return 3; }
    done
}

# The unpacked package folder (holding manifest.json and images/) whose manifest names $2, under $1.
package_dir() {
    python - "$1" "$2" <<'PY'
import json, os, sys, zipfile
root, name = sys.argv[1], sys.argv[2]
for d, dirs, files in os.walk(root):
    if "manifest.json" in files and "images" in dirs:
        if json.load(open(os.path.join(d, "manifest.json"))).get("name") == name:
            print(d); sys.exit(0)
for d, _, files in os.walk(root):
    if name + ".zip" in files:
        zipfile.ZipFile(os.path.join(d, name + ".zip")).extractall(root)
        print(os.path.join(root, name)); sys.exit(0)
sys.exit(1)
PY
}

# CPUs this container may use (cgroup quota), not the host's count that nproc reports.
usable_cpus() {
    python - <<'PY'
import os
n = len(os.sched_getaffinity(0))
def limit():  # CPU quota from cgroup v2, else cgroup v1 (some hosts still use v1)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None
q = limit()
if q is not None:
    n = min(n, max(1, int(q)))
print(n)
PY
}

