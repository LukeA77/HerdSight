# Shared steps for gpu/runpod/job_*.sh (sourced, not run).

# Download every HerdSight tile package from the owner's private Kaggle datasets into $1.
download_tiles() {
    local dest="$1"
    mkdir -p "$dest"
    if ! kaggle datasets list --mine --csv > /tmp/mine.csv 2>&1; then
        echo "Kaggle refused the request (is the kaggle_api_token secret valid?):"
        head -5 /tmp/mine.csv
        return 3
    fi
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
