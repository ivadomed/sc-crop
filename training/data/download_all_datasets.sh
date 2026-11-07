#!/bin/bash
# Download every git dataset listed in datasets.yaml (next to this script) into OUTPUT_FOLDER,
# fetch all their git-annex files and record their versions in OUTPUT_FOLDER/datasets_versions.log.
#
# Usage:
#   bash download_all_datasets.sh <output_folder>
#
# Zenodo datasets are not git repos and must be downloaded manually.
#
# Inspired from: https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo/blob/main/scripts/download_all_datasets.sh
#
# Author: Pierre-Louis Benveniste

set -uo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: bash $0 <output_folder>"
    exit 1
fi
mkdir -p "$1"
OUT_DIR="$(cd "$1" && pwd)"
LOG="$OUT_DIR/datasets_versions.log"
# One line per git dataset: "name url_ssh url_https"
DATASETS=$(python - "$(dirname "$0")/datasets.yaml" <<'EOF'
import sys, yaml
for d in yaml.safe_load(open(sys.argv[1]))["datasets"]:
    if d["host"] != "zenodo":
        print(d["name"], d.get("url_ssh") or "-", d.get("url_https") or "-")
EOF
) || exit 1

# Clone each dataset (SSH, then HTTPS) and fetch its annex files, one dataset at a time.
# The list is read from fd 3 so that git/ssh can't swallow it by reading stdin.
while read -r name url_ssh url_https <&3; do
    dir="$OUT_DIR/$name"
    echo "=== $name ==="
    if [ -d "$dir" ]; then
        echo "  Already cloned, pulling latest version..."
        git -C "$dir" pull || echo "  WARNING: could not pull $name, keeping current version."
    else
        echo "  Cloning..."
        if ! { git clone "$url_ssh" "$dir" || git clone "$url_https" "$dir"; }; then
            echo "  ERROR: could not clone $name (SSH and HTTPS failed), skipping."
            continue
        fi
        # Don't advertise this clone as a copy of the data on the remote
        git -C "$dir" annex dead here
    fi
    echo "  Fetching git-annex files..."
    # Output goes straight to the terminal
    if git -C "$dir" annex get .; then
        echo "  Done."
    else
        echo "  FAILED: some files of $name could not be downloaded (see output above)."
    fi
done 3<<< "$DATASETS"

# Record the version of every dataset
echo "=== Recording dataset versions ==="
echo "# Dataset versions ($(date '+%Y-%m-%d %H:%M:%S')): name branch commit" > "$LOG"
while read -r name _ _; do
    dir="$OUT_DIR/$name"
    if [ -d "$dir" ]; then
        echo "$name $(git -C "$dir" rev-parse --abbrev-ref HEAD) $(git -C "$dir" rev-parse HEAD)" >> "$LOG"
    else
        echo "$name MISSING (clone failed)" >> "$LOG"
    fi
done <<< "$DATASETS"

echo "Done. Versions in $LOG"
