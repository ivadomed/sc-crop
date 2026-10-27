#!/bin/bash
# Download all datasets defined in configs/datasets.yaml.
# Usage: bash scripts/download_all_datasets.sh
#
# Prerequisites:
#   - git-annex installed
#   - conda environment activated
#   - public SSH key added to data.neuro.polymtl.ca and spineimage.ca (falls back to
#     HTTPS per-dataset if SSH fails)
#
# A dataset that fails to clone or to fully download does NOT stop the script — it's
# reported in the final summary instead. preprocess.py skips subjects/datasets whose
# NIfTI files are missing (logged to processed/<variant>/skipped.log).
#
# Inspired from: https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo/blob/main/scripts/download_all_datasets.sh


set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # run from project root

DATA_DIR="/home/plbenveniste/net/sc_crop/data"
LOG="$DATA_DIR/git_branch_commit.log"

mkdir -p "$DATA_DIR"

failed_clone=()
failed_annex=()

# Parse datasets.yaml once → lines of "name|url_ssh|url_https|commit|host"
DATASETS=$(python - <<'EOF'
import yaml
with open("configs/datasets.yaml") as f:
    for d in yaml.safe_load(f)["datasets"]:
        print(f"{d['name']}|{d.get('url_ssh') or ''}|{d.get('url_https') or ''}|{d.get('commit') or ''}|{d.get('host') or ''}")
EOF
)

# ---- Clone each dataset ----
while IFS='|' read -r name url_ssh url_https commit host; do
    # Datasets Zenodo : délégué à scripts/download_<name>.sh
    if [ "$host" = "zenodo" ]; then
        script="scripts/download_${name}.sh"
        if [ -f "$script" ]; then
            echo "=========================================="
            echo "Zenodo dataset: $name → $script"
            echo "=========================================="
            bash "$script"
        else
            echo "  WARNING: $name (zenodo) — script $script introuvable, skipped."
        fi
        continue
    fi
    echo "=========================================="
    echo "Cloning: $name"
    echo "=========================================="

    if [ -d "$DATA_DIR/$name" ]; then
        echo "  -> Already exists, skipping clone."
        continue
    fi

    cloned=0
    if [ -n "$url_ssh" ] && git clone "$url_ssh" "$DATA_DIR/$name"; then
        echo "  -> Cloned via SSH."
        cloned=1
    elif [ -n "$url_https" ] && git clone "$url_https" "$DATA_DIR/$name"; then
        echo "  -> SSH unavailable, cloned via HTTPS."
        cloned=1
    fi

    if [ "$cloned" -eq 0 ]; then
        echo "  ERROR: failed to clone $name (SSH and HTTPS both failed, see git errors above) — skipping."
        failed_clone+=("$name")
        continue
    fi

    git -C "$DATA_DIR/$name" annex dead here
    [ -n "$commit" ] && git -C "$DATA_DIR/$name" checkout -q "$commit"

    branch=$(git -C "$DATA_DIR/$name" rev-parse --abbrev-ref HEAD)
    actual=$(git -C "$DATA_DIR/$name" rev-parse HEAD)
    echo "$name: git-$branch-$actual" >> "$LOG"
done <<< "$DATASETS"

# ---- Download NIfTI files via git-annex (parallel) ----
echo ""
echo "=========================================="
echo "Fetching NIfTI files with git annex get (parallel)..."
echo "=========================================="

pids=()
dataset_names=()
while IFS='|' read -r name url_ssh url_https commit host; do
    [ "$host" = "zenodo" ] && continue   # pas de git annex pour les datasets zenodo
    if [ -d "$DATA_DIR/$name" ]; then
        echo "  -> git annex get: $name"
        (cd "$DATA_DIR/$name" && git annex get .) &
        pids+=($!)
        dataset_names+=("$name")
    fi
done <<< "$DATASETS"

for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || failed_annex+=("${dataset_names[$i]}")
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
if (( ${#failed_clone[@]} > 0 )); then
    echo "Failed to clone (dataset entirely missing):"
    printf "  - %s\n" "${failed_clone[@]}"
fi
if (( ${#failed_annex[@]} > 0 )); then
    echo "Incomplete git-annex get (some NIfTI files missing):"
    printf "  - %s\n" "${failed_annex[@]}"
fi
if (( ${#failed_clone[@]} > 0 || ${#failed_annex[@]} > 0 )); then
    echo ""
    echo "These datasets/subjects will be skipped by preprocess.py, not fail the pipeline."
    echo "Common cause: public SSH key not registered on the data host — add it at"
    echo "https://data.neuro.polymtl.ca/user/settings/keys (or spineimage.ca) and rerun."
else
    echo "Done — all datasets fully downloaded."
fi
echo "=========================================="
