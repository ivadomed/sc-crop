#!/bin/bash
#
# Launch 4 trainings (train_detection.py) at the same time, one per GPU (0 to 3):
#
#   GPU 0 : grayscale dataset, without extra augmentations
#   GPU 1 : grayscale dataset, with extra augmentations (--extra-augment)
#   GPU 2 : RGB dataset,       without extra augmentations
#   GPU 3 : RGB dataset,       with extra augmentations (--extra-augment)
#
# Each experiment is saved in <output_folder>/<YYYYMMDD_HHMMSS>_<name>/ (checkpoints, evaluation, training.log),
# with the terminal output of the training in stdout.log. The trainings keep running if the terminal is closed.
# The script waits until the 4 trainings are done and prints their exit status.
#
# Usage:
#   bash run_experiments.sh <gray_dataset.yaml> <rgb_dataset.yaml> <output_folder> [other train_detection.py arguments]
#
#
# Author: Pierre-Louis Benveniste

if [ "$#" -lt 3 ]; then
    echo "Usage: bash $0 <gray_dataset.yaml> <rgb_dataset.yaml> <output_folder> [other train_detection.py arguments]"
    exit 1
fi

GRAY_DATASET="$1"
RGB_DATASET="$2"
OUTPUT_DIR="$3"
shift 3
EXTRA_ARGS=("$@")   # passed to every training (e.g. --epochs 100)

TRAIN_SCRIPT="$(dirname "$0")/train_detection.py"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

# Experiment i runs on GPU i
NAMES=(gray gray_extra rgb rgb_extra)
DATASETS=("$GRAY_DATASET" "$GRAY_DATASET" "$RGB_DATASET" "$RGB_DATASET")
EXTRA_AUGMENT=(no yes no yes)

PIDS=()
for gpu in 0 1 2 3; do
    name="${NAMES[$gpu]}"
    run_dir="$OUTPUT_DIR/${DATE}_${name}"
    mkdir -p "$run_dir"
    args=(--dataset "${DATASETS[$gpu]}" --run-dir "$run_dir" --device "$gpu")
    if [ "${EXTRA_AUGMENT[$gpu]}" = yes ]; then
        args+=(--extra-augment)
    fi
    echo "GPU $gpu: $name (${DATASETS[$gpu]}, extra augmentations: ${EXTRA_AUGMENT[$gpu]}) -> $run_dir"
    # nohup: the training keeps running if the terminal is closed
    nohup python "$TRAIN_SCRIPT" "${args[@]}" "${EXTRA_ARGS[@]}" > "$run_dir/stdout.log" 2>&1 &
    PIDS+=($!)
done

echo "Trainings launched (PIDs: ${PIDS[*]}). Follow one with: tail -f $OUTPUT_DIR/${DATE}_<name>/stdout.log"

# Wait for the 4 trainings and report how each one ended
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "${NAMES[$i]}: done"
    else
        echo "${NAMES[$i]}: FAILED (see $OUTPUT_DIR/${DATE}_${NAMES[$i]}/stdout.log)"
    fi
done
