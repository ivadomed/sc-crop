#!/bin/bash
#
# Same as run_experiments.sh, with the yolo26x model only: 4 configurations x 1 model = 4 trainings.
# Each GPU (0 to 3) has one configuration and trains yolo26x for 100 epochs, while the 4 GPUs run at the same time:
#
#   GPU 0 : grayscale dataset, without extra augmentations
#   GPU 1 : grayscale dataset, with extra augmentations (--extra-augment)
#   GPU 2 : RGB dataset,       without extra augmentations
#   GPU 3 : RGB dataset,       with extra augmentations (--extra-augment)
#
# Each training is saved in <output_folder>/<YYYYMMDD_HHMMSS>_<configuration>_<model>/ (checkpoints, evaluation,
# training.log), with the terminal output of the training in stdout.log. A failed training does not stop the
# next ones. The end of each training (done or FAILED) is written in <output_folder>/<YYYYMMDD_HHMMSS>_status.txt.
# The trainings keep running if the terminal is closed. The script waits until all trainings are done.
# Do not pass --model in the other arguments: it would replace the model size of every training.
#
# Usage:
#   bash run_experiments_x.sh <gray_dataset.yaml> <rgb_dataset.yaml> <output_folder> [other train_detection.py arguments]
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

# Configuration i runs on GPU i
NAMES=(gray gray_extra rgb rgb_extra)
DATASETS=("$GRAY_DATASET" "$GRAY_DATASET" "$RGB_DATASET" "$RGB_DATASET")
EXTRA_AUGMENT=(no yes no yes)
# Model size trained on each GPU
MODELS=(yolo26x)
EPOCHS=100
STATUS="$OUTPUT_DIR/${DATE}_status.txt"

PIDS=()
for gpu in 0 1 2 3; do
    echo "GPU $gpu: ${NAMES[$gpu]} (${DATASETS[$gpu]}, extra augmentations: ${EXTRA_AUGMENT[$gpu]}), models ${MODELS[*]}"
    (
        trap '' HUP   # the trainings keep running if the terminal is closed
        for model in "${MODELS[@]}"; do
            run_dir="$OUTPUT_DIR/${DATE}_${NAMES[$gpu]}_${model}"
            mkdir -p "$run_dir"
            args=(--dataset "${DATASETS[$gpu]}" --run-dir "$run_dir" --device "$gpu" --model "$model.pt"
                  --epochs "$EPOCHS")
            if [ "${EXTRA_AUGMENT[$gpu]}" = yes ]; then
                args+=(--extra-augment)
            fi
            if python "$TRAIN_SCRIPT" "${args[@]}" "${EXTRA_ARGS[@]}" > "$run_dir/stdout.log" 2>&1; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') GPU $gpu ${NAMES[$gpu]}_${model}: done" >> "$STATUS"
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S') GPU $gpu ${NAMES[$gpu]}_${model}: FAILED (see $run_dir/stdout.log)" >> "$STATUS"
            fi
        done
    ) &
    PIDS+=($!)
done

echo "Trainings launched. Follow them with: cat $STATUS, or tail -f $OUTPUT_DIR/${DATE}_<configuration>_<model>/stdout.log"

# Wait for the 4 GPUs, then print how each training ended
wait "${PIDS[@]}"
cat "$STATUS"
