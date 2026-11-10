#!/bin/bash
#
# Train all the configurations of run_experiments.sh on Compute Canada (SLURM): 4 configurations x 5 model sizes
# = 20 trainings, in one job on one node with 4 GPUs. The model sizes are trained one after the other (yolo26n,
# s, m, l, x); for each size, the 4 configurations are trained at the same time, one per GPU, and the next size
# starts when the 4 are done:
#
#   GPU 0 : grayscale dataset, without extra augmentations
#   GPU 1 : grayscale dataset, with extra augmentations (--extra-augment)
#   GPU 2 : RGB dataset,       without extra augmentations
#   GPU 3 : RGB dataset,       with extra augmentations (--extra-augment)
#
# Each training is saved in OUTPUT_DIR/<YYYYMMDD_HHMMSS>_<configuration>_<model>/ (same layout as
# run_experiments.sh, so evaluate_external_test.py and compare_inference_speed.py work on OUTPUT_DIR), with the
# terminal output of the training in stdout.log. A failed training does not stop the others. The end of each
# training (done or FAILED) is written in OUTPUT_DIR/<YYYYMMDD_HHMMSS>_status.txt.
# The output and errors of the job are in the files of the --output and --error SBATCH options.
# --time must cover the 5 model sizes one after the other.
#
# Before the first job (on a login node, which has internet; compute nodes do not):
#   1. Create the virtual environment, with the modules of MODULES loaded BEFORE activating it (OpenCV comes from
#      the opencv module):
#        module purge
#        module load StdEnv/2023 gcc/12.3 python/3.11 opencv/4.10.0
#        virtualenv --no-download <VENV>
#        source <VENV>/bin/activate
#        pip install --no-index --upgrade pip
#        pip install --no-index torch torchvision
#        pip install nibabel==5.4.2 ultralytics==8.4.192 albumentations==2.0.8 numpy scipy pillow pyyaml matplotlib
#   2. Download the YOLO weights of the 5 model sizes in WEIGHTS_DIR (yolo26n.pt is also used by YOLO to check
#      mixed precision before training):
#        cd <WEIGHTS_DIR> && python -c "from ultralytics import YOLO; [YOLO(f'yolo26{s}.pt') for s in 'nsmlx']"
#   3. Edit the paths below and the SBATCH options (account, time, CPUs, memory).
#
# Usage:
#   sbatch train_compute_canada_all.sh
#   (only some model sizes: edit MODELS below)
#
# Author: Pierre-Louis Benveniste

#SBATCH --account=aip-jcohen
#SBATCH --job-name=sc_crop_all
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/p/plb/links/scratch/sc_crop/trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/sc_crop/trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL

# ── Paths and settings (to edit) ──────────────────────────────────────────────
REPO=/home/p/plb/links/scratch/sc_crop/sc-crop
VENV=/home/p/plb/links/scratch/sc_crop/venv_sc_crop
GRAY_DATASET=/home/p/plb/links/scratch/sc_crop/yolo_dataset/normal_dataset_no_totalsegmentator_mri/dataset.yaml
RGB_DATASET=/home/p/plb/links/scratch/sc_crop/yolo_dataset/rgb_dataset_no_totalsegmentator_mri/dataset.yaml
WEIGHTS_DIR=/home/p/plb/links/scratch/sc_crop/weights
OUTPUT_DIR=/home/p/plb/links/scratch/sc_crop/trainings
EPOCHS=100
MODULES="StdEnv/2023 gcc/12.3 python/3.11 opencv/4.10.0"   # same modules as when the venv was created
# ──────────────────────────────────────────────────────────────────────────────

# The modules must be loaded before activating the virtual environment (OpenCV comes from the opencv module)
module purge
module load $MODULES
source "$VENV/bin/activate"

# Model sizes, trained one after the other (smallest first)
MODELS=(yolo26n yolo26s yolo26m yolo26l yolo26x)
# Configuration i runs on GPU i
NAMES=(gray gray_extra rgb rgb_extra)
DATASETS=("$GRAY_DATASET" "$GRAY_DATASET" "$RGB_DATASET" "$RGB_DATASET")
EXTRA_AUGMENT=(no yes no yes)

DATE=$(date +%Y%m%d_%H%M%S)
STATUS="$OUTPUT_DIR/${DATE}_status.txt"
mkdir -p "$OUTPUT_DIR"
# YOLO finds the weights (and yolo26n.pt for the mixed precision check) in the current folder: no download needed
cd "$WEIGHTS_DIR"

for model in "${MODELS[@]}"; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $model: training the 4 configurations"
    for gpu in 0 1 2 3; do
        run_dir="$OUTPUT_DIR/${DATE}_${NAMES[$gpu]}_${model}"
        mkdir -p "$run_dir"
        # The CPUs of the job are shared between the 4 trainings (data loading workers)
        args=(--dataset "${DATASETS[$gpu]}" --run-dir "$run_dir" --device "$gpu" --model "$model.pt"
              --epochs "$EPOCHS" --workers $((SLURM_CPUS_PER_TASK / 4)))
        if [ "${EXTRA_AUGMENT[$gpu]}" = yes ]; then
            args+=(--extra-augment)
        fi
        echo "  GPU $gpu: ${NAMES[$gpu]}_${model} (${DATASETS[$gpu]}, extra augmentations: ${EXTRA_AUGMENT[$gpu]}) → $run_dir"
        (
            if python "$REPO/training/train_detection.py" "${args[@]}" > "$run_dir/stdout.log" 2>&1; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') GPU $gpu ${NAMES[$gpu]}_${model}: done" >> "$STATUS"
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S') GPU $gpu ${NAMES[$gpu]}_${model}: FAILED (see $run_dir/stdout.log)" >> "$STATUS"
            fi
        ) &
    done
    # The next model size starts when the 4 trainings of this one are done
    wait
done

# How each training ended
cat "$STATUS"
