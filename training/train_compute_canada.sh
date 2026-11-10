#!/bin/bash
#
# Train yolo26x on Compute Canada (SLURM), on one node with 4 GPUs: runs run_experiments_x.sh, which trains the
# 4 configurations at the same time, one per GPU:
#
#   GPU 0 : grayscale dataset, without extra augmentations
#   GPU 1 : grayscale dataset, with extra augmentations (--extra-augment)
#   GPU 2 : RGB dataset,       without extra augmentations
#   GPU 3 : RGB dataset,       with extra augmentations (--extra-augment)
#
# Each training is saved in OUTPUT_DIR/<YYYYMMDD_HHMMSS>_<configuration>_yolo26x/ (stdout.log, checkpoints,
# evaluation), and the end of each training in OUTPUT_DIR/<YYYYMMDD_HHMMSS>_status.txt (see run_experiments.sh).
# The output and errors of the job are in the files of the --output and --error SBATCH options.
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
#   2. Download the YOLO weights in WEIGHTS_DIR: yolo26x.pt (model trained) and yolo26n.pt (used by YOLO to check
#      mixed precision before training):
#        cd <WEIGHTS_DIR> && python -c "from ultralytics import YOLO; YOLO('yolo26x.pt'); YOLO('yolo26n.pt')"
#   3. Edit the paths below and the SBATCH options (account, time, CPUs, memory).
#
# Usage:
#   sbatch train_compute_canada.sh
#
# Author: Pierre-Louis Benveniste

#SBATCH --account=aip-jcohen
#SBATCH --job-name=sc_crop_yolo26x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/p/plb/links/scratch/sc_crop/trainings/%x_%A_v2.out
#SBATCH --error=/home/p/plb/links/scratch/sc_crop/trainings/%x_%A_v2.err
#SBATCH --mail-user=pierrelouis.benveniste03@gmail.com     # whenever the job starts/fails/completes, an email will be sent 
#SBATCH --mail-type=ALL


# ── Paths (to edit) ───────────────────────────────────────────────────────────
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

# YOLO finds yolo26x.pt and yolo26n.pt (mixed precision check) in the current folder: no download needed.
# The CPUs of the job are shared between the 4 trainings (data loading workers).
cd "$WEIGHTS_DIR"
bash "$REPO/training/run_experiments_x.sh" "$GRAY_DATASET" "$RGB_DATASET" "$OUTPUT_DIR" \
    --epochs "$EPOCHS" --workers $((SLURM_CPUS_PER_TASK / 4))
