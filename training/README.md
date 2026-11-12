# sc-crop training

Scripts to download the datasets, build the training data and train the YOLO spinal cord detection model.

## Installation

The training scripts were set up with Python 3.10 on a Linux server with NVIDIA GPUs.

```bash
conda create -n venv_sc_crop python=3.10 -y
conda activate venv_sc_crop
```

Install pytorch

```bash
python3 -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

Then the other librairies

```bash
pip install -r training/requirements.txt
```

## Files

In order of use:

- `data/datasets.yaml`: list of the datasets with, for each one, where to download it and how to find its SC segmentations.
- `data/exclude.txt`: images excluded from the MSD json (one path per line, relative to the data folder).
- `data/download_all_datasets.sh`: clones (or updates) every dataset of `datasets.yaml`, downloads their git-annex files and logs the version (branch and commit) of each dataset.
- `data/segment_sc_totalseg_mri.py`: segments the SC of the fullbody-totalsegmentator-mri images with `sct_deepseg spinalcord` (GPU) into `derivatives/pred-sc-seg/`, and logs failed and empty segmentations and SC outside the TotalSegmentator canal.
- `data/fix_seg_affine.py`: resamples the segmentations whose affine differs from their image onto the image (`sct_register_multimodal -identity 1 -x nn`) and overwrites them (canal masks of dcm-oklahoma and dcm-zurich, SC masks of spider-challenge-2023).
- `data/build_msd_dataset.py`: gathers every SC segmentation with its image and metadata in an MSD json. Skips empty masks and labels with different affine from image, splits the subjects of each dataset 60/20/20 into training/validation/test and logs statistics. External test datasets (`external_test: true` in `datasets.yaml`) all go to an `external_test` split, keeping only images with both an SC and a gold canal segmentation.
- `data/segment_canal.py`: segments the spinal canal of every image of the MSD json with `sct_deepseg canal` (GPU) and saves the predictions in `derivatives/pred-canal-seg/` of each dataset, a copy of the MSD json with the path of each prediction (`canal` field), and logs of failed and empty segmentations.
- `data/compute_volumes.py`: from the MSD json of `segment_canal.py`, computes for each image the volume of the image, of the SC and of the canal (gold, else silver) and their ratios (csv), with the counts and mean values per dataset (log).
- `data/build_yolo_dataset.py`: converts the MSD json of `segment_canal.py` into 2D axial slices (LAS, 1 x 1 x 10 mm, grayscale or pseudo-RGB with `--rgb`) with one YOLO label file per slice and two classes: SC (0) and canal (gold, else silver) (1), keeping only the largest component of each mask.
- `data/debug_yolo_dataset.py`: runs the processing of `build_yolo_dataset.py` on one image and its SC and canal masks, and plots each slice with its SC (green) and canal (red) boxes, to check the dataset.
- `train_detection.py`: trains the YOLO model detecting the SC and the canal on the slices (`--extra-augment` adds MRI intensity augmentations: downscale, blur, noise, gamma, bias field, inversion), then evaluates the best model on the training and validation sets (with `evaluate_detection.py`).
- `evaluate_detection.py`: evaluates a model on one split (by default the test set): slice, box and volume metrics, plots, and the 100 worst slices with their ground truth and predicted boxes.
- `scripts/run_experiments.sh`: launches 4 trainings at the same time, one per GPU (0 to 3): grayscale or RGB dataset, with or without `--extra-augment`. `scripts/run_experiments_x.sh`: same with yolo26x only.
- `scripts/train_compute_canada.sh`: SLURM job running `scripts/run_experiments_x.sh` on one Compute Canada node with 4 GPUs (yolo26x, 4 configurations at the same time; paths, virtual environment and setup steps at the top of the script).
- `scripts/train_compute_canada_all.sh`: SLURM job training all the configurations of `run_experiments.sh` on Compute Canada (20 trainings) on one node with 4 GPUs: the model sizes one after the other, the 4 configurations of each size at the same time.
- `evaluate_image.py`: runs a model on one image and compares its boxes to the GT boxes of the SC and canal segmentations: metrics of `evaluate_detection.py`, and plots of each axial slice and of the sagittal view with the GT and predicted boxes.
- `evaluate_sc_only_model.py`: evaluates a previous model detecting only the SC on the same YOLO dataset and metrics (SC class only), and compares it with an SC and canal model (`--compare-with <run_dir>/evaluation`).
- `evaluate_external_test.py`: evaluates all the models trained by `run_experiments.sh` on the external test set (with their validation thresholds), with a summary csv of the main metrics per model, class and dataset.
- `compare_inference_speed.py`: compares the inference time of all the models trained by `run_experiments.sh`, on the same 50 test volumes of the MSD json selected at random: time from the nifti image to the boxes, i.e. preprocessing and prediction, with the functions of `../inference/`.