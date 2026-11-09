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
- `data/build_msd_dataset.py`: gathers every SC segmentation with its image and metadata in an MSD json. Skips empty masks and labels with different affine from image, splits the subjects of each dataset 60/20/20 into training/validation/test and logs statistics. External test datasets (`external_test: true` in `datasets.yaml`) all go to an `external_test` split, keeping only images with both an SC and a gold canal segmentation.
- `data/segment_canal.py`: segments the spinal canal of every image of the MSD json with `sct_deepseg canal` (GPU) and saves the predictions in `derivatives/pred-canal-seg/` of each dataset, a copy of the MSD json with the path of each prediction (`canal` field), and logs of failed and empty segmentations.
- `data/compute_volumes.py`: from the MSD json of `segment_canal.py`, computes for each image the volume of the image, of the SC and of the canal (gold, else silver) and their ratios (csv), with the counts and mean values per dataset (log).
- `data/build_yolo_dataset.py`: converts the MSD json of `segment_canal.py` into 2D axial slices (LAS, 1 x 1 x 10 mm, grayscale or pseudo-RGB with `--rgb`) with one YOLO label file per slice and two classes: SC (0) and canal (gold, else silver) (1), keeping only the largest component of each mask.
- `data/debug_yolo_dataset.py`: runs the processing of `build_yolo_dataset.py` on one image and its SC and canal masks, and plots each slice with its SC (green) and canal (red) boxes, to check the dataset.
- `train_detection.py`: trains the YOLO model detecting the SC and the canal on the slices (`--extra-augment` adds MRI intensity augmentations: downscale, blur, noise, gamma, bias field, inversion), then evaluates the best model on the training and validation sets (with `evaluate_detection.py`).
- `evaluate_detection.py`: evaluates a model on one split (by default the test set): slice, box and volume metrics, plots, and the 100 worst slices with their ground truth and predicted boxes.
- `run_experiments.sh`: launches 4 trainings at the same time, one per GPU (0 to 3): grayscale or RGB dataset, with or without `--extra-augment`.
- `compare_inference_speed.py`: compares the inference time of all the models trained by `run_experiments.sh`, on the same 50 test volumes of the MSD json selected at random: time from the nifti image to the boxes, i.e. preprocessing and prediction, with the functions of `../inference/`.