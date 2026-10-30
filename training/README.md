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
- `data/download_all_datasets.sh`: clones (or updates) every dataset of `datasets.yaml`, downloads their git-annex files and logs the version (branch and commit) of each dataset.
- `data/build_msd_dataset.py`: gathers every SC segmentation with its image and metadata in an MSD json. Skips empty masks and labels with different affine from image, splits the subjects of each dataset 60/20/20 into training/validation/test and logs statistics.
- `data/build_yolo_dataset.py`: converts the MSD json into 2D axial slices (LAS, 1 x 1 x 10 mm, grayscale or pseudo-RGB with `--rgb`) with one YOLO bounding box file per slice, keeping only the largest SC component.
- `train_detection.py`: trains the YOLO model detecting the SC on the slices, with optional MRI-specific augmentations and Weights & Biases logging.
