# sc-crop — Spinal cord detection and cropping

Detects the spinal cord on any MRI volume and outputs a tight 3D bounding box. Works across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## Install

**Option A — conda (recommended)**

```bash
conda create -n sc_crop python=3.12
conda activate sc_crop
pip install git+https://github.com/ivadomed/sc-crop.git
```

**Option B — pip**

```bash
pip install git+https://github.com/ivadomed/sc-crop.git
```

**_Optional_ — use `sc_crop` without activating the environment each time:**

```bash
# conda
mkdir -p ~/.local/bin
ln -s $(conda run -n sc_crop which sc_crop) ~/.local/bin/sc_crop

# venv
mkdir -p ~/.local/bin
ln -s $(pwd)/venv/bin/sc_crop ~/.local/bin/sc_crop
```

Make sure `~/.local/bin` is in your `PATH` (add to `~/.bashrc` or `~/.zshrc` if needed):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## Usage

### Download the model (first use only)

```bash
sc_crop download
```

### Crop a volume around the spinal cord

```bash
sc_crop -i t2.nii.gz
```

Outputs `t2_bbox.txt` next to the input with the inclusive voxel bounding box in native image space, compatible with SCT's `sct_crop_image`.

### Optional parameters

| Parameter | Description | Default |
|---|---|---|
| `-o OUTPUT` | Output path (bbox txt, or crop volume if `--crop`) | `<stem>_bbox.txt` |
| `--crop` | Also save the cropped volume | off |
| `--las` | Output cropped volume in LAS orientation (requires `--crop`) | off |
| `--no-translate` | Do not update affine (by default affine is updated for correct FSLeyes overlay) | off |
| `--padding-rl MM` | Right-Left padding in mm | 10 |
| `--padding-ap MM` | Anterior-Posterior padding in mm | 15 |
| `--padding-si MM` | Superior-Inferior padding in mm (`'sup inf'` or single value) | `'30 20'` |
| `--conf FLOAT` | Detection confidence threshold | from config |
| `--regularization` | Regularization method: `cls`, `graphtrim`, `none` | `cls` |
| `--no-onnx` | Use PyTorch `.pt` inference instead of ONNX Runtime | off |
| `--device` | Inference device: `cpu`, `cuda`, `mps` (only with `--no-onnx`) | auto |
| `--debug` | Save `<stem>_debug.png` (per-slice panel with bbox) | off |
| `--time` | Print elapsed time per pipeline step | off |

### GPU inference

By default, `sc_crop` uses ONNX Runtime (CPU), which loads ~15× faster than PyTorch and requires no GPU.
For GPU inference, use PyTorch with `--no-onnx`:

```bash
sc_crop -i t2.nii.gz --no-onnx --device cuda
```

> **Note:** `onnxruntime-gpu` is not supported as it conflicts with `onnxruntime` and requires a specific CUDA version. GPU users should use `--no-onnx --device cuda` instead.

---

## Python API

```python
from sc_crop import run

result = run("t2.nii.gz")                             # bbox txt only
result = run("t2.nii.gz", crop=True)                  # + cropped volume (native)
result = run("t2.nii.gz", crop=True, las=True)        # + cropped volume (LAS)
result = run("t2.nii.gz", crop=True, translate=False) # affine NOT updated

# result keys: bbox_file, xmin, xmax, ymin, ymax, zmin, zmax, original_axcodes
# + output (if crop=True)
```

---

## Requirements

Python ≥ 3.8. Installed automatically by pip:
`nibabel`, `numpy`, `pillow`, `pyyaml`, `onnxruntime`.

`ultralytics` is an optional dependency required only for PyTorch/GPU inference (`--no-onnx`):

```bash
pip install "sc-crop[yolo]"
```

---

## Use in your training pipeline

The key idea is **train/test consistency**: at inference time, sc-crop crops the image before the segmentation model sees it. If the model was trained on full volumes, there is a domain shift. Training on sc-crop-cropped volumes removes this shift and yields better results.

```
Full volume                   Cropped volume
┌─────────────────────────┐   ┌──────────────┐
│  background  background  │   │              │
│     ┌──────────────┐     │   │  spinal cord │
│     │  spinal cord │     │──▶│  + padding   │
│     └──────────────┘     │   │              │
│  background  background  │   └──────────────┘
└─────────────────────────┘
      training (before)            training (after sc-crop)
                                   = same as inference
```

### Why crop at training time?

Without cropping, the model is trained on full volumes where the spinal cord occupies a small fraction of voxels. At inference time, sc-crop crops the volume first, exposing the model to a different input distribution (no large background regions). Cropping at training time eliminates this mismatch. In practice, this gives:

- **Better Dice** on cropped volumes (no domain shift)
- **Faster training** (smaller volumes → more samples per GPU hour)
- **Faster inference** (the segmentation model runs on fewer voxels)

### Install with GPU support

GPU batch inference is strongly recommended for preprocessing large datasets (thousands of volumes). Install `ultralytics`:

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git"
```

### Preprocessing your dataset

`sc_crop` provides a `preprocess-nnunet` command that takes your MSD-format datalists (JSON files with image/label pairs) and outputs a ready-to-use nnUNet raw dataset where every volume has been cropped around the detected spinal cord.

**How it works internally — 3-phase GPU pipeline:**

```
Phase 1 (parallel CPU)  →  load all volumes, resample, extract 2D axial slices
Phase 2 (GPU)           →  pool ALL slices from ALL volumes → single YOLO pass in batches
                            → maximum GPU utilisation, no per-volume overhead
Phase 3 (parallel CPU)  →  aggregate per-volume 3D bbox, apply padding, crop image+label, save
```

This is ~10-30× faster than processing volumes one by one.

**CLI:**

```bash
sc_crop preprocess-nnunet \
    --input     /path/to/msd_datalists/ \
    --output    /path/to/nnUNet_raw/ \
    --taskname  MyDatasetCropped \
    --tasknumber 1000 \
    --device    cuda \
    --pad-rl    10 \
    --pad-ap    15 \
    --pad-superior 30 \
    --pad-inferior 30
```

The `--input` folder must contain JSON files in MSD format (one per dataset), each with `"train"`, `"validation"`, and `"test"` keys listing `{"image": ..., "label": ...}` pairs.

The command outputs:
```
nnUNet_raw/
└── Dataset1000_MyDatasetCropped/
    ├── imagesTr/     # cropped training images
    ├── labelsTr/     # cropped training labels  (same bbox as image)
    ├── imagesTs/     # cropped test images
    ├── labelsTs/     # cropped test labels
    └── dataset.json  # nnUNet metadata + sc_crop parameters used
```

All outputs are reoriented to **RPI**, binarized (labels > 0.5), and have orthonormal direction cosines (SVD-corrected) so that SimpleITK/nnUNet never rejects them.

**Resume after a crash:** if the job is interrupted, re-run the same command. Already-cropped volumes are automatically skipped.

**All CLI options:**

```
--input            Folder of MSD JSON datalists (*_seed50.json)
--nnunet-dir       Alternatively, an existing nnUNet raw dataset directory
--output           Output parent directory (nnUNet_raw)
--taskname         nnUNet dataset name suffix        [default: SCCropped]
--tasknumber       nnUNet dataset number             [default: 999]
--device           cuda or cpu                       [default: cuda]
--batch-size       2D slices per GPU batch           [default: 64]
--workers          CPU threads for parallel I/O      [default: min(8, cpu_count)]
--pad-left         Left padding in mm                [default: 20]
--pad-right        Right padding in mm               [default: 20]
--pad-anterior     Anterior padding in mm            [default: 30]
--pad-posterior    Posterior padding in mm           [default: 30]
--pad-superior     Superior padding in mm            [default: 40]
--pad-inferior     Inferior padding in mm            [default: 40]
--conf             YOLO detection confidence         [default: 0.1]
--cls-conf         Classifier confidence             [default: 0.5]
--skip-failed      Skip volumes where detection fails (default: raise)
```

**Python API:**

```python
from pathlib import Path
from sc_crop.nnunet import preprocess_dataset

preprocess_dataset(
    datalist_dir = Path("/path/to/msd_datalists/"),
    output_dir   = Path("/path/to/nnUNet_raw/"),
    taskname     = "MyDatasetCropped",
    tasknumber   = 1000,
    device       = "cuda",
    batch_size   = 64,
    pad_left     = 10.0,   # mm
    pad_right    = 10.0,
    pad_anterior = 15.0,
    pad_posterior= 15.0,
    pad_superior = 30.0,
    pad_inferior = 30.0,
)
```

### Full training example (nnUNet)

The complete pipeline from raw BIDS datasets to a trained nnUNet model:

```bash
# 1. Create datalists (JSON with image/label pairs, train/val/test split)
python 02_create_msd_data.py \
    --path-data /data/my-dataset \
    --path-out  /data/datalists/ \
    --seed 50

# 2. Crop all volumes with sc-crop and write nnUNet raw dataset
sc_crop preprocess-nnunet \
    --input      /data/datalists/ \
    --output     /data/nnUNet_raw/ \
    --taskname   MyDatasetCropped \
    --tasknumber 1000 \
    --device     cuda \
    --pad-rl 10 --pad-ap 15 --pad-superior 30 --pad-inferior 30

# 3. nnUNet preprocessing (plan + preprocess)
nnUNetv2_plan_and_preprocess -d 1000 --verify_dataset_integrity -c 3d_fullres

# 4. Train
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 1000 3d_fullres 0 -tr nnUNetTrainer -p nnUNetPlans
```

### Padding recommendations

Padding controls how much context around the spinal cord is kept in the cropped volume. Tighter padding = faster training but less context for the model.

| Use case | RL | AP | SI |
|---|---|---|---|
| Tight crop (fast, less context) | 10 mm | 15 mm | 30 mm |
| Standard crop | 20 mm | 30 mm | 40 mm |
| Loose crop (more context) | 30 mm | 40 mm | 60 mm |

The contrast-agnostic SC segmentation model (v3.0) was trained with **RL 10 mm · AP 15 mm · SI 30 mm**.

### sc-crop in your inference script

At inference time, use the same padding as at training time to ensure consistency. The recommended pattern uses `detect_and_crop` + `restore_segmentation` — no intermediate file is written and the segmentation is returned in the exact same space as the input.

```python
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from sc_crop import detect_and_crop, restore_segmentation

# Step 1 — detect SC bbox and get cropped volume (native orientation)
crop_nii, ctx = detect_and_crop("image.nii.gz")

# Step 2 — reorient to RPI (required by nnUNet; skip if your model doesn't need it)
rpi  = axcodes2ornt(('R', 'P', 'I'))
orig = io_orientation(crop_nii.affine)
crop_rpi = crop_nii.as_reoriented(ornt_transform(orig, rpi))

# Step 3 — run your segmentation model → seg_rpi (nib.Nifti1Image, same space as crop_rpi)
seg_rpi = my_model(crop_rpi)

# Step 4 — reorient segmentation back to original orientation
seg_crop = seg_rpi.as_reoriented(ornt_transform(rpi, orig))

# Step 5 — paste segmentation back into the full original image space
seg_full = restore_segmentation(seg_crop, ctx)
nib.save(seg_full, "seg.nii.gz")
```

A complete minimal template is available in [`examples/infer_with_sc_crop.py`](examples/infer_with_sc_crop.py).

Models are downloaded automatically on first call and cached in `~/.cache/sc_crop/`. SHA256 is verified on every load.

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
