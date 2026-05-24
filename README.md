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

sc-crop is designed to be dropped into any nnUNet training script as a preprocessing step.
It crops volumes to a tight bbox around the spinal cord before training → fewer voxels,
faster training, better generalisation.

### Install with GPU support (batch preprocessing)

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git"
```

`[yolo]` adds `ultralytics` for GPU batch inference. The base install (ONNX, CPU) is
sufficient for single-image inference at test time.

### Step 3 of your training script

```bash
# Set to true to use sc-crop detection (realistic pipeline, matches inference)
# Set to false to use GT-bbox crop (oracle upper bound)
USE_SC_CROP=true

if [ "${USE_SC_CROP}" = "true" ]; then
    sc_crop preprocess-nnunet \
        --input    /path/to/msd_datalists/ \
        --output   /path/to/nnUNet_raw/ \
        --taskname MyDatasetCropped \
        --device   cuda
else
    python convert_msd_to_nnunet.py \
        --input    /path/to/msd_datalists/ \
        --output   /path/to/nnUNet_raw/ \
        --taskname MyDatasetCropped
fi
```

`preprocess-nnunet` handles GPU batch inference, parallel I/O, padding, reorientation
to RPI, and writes a valid `dataset.json`. See `sc_crop preprocess-nnunet --help` for
all options.

### sc-crop in your inference script

```python
from sc_crop import run as sc_crop_detect

# Detect the SC and get a cropped volume
result = sc_crop_detect("image.nii.gz", crop=True)
cropped_path = result["output"]  # use this as input to your segmentation model

# Or just get the bounding box indices
result = sc_crop_detect("image.nii.gz")
xmin, xmax = result["xmin"], result["xmax"]
ymin, ymax = result["ymin"], result["ymax"]
zmin, zmax = result["zmin"], result["zmax"]
```

No configuration needed. Models are downloaded automatically on first call and cached in
`~/.cache/sc_crop/`. SHA256 is verified on every load.

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
