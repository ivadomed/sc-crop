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

Add sc_crop to your existing dataset conversion loop. `detect_and_crop` crops the image and returns a context with the bbox coordinates. Pass that context to `crop_nifti` to apply the same bbox to the label.

```python
from sc_crop import detect_and_crop, crop_nifti
import nibabel as nib

# Inside your existing conversion loop (after reorienting image and label to RPI):
crop_img, ctx = detect_and_crop(image_rpi, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))
crop_label    = crop_nifti(nib.load(label_rpi), ctx)

nib.save(crop_img,   out_image)
nib.save(crop_label, out_label)
```

---

## Inference with a model trained with sc_crop

Use `detect_and_crop` + `restore_segmentation` — the segmentation is returned in the exact same space as the original input image.

```python
from sc_crop import detect_and_crop, restore_segmentation
import nibabel as nib

# 1. Detect bbox and crop (use same padding as at training time)
crop_nii, ctx = detect_and_crop(image_path, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))

# 2. Run your segmentation model on the crop → seg_crop (nib.Nifti1Image, same space as crop_nii)
seg_crop = my_model(crop_nii)

# 3. Restore segmentation to the full original image space
seg_full = restore_segmentation(seg_crop, ctx)
nib.save(seg_full, out_path)
```

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
