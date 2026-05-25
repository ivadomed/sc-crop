# sc-crop — Spinal cord detection and cropping

Detects the spinal cord on any MRI volume and outputs a tight 3D bounding box. Works across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## Install

Install into any Python environment (≥ 3.8):

```bash
pip install git+https://github.com/ivadomed/sc-crop.git
```

Or into a new dedicated environment:

```bash
conda create -n sc_crop python=3.12 && conda activate sc_crop
pip install git+https://github.com/ivadomed/sc-crop.git
```

GPU/batch preprocessing only — adds `ultralytics`:

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git"
```

---

## CLI

```bash
sc_crop t2.nii.gz                        # detect only → t2_bbox.txt
sc_crop t2.nii.gz --crop                 # + cropped volume → t2_crop.nii.gz
sc_crop t2.nii.gz --crop --las           # cropped volume in LAS orientation
sc_crop -i t2.nii.gz -o out.nii.gz --crop
```

`t2_bbox.txt` contains inclusive voxel indices `xmin xmax ymin ymax zmin zmax` in the native image space.

| Parameter | Description | Default |
|---|---|---|
| `-i` / positional | Input NIfTI volume | — |
| `-o OUTPUT` | Output path | `<stem>_bbox.txt` or `<stem>_crop.nii.gz` |
| `--crop` | Save cropped volume | off |
| `--las` | Output crop in LAS orientation (requires `--crop`) | off |
| `--no-translate` | Do not update affine origin in the crop | off |
| `--padding-rl MM` | Right-Left padding in mm | 10 |
| `--padding-ap MM` | Anterior-Posterior padding in mm | 15 |
| `--padding-si MM` | Superior-Inferior padding in mm (single value or `'sup inf'`) | 20 |
| `--conf FLOAT` | Detection confidence threshold | from config |
| `--cls-conf FLOAT` | Classifier confidence threshold | 0.5 |
| `--regularization` | `cls`, `graphtrim`, or `none` | `cls` |
| `--no-onnx` | Use PyTorch `.pt` instead of ONNX Runtime | off |
| `--device` | `cpu`, `cuda`, `mps` — only with `--no-onnx` | auto |
| `--norm-scope` | `volume` or `slice` normalisation | `volume` |
| `--debug` | Save `<stem>_debug.png` (per-slice bbox panel) | off |
| `--time` | Print elapsed time per step | off |

> By default sc_crop uses ONNX Runtime (CPU). For GPU inference use `--no-onnx --device cuda`.

---

## Python API

### Detection and cropping

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

# Detect the spinal cord bbox
ctx = detect("t2.nii.gz", padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=30)

# Apply the bbox to any volume in the same space (image, label, …)
crop_img   = crop(nib.load("t2.nii.gz"),       ctx)  # nib.Nifti1Image
crop_label = crop(nib.load("t2_label.nii.gz"), ctx)  # same bbox

# ctx keys: xmin, xmax, ymin, ymax, zmin, zmax, bbox_file, original_axcodes
```

### Restore a segmentation to the original space

```python
seg_full = restore_segmentation(seg_crop, ctx)  # same shape + affine as the original input
```

### run() — lower-level, writes files to disk

```python
from sc_crop import run

result = run("t2.nii.gz")                              # bbox txt only
result = run("t2.nii.gz", crop=True)                   # + cropped volume (native)
result = run("t2.nii.gz", crop=True, las=True)         # + cropped volume (LAS)
result = run("t2.nii.gz", crop=True, translate=False)  # affine origin NOT updated

# result keys always: bbox_file, xmin, xmax, ymin, ymax, zmin, zmax, original_axcodes
# + "output" (path to crop file) only when crop=True
```

Parameters mirror the CLI: `padding_rl_mm`, `padding_ap_mm`, `padding_si_mm`, `conf`, `cls_conf`,
`regularization`, `use_onnx`, `device`, `norm_scope`, `debug`, `time_steps`, `output_path`.

### Convenience aliases

```python
from sc_crop import detect_and_crop  # equivalent to: crop(nib.load(img), detect(img))
crop_nii, ctx = detect_and_crop("t2.nii.gz")

from sc_crop import crop_nifti  # alias for crop()
```

---

## Use in your training pipeline

Add sc_crop to your dataset conversion loop. Detect once, crop image and label with the same bbox.

```python
from sc_crop import detect, crop
import nibabel as nib

# After reorienting image and label to RPI:
ctx        = detect(image_rpi, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=30)
crop_img   = crop(nib.load(image_rpi), ctx)
crop_label = crop(nib.load(label_rpi), ctx)

nib.save(crop_img,   out_image)
nib.save(crop_label, out_label)
```

---

## Inference with a model trained with sc_crop

Use the **same padding as at training time**. The segmentation is returned in the original image space.

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

ctx      = detect(image_path, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=30)
crop_img = crop(nib.load(image_path), ctx)

seg_crop = my_model(crop_img)           # nib.Nifti1Image in cropped space

seg_full = restore_segmentation(seg_crop, ctx)
nib.save(seg_full, out_path)
```

---

## Requirements

Python ≥ 3.8. Dependencies installed automatically: `nibabel`, `numpy`, `pillow`, `pyyaml`, `onnxruntime`.

`ultralytics` is optional — only needed for GPU/batch inference (`--no-onnx` or `preprocess_dataset`).

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
