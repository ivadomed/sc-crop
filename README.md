# sc-crop — Spinal cord detection and cropping

Detects the spinal cord on any MRI volume and returns a tight 3D bounding box. Works across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## Install

```bash
pip install git+https://github.com/ivadomed/sc-crop.git@v0.0.5
```

For GPU inference and batch preprocessing (adds `ultralytics`):

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git@v0.0.5"
```

---

## CLI

```bash
sc_crop t2.nii.gz                          # detect only → t2_bbox.txt
sc_crop t2.nii.gz --crop                   # + cropped volume → t2_crop.nii.gz
sc_crop t2.nii.gz --crop --las             # cropped volume in LAS orientation
sc_crop -i t2.nii.gz -o out.nii.gz --crop  # explicit input/output paths
sc_crop t2.nii.gz --crop --time            # print elapsed time per step
```

**Padding** (all values in mm, padding is clamped to image boundaries):

```bash
# Individual per face — highest priority
sc_crop t2.nii.gz --pad-sup 50 --pad-inf 80
sc_crop t2.nii.gz --pad-left 5 --pad-right 15
sc_crop t2.nii.gz --pad-ant 10 --pad-post 20

# Symmetric shorthand
sc_crop t2.nii.gz --pad-si 30              # superior=30 inferior=30
sc_crop t2.nii.gz --pad-si 30 --pad-inf 60 # shorthand + override one face
```

Defaults: `superior=40mm, inferior=60mm, left=right=10mm, anterior=posterior=15mm`.
Priority per face: individual > shorthand > default.

`t2_bbox.txt` contains inclusive voxel indices `xmin xmax ymin ymax zmin zmax` in native image space.

**GPU inference** (requires `[yolo]` extra):

```bash
sc_crop t2.nii.gz --crop --no-onnx --device cuda
```

Run `sc_crop --help` for all options.

---

## Python API

`detect()` is pure — it writes no files. All file I/O is explicit in your code.

### Detect + crop

```python
from sc_crop import detect, crop
import nibabel as nib

ctx = detect("t2.nii.gz")

# crop() works on any volume in the same space (image, label, …)
crop_img   = crop(nib.load("t2.nii.gz"),       ctx)
crop_label = crop(nib.load("t2_label.nii.gz"), ctx)

nib.save(crop_img,   "t2_crop.nii.gz")
nib.save(crop_label, "t2_label_crop.nii.gz")
```

`ctx` contains: `xmin, xmax, ymin, ymax, zmin, zmax` (inclusive, native space), `original_axcodes`.

### Padding

Priority per face: **individual > shorthand > default**.

```python
# Defaults: superior=40mm, inferior=60mm, left=right=10mm, anterior=posterior=15mm
ctx = detect("t2.nii.gz")

# Adjust SI only (most common)
ctx = detect("t2.nii.gz", pad_superior=50, pad_inferior=80)

# Symmetric SI shorthand
ctx = detect("t2.nii.gz", pad_si=30)

# Shorthand + override one face
ctx = detect("t2.nii.gz", pad_si=30, pad_inferior=60)

# Full per-face control
ctx = detect("t2.nii.gz",
             pad_superior=40, pad_inferior=60,
             pad_left=10,     pad_right=10,
             pad_anterior=15, pad_posterior=15)
```

Padding is always clamped to the image boundaries — no out-of-bounds indices are ever produced.

### Restore a segmentation to the original space

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

ctx      = detect("t2.nii.gz")
crop_img = crop(nib.load("t2.nii.gz"), ctx)

seg_crop = my_model(crop_img)              # your model — returns NIfTI in cropped space

seg_full = restore_segmentation(seg_crop, ctx)
nib.save(seg_full, "t2_seg.nii.gz")
```

### Convenience wrapper

```python
from sc_crop import detect_and_crop

crop_nii, ctx = detect_and_crop("t2.nii.gz")  # detect + crop in one call
```

---

## Examples

The `examples/` directory contains two runnable scripts:

**`examples/api_examples.py`** — Python API cookbook. Covers all usage patterns:

```bash
python examples/api_examples.py t2.nii.gz        # run all examples
python examples/api_examples.py t2.nii.gz --ex 3 # run example 3 only (padding)
```

| # | Pattern |
|---|---------|
| 1 | Basic `detect()` + `crop()` |
| 2 | Multi-volume: detect once, crop image + label with the same bbox |
| 3 | Padding variants: default / symmetric / mixed / full individual |
| 4 | `detect_and_crop()` one-liner |
| 5 | `restore_segmentation()` round-trip |
| 6 | GPU inference (`use_onnx=False`) |

**`examples/infer_with_sc_crop.py`** — Inference pipeline template (detect → crop → your model → restore). Replace `run_my_model()` with your own segmentation model.

---

## Requirements

Python ≥ 3.8. Core dependencies installed automatically: `nibabel`, `numpy`, `pillow`, `pyyaml`, `onnxruntime`.

`ultralytics` is optional — required only for GPU/PyTorch inference (`use_onnx=False`), the debug panel (`debug=True`), and batch preprocessing (`preprocess_dataset`). Install with the `[yolo]` extra above.

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
