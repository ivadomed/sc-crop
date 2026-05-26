# sc-crop — Spinal cord detection and cropping

Detects the spinal cord on any MRI volume and outputs a tight 3D bounding box. Works across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## Install

Install into any Python environment (≥ 3.8):

```bash
pip install git+https://github.com/ivadomed/sc-crop.git@v0.0.5
```

Or into a new dedicated environment:

```bash
conda create -n sc_crop python=3.12 && conda activate sc_crop
pip install git+https://github.com/ivadomed/sc-crop.git@v0.0.5
```

To always track the latest version (development):

```bash
pip install git+https://github.com/ivadomed/sc-crop.git
```

GPU inference and batch dataset preprocessing — adds `ultralytics`:

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git@v0.0.5"
```

---

## CLI

```bash
sc_crop t2.nii.gz                                    # detect only → t2_bbox.txt
sc_crop t2.nii.gz --crop                             # + cropped volume → t2_crop.nii.gz
sc_crop t2.nii.gz --crop --las                       # cropped volume in LAS orientation
sc_crop -i t2.nii.gz -o out.nii.gz --crop            # explicit input/output
sc_crop t2.nii.gz --crop --time                      # print elapsed time per step

# Padding — individual (highest priority):
sc_crop t2.nii.gz --pad-sup 50 --pad-inf 80          # SI per face
sc_crop t2.nii.gz --pad-left 5 --pad-right 15        # RL per face
sc_crop t2.nii.gz --pad-ant 10 --pad-post 20         # AP per face

# Padding — symmetric shorthands:
sc_crop t2.nii.gz --pad-si 30                        # SI symmetric 30mm
sc_crop t2.nii.gz --pad-si 30 --pad-inf 60           # shorthand + override one face

# GPU inference (requires pip install "sc-crop[yolo]" and sc_crop download)
sc_crop t2.nii.gz --crop --no-onnx --device cuda
```

Padding defaults: superior=40mm, inferior=60mm, left=right=10mm, anterior=posterior=15mm.
Priority per face: individual (`--pad-sup/inf/left/right/ant/post`) > shorthand (`--pad-si/rl/ap`) > default.
`t2_bbox.txt` contains inclusive voxel indices `xmin xmax ymin ymax zmin zmax` in native image space. Run `sc_crop --help` for all options.

---

## Python interface

### Basic usage

```python
from sc_crop import detect, crop
import nibabel as nib

# 1. Detect the spinal cord — returns a bbox context dict
ctx = detect("t2.nii.gz")

# 2. Crop any volume in the same space (image, label, …) with the same bbox
crop_img   = crop(nib.load("t2.nii.gz"),       ctx)
crop_label = crop(nib.load("t2_label.nii.gz"), ctx)

nib.save(crop_img,   "t2_crop.nii.gz")
nib.save(crop_label, "t2_label_crop.nii.gz")
```

`ctx` always contains: `xmin, xmax, ymin, ymax, zmin, zmax` (inclusive, native space), `bbox_file`, `original_axcodes`.

### Padding

Padding priority per face: **individual > shorthand > default**.

```python
# Defaults: superior=40mm, inferior=60mm, left=right=10mm, anterior=posterior=15mm
ctx = detect("t2.nii.gz")

# Adjust SI only (most common)
ctx = detect("t2.nii.gz", pad_superior=50, pad_inferior=80)

# Symmetric SI shorthand (e.g. focused cervical dataset)
ctx = detect("t2.nii.gz", pad_si=30)

# Shorthand + override one face
ctx = detect("t2.nii.gz", pad_si=30, pad_inferior=60)

# Full individual control
ctx = detect("t2.nii.gz",
             pad_superior=40, pad_inferior=60,
             pad_left=10, pad_right=10,
             pad_anterior=15, pad_posterior=15)
```

### Restore a segmentation to the original space

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

ctx      = detect("t2.nii.gz")
crop_img = crop(nib.load("t2.nii.gz"), ctx)

seg_crop = my_model(crop_img)  # your segmentation model — NIfTI in cropped space

seg_full = restore_segmentation(seg_crop, ctx)  # back to original shape + affine
nib.save(seg_full, "t2_seg.nii.gz")
```

See `examples/infer_with_sc_crop.py` for a full inference template.

### run() — lower-level, writes files to disk

```python
from sc_crop import run

result = run("t2.nii.gz")                              # writes t2_bbox.txt
result = run("t2.nii.gz", crop=True)                   # + t2_crop.nii.gz (native)
result = run("t2.nii.gz", crop=True, las=True)         # + t2_crop_las.nii.gz
result = run("t2.nii.gz", crop=True, translate=False)  # affine NOT updated

# result keys: bbox_file, xmin, xmax, ymin, ymax, zmin, zmax, original_axcodes
# + "output" (crop path) only when crop=True
```

All padding parameters (`pad_superior`, `pad_inferior`, `pad_si`, `pad_left`, `pad_right`,
`pad_rl`, `pad_anterior`, `pad_posterior`, `pad_ap`) and other options (`conf`, `cls_conf`,
`regularization`, `use_onnx`, `device`, `norm_scope`, `debug`, `time_steps`) are accepted
by both `run()` and `detect()`.

### Shorthand functions

```python
from sc_crop import detect_and_crop

# Detect + crop in one call — returns (crop_nii, ctx)
crop_nii, ctx = detect_and_crop("t2.nii.gz")
```

---

## Use in a training pipeline

Detect once, crop image and label with the same bbox. Use the **same padding at inference time**.

```python
from sc_crop import detect, crop
import nibabel as nib

image_path = "sub-001_T2w.nii.gz"
label_path = "sub-001_T2w_label-SC.nii.gz"

ctx        = detect(image_path, pad_rl=10, pad_ap=15, pad_superior=40, pad_inferior=60)
crop_img   = crop(nib.load(image_path), ctx)
crop_label = crop(nib.load(label_path), ctx)

nib.save(crop_img,   "sub-001_T2w_crop.nii.gz")
nib.save(crop_label, "sub-001_T2w_label-SC_crop.nii.gz")
```

See `examples/infer_with_sc_crop.py` for the matching inference pipeline.

---

## Requirements

Python ≥ 3.8. Dependencies installed automatically: `nibabel`, `numpy`, `pillow`, `pyyaml`, `onnxruntime`.

`ultralytics` is optional — required only for:
- GPU/PyTorch inference (`--no-onnx` flag or `use_onnx=False`)
- Debug panel (`--debug` flag or `debug=True`)
- Batch dataset preprocessing (`preprocess_dataset`)

Install with: `pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git@v0.0.5"`

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
