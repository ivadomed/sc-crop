# sc-crop — Spinal cord detection and cropping

Segmentation models for spinal cord pathologies (tumors, lesions, SC itself) are typically trained and run on full volumes where the spinal cord occupies only a small fraction of the voxels. **sc-crop** solves this by automatically detecting the spinal cord and cropping the volume tightly around it, so your segmentation model only ever sees the relevant region.

This reduces memory usage, speeds up inference, and often improves model accuracy by removing irrelevant background. The recommended workflow is:

1. **Preprocessing** — crop all training images and labels around the detected spinal cord, adding a fixed security margin
2. **Training** — train your segmentation model on the cropped volumes
3. **Inference** — apply the same sc-crop preprocessing to new images, run your model, then optionally restore the segmentation to the original space

Works on both **MRI and CT**, across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine. Available as a **CLI tool** and as an **importable Python package**.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## Install

Into an existing environment:

```bash
pip install git+https://github.com/ivadomed/sc-crop.git@v0.0.5
```

Or into a fresh dedicated environment (recommended for testing):

```bash
conda create -n sc_crop python=3.12 && conda activate sc_crop
pip install git+https://github.com/ivadomed/sc-crop.git@v0.0.5
```

For GPU inference and batch preprocessing (adds `ultralytics`):

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git@v0.0.5"
```

---

## CLI

**Quick test with the [SCT tutorial image](https://spinalcordtoolbox.com/stable/user_section/tutorials/segmentation/before-starting.html):**

```bash
mkdir ~/sc-crop-test && cd ~/sc-crop-test
curl -L https://github.com/spinalcordtoolbox/sct_tutorial_data/releases/download/r20260508/data_spinalcord-segmentation.zip -o sct_tutorial.zip
unzip sct_tutorial.zip
```

Get the bounding box coordinates:

```bash
sc_crop -i single_subject/data/t2/t2.nii.gz
```

Crop the volume:

```bash
sc_crop -i single_subject/data/t2/t2.nii.gz -o t2_crop.nii.gz --crop
```

You can define a margin around the detected spinal cord (mm, clamped to image boundaries). Defaults: `sup=40, inf=60, left=right=10, ant=post=15`. Priority: individual (e.g. `--pad-sup`) > symmetric (e.g. `--pad-si`) > default.

```bash
sc_crop t2.nii.gz --crop --pad-sup 50 --pad-inf 80 --pad-left 10 --pad-right 10 --pad-ant 15 --pad-post 15
```

```bash
sc_crop t2.nii.gz --crop --pad-si 30 --pad-rl 10 --pad-ap 15
```

Run `sc_crop --help` for all options.

---

## Python API

`detect()` returns the bounding box coordinates and the original image orientation. No files are written — all file I/O is explicit in your code.

### Detect + crop

```python
from sc_crop import detect, crop
import nibabel as nib

bbox = detect("t2.nii.gz")

# crop() works on any volume in the same space (image, label, …)
crop_img   = crop(nib.load("t2.nii.gz"),       bbox)
crop_label = crop(nib.load("t2_label.nii.gz"), bbox)

nib.save(crop_img,   "t2_crop.nii.gz")
nib.save(crop_label, "t2_label_crop.nii.gz")
```

`bbox` contains: `xmin, xmax, ymin, ymax, zmin, zmax` (inclusive, native space), `original_axcodes`.

### Padding

All values define the margin **around the detected spinal cord**, in mm. Priority per face: **individual > symmetric > default**.

```python
# Defaults: superior=40mm, inferior=60mm, left=right=10mm, anterior=posterior=15mm
bbox = detect("t2.nii.gz")

# Adjust SI only (most common)
bbox = detect("t2.nii.gz", pad_superior=50, pad_inferior=80)

# Symmetric SI
bbox = detect("t2.nii.gz", pad_si=30)

# Symmetric + override one face
bbox = detect("t2.nii.gz", pad_si=30, pad_inferior=60)

# Full per-face control
bbox = detect("t2.nii.gz",
             pad_superior=40, pad_inferior=60,
             pad_left=10,     pad_right=10,
             pad_anterior=15, pad_posterior=15)
```

Padding is always clamped to the image boundaries — no out-of-bounds indices are ever produced.

### Restore a segmentation to the original space

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

bbox      = detect("t2.nii.gz")
crop_img = crop(nib.load("t2.nii.gz"), bbox)

seg_crop = my_model(crop_img)              # your model — returns NIfTI in cropped space

seg_full = restore_segmentation(seg_crop, bbox)
nib.save(seg_full, "t2_seg.nii.gz")
```

### Convenience wrapper

```python
from sc_crop import detect_and_crop

crop_nii, bbox = detect_and_crop("t2.nii.gz")  # detect + crop in one call
```

---

## Use in a training + inference pipeline

### ⚠️ Critical: use the same padding at training and inference

sc-crop must be applied with **identical padding around the detected spinal cord** at both training and inference time. Different padding changes the crop boundaries and the distribution of what the model sees, which degrades performance. Pick your padding values once and keep them fixed throughout.

### Step 1 — Preprocess training data

Crop all images and their labels with the same padding:

```python
from sc_crop import detect, crop
import nibabel as nib

# Use the same padding for every subject
PAD = dict(pad_superior=40, pad_inferior=60, pad_left=10, pad_right=10,
           pad_anterior=15, pad_posterior=15)

for subject in subjects:
    bbox        = detect(subject.image, **PAD)
    crop_img   = crop(nib.load(subject.image), bbox)
    crop_label = crop(nib.load(subject.label), bbox)
    nib.save(crop_img,   subject.image_crop)
    nib.save(crop_label, subject.label_crop)
```

### Step 2 — Train your model on the cropped volumes

Train normally on `*_crop.nii.gz` images and labels.

### Step 3 — Inference on a new image

Apply the **same padding**, run your model on the crop, then restore the segmentation to the original space:

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

PAD = dict(pad_superior=40, pad_inferior=60, pad_left=10, pad_right=10,
           pad_anterior=15, pad_posterior=15)  # identical to training

bbox      = detect("new_subject.nii.gz", **PAD)
crop_img = crop(nib.load("new_subject.nii.gz"), bbox)

seg_crop = my_model(crop_img)                  # run your segmentation model
seg_full = restore_segmentation(seg_crop, bbox) # back to original space + affine
nib.save(seg_full, "new_subject_seg.nii.gz")
```

### nnUNet integration

sc-crop fits naturally before nnUNet: crop the raw dataset first, then run the standard nnUNet pipeline on the cropped images.

**Preprocessing (once, before training):**

Use the Python API so that image and label always share the exact same bbox — `detect()` is called once per subject, then `crop()` is applied to both:

```python
from sc_crop import detect, crop
import nibabel as nib

PAD = dict(pad_superior=40, pad_inferior=60, pad_left=10, pad_right=10,
           pad_anterior=15, pad_posterior=15)

for subject in subjects:
    bbox   = detect(subject.image, **PAD)                             # detect once
    nib.save(crop(nib.load(subject.image), bbox), subject.image_crop) # same bbox
    nib.save(crop(nib.load(subject.label), bbox), subject.label_crop) # same bbox
```

**Training:** run `nnUNetv2_plan_and_preprocess` and `nnUNetv2_train` on the cropped dataset as usual.

**Inference on a new image:**

```python
from sc_crop import detect, crop, restore_segmentation
import nibabel as nib

PAD = dict(pad_superior=40, pad_inferior=60, pad_left=10, pad_right=10,
           pad_anterior=15, pad_posterior=15)  # same as training

bbox      = detect("new_subject.nii.gz", **PAD)
crop_img = crop(nib.load("new_subject.nii.gz"), bbox)
nib.save(crop_img, "new_subject_crop.nii.gz")

# Run nnUNet predictor on the cropped image
# nnUNetv2_predict -i new_subject_crop.nii.gz -o seg_crop/ ...

seg_full = restore_segmentation(nib.load("seg_crop/new_subject_crop.nii.gz"), bbox)
nib.save(seg_full, "new_subject_seg.nii.gz")
```

---

## Examples

The `examples/` directory contains two runnable scripts:

**`examples/api_examples.py`** — Python API cookbook covering all usage patterns:

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
| 5 | Fake segmentation model + `restore_segmentation()` round-trip |
| 6 | GPU inference (`use_onnx=False`) |

**`examples/infer_with_sc_crop.py`** — Full inference pipeline template with a built-in fake model (center-cylinder segmentation). Runs out of the box — no real model needed:

```bash
python examples/infer_with_sc_crop.py -i t2.nii.gz -o seg.nii.gz
```

Replace `fake_sc_segmentation()` with your own model to use in production.

---

## Requirements

Python ≥ 3.8. Core dependencies installed automatically: `nibabel`, `numpy`, `pillow`, `pyyaml`, `onnxruntime`.

`ultralytics` is optional — required only for GPU/PyTorch inference (`use_onnx=False`), the debug panel (`debug=True`), and batch preprocessing (`preprocess_dataset`). Install with the `[yolo]` extra above.

---

## Training the detector

The detection model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
