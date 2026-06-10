# sc-crop — Spinal cord detection and cropping

Segmentation models for spinal cord pathologies (tumors, lesions, SC itself) are typically trained and run on full volumes where the spinal cord occupies only a small fraction of the volume. **sc-crop** solves this by automatically detecting the spinal cord and cropping the volume tightly around it, so your segmentation model only ever sees the relevant region.

This reduces memory usage, speeds up inference, and often improves model accuracy by removing irrelevant background. The recommended workflow is:

1. **Preprocessing** — crop all training images and labels around the detected spinal cord, adding a fixed security margin
2. **Training** — train your segmentation model on the cropped volumes
3. **Inference** — apply the same sc-crop preprocessing to new images, run your model, then optionally restore the segmentation to the original space

Works on both **MRI and CT**, across contrasts (T1, T2, MP2RAGE, DWI…), field strengths, and pathologies. Based on a YOLO26n model trained on multiple datasets covering cervical and lumbar spine. Available as a **CLI tool** and as an **importable Python package**.

<img width="1713" height="727" alt="image" src="https://github.com/user-attachments/assets/d8958227-06b6-4430-9378-4a6f91e9741d" />

---

## Install

Fresh dedicated environment (for testing):

```bash
conda create -n sc_crop python=3.13 && conda activate sc_crop
pip install git+https://github.com/ivadomed/sc-crop.git@v0.4.1
```

Or into an existing environment:

```bash
pip install git+https://github.com/ivadomed/sc-crop.git@v0.4.1
```

*To pin in a `requirements.txt`: add `sc-crop @ git+https://github.com/ivadomed/sc-crop.git@v0.4.1`*

For GPU inference (adds `ultralytics`):

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git@v0.4.1"
```

---

## Command-line interface: detect and crop spinal cord volumes

> The environment where sc-crop was installed must be active for the `sc_crop` command to be available.

### Quick start — download the test data

Image and label are available from the same release:

```bash
mkdir ~/sc-crop-test && cd ~/sc-crop-test
curl -L https://github.com/ivadomed/sc-crop/releases/download/test-data/t2.nii.gz -o t2.nii.gz
curl -L https://github.com/ivadomed/sc-crop/releases/download/test-data/t2_seg.nii.gz -o t2_seg.nii.gz
```

### Crop the image

```bash
sc_crop -i t2.nii.gz
```

Three files are written:

| File | Content |
|---|---|
| `t2_crop.nii.gz` | Cropped image (affine origin updated) |
| `t2_cropbox.nii.gz` | Binary mask of the bounding box used (FSLeyes overlay) |
| `t2_bbox.txt` | Bounding box coordinates in voxel space (human-readable) |

The command also prints a ready-to-use FSLeyes command to visualise the crop and its bounding box:

```bash
fsleyes t2.nii.gz t2_crop.nii.gz t2_cropbox.nii.gz -ot mask -mc 1 0 0 --outline -w 3 &
```

<table><tr>
<td><img width="560" alt="FSLeyes bounding box overlay" src="https://github.com/user-attachments/assets/510fff6e-5436-47d1-89b3-ebe549e38449" /></td>
<td><img width="200" alt="FSLeyes overlay panel" src="https://github.com/user-attachments/assets/4d83d93d-efc5-4695-97f3-0535aec30fce" /></td>
</tr></table>



### Crop a label with the same bounding box

Use the `t2_cropbox.nii.gz` (or `t2_bbox.txt`) produced above to crop any other volume — label, atlas, or additional contrast — with the exact same boundaries:

```bash
sc_crop -i t2_seg.nii.gz --bbox t2_cropbox.nii.gz -o t2_seg_crop.nii.gz
```

### Adjust the bounding box margin

```bash
sc_crop -i t2.nii.gz --pad-sup 50 --pad-inf 80 --pad-left 10 --pad-right 10 --pad-ant 15 --pad-post 15
```

```bash
sc_crop -i t2.nii.gz --pad-si 30 --pad-rl 10 --pad-ap 15
```

Priority: individual (e.g. `--pad-sup`) > symmetric (e.g. `--pad-si`) > default.

Use `--detect` to run detection only (outputs cropbox + bbox.txt, skips the crop step).

Run `sc_crop --help` for all options.

---

## Python API

Three functions cover all use cases:

| Function | Description |
|---|---|
| `detect(img_path)` | Runs the SC detector and returns the bounding box coordinates |
| `crop(img, bbox)` | Crops any NIfTI volume (image or label) to the bounding box |
| `uncrop(seg, bbox)` | Restores a segmentation from the cropped space back to the original full image space |
| `check_label_crop(label, bbox)` | Checks the crop preserves every label voxel; reports the extra padding (mm) needed per face if not |
| `CropReport` | Accumulates per-volume `check_label_crop` results and writes a CSV report + JSON summary |

```python
from sc_crop import detect, crop, uncrop
import nibabel as nib

img  = nib.load("t2.nii.gz")
bbox = detect(img)                              # detect the spinal cord, return bounding box
```

```python
from sc_crop import crop
import nibabel as nib

crop_img = crop(nib.load("t2.nii.gz"), bbox)   # also works on t2_seg.nii.gz
```

```python
from sc_crop import uncrop

full_img = uncrop(crop_img, bbox)               # restore to the original space
```

### Bounding box padding

The margin around the detected spinal cord is adjustable per face (in mm).

```python
# Adjust SI only (most common)
bbox = detect(img, pad_superior=50, pad_inferior=80)

# All 3 symmetric
bbox = detect(img, pad_si=30, pad_rl=10, pad_ap=15)

# Symmetric + override one face
bbox = detect(img, pad_si=30, pad_inferior=60)

# Full per-face control
bbox = detect(img, pad_superior=40, pad_inferior=100,
                   pad_left=15,     pad_right=15,
                   pad_anterior=15, pad_posterior=22)
```

Padding is always clamped to the image boundaries. Priority per face: **individual > symmetric > default** (sup=40, inf=100, left=right=15, ant=15, post=22).

---

## Use in a training + inference pipeline

### ⚠️ Critical: use the same padding at training and inference

sc-crop must be applied with **identical padding around the detected spinal cord** at both training and inference time. Different padding changes the crop boundaries and the distribution of what the model sees, which degrades performance. Pick your padding values once and keep them fixed throughout.

### Step 1 — Preprocess training data

Crop all images and their labels with the same padding. Use `check_label_crop`
and `CropReport` to verify no label voxel is cut by the crop — any loss means the
detected box (plus padding) does not fully contain the cord, which would silently
remove ground-truth voxels from training:

```python
from sc_crop import detect, crop, check_label_crop, CropReport
import nibabel as nib

# Use the same padding for every subject
PAD = dict(pad_superior=40, pad_inferior=100, pad_left=15, pad_right=15,
           pad_anterior=15, pad_posterior=22)

report = CropReport()                       # accumulates per-volume QC

for subject in subjects:
    bbox       = detect(subject.image, **PAD)
    label_nii  = nib.load(subject.label)

    qc = check_label_crop(label_nii, bbox)  # check BEFORE cropping
    report.add(subject.label, qc)           # qc["ok"], voxels_before/after,
                                            # extra_pad_<face>_mm if a face is short

    crop_img   = crop(nib.load(subject.image), bbox)
    crop_label = crop(label_nii, bbox)
    nib.save(crop_img,   subject.image_crop)
    nib.save(crop_label, subject.label_crop)

report.save("crop_qc_report.csv")           # one row per volume
report.save_summary("crop_qc_summary.json") # totals + max extra padding needed per face
print(f"{report.n_failed()} / {len(report)} crops lost label voxels")
```

If `report.n_failed() > 0`, inspect `crop_qc_summary.json`: its `max_extra_padding_mm`
field tells you how many mm to add on each face (e.g. raise `pad_posterior`) so every
cord is fully contained. The defaults above were tuned this way.

### Step 2 — Train your model on the cropped volumes

Train normally on `*_crop.nii.gz` images and labels.

### Step 3 — Inference on a new image

Apply the **same padding**, run your model on the crop, then restore the segmentation to the original space:

```python
from sc_crop import detect, crop, uncrop
import nibabel as nib

PAD = dict(pad_superior=40, pad_inferior=100, pad_left=15, pad_right=15,
           pad_anterior=15, pad_posterior=22)  # identical to training

bbox      = detect("new_subject.nii.gz", **PAD)
crop_img = crop(nib.load("new_subject.nii.gz"), bbox)

seg_crop = my_model(crop_img)                  # run your segmentation model
seg_full = uncrop(seg_crop, bbox) # back to original space + affine
nib.save(seg_full, "new_subject_seg.nii.gz")
```

### nnUNet integration

sc-crop is applied **before** nnUNet's own preprocessing. The cropped images go directly into the nnUNet raw dataset folder; nnUNet never sees the original full volumes.

**Step 1 — Populate `nnUNet_raw/Dataset{ID}_{Name}/`**

nnUNet expects images in `imagesTr/` named `{case}_{0000}.nii.gz` and labels in `labelsTr/` named `{case}.nii.gz`. Crop image and label with the same bbox before saving there:

```python
from sc_crop import detect, crop
import nibabel as nib
from pathlib import Path

RAW = Path("nnUNet_raw/Dataset001_MyTask")
PAD = dict(pad_superior=40, pad_inferior=100, pad_left=15, pad_right=15,
           pad_anterior=15, pad_posterior=22)

for case_id, img_path, lbl_path in subjects:
    bbox = detect(img_path, **PAD)                                          # detect once
    nib.save(crop(nib.load(img_path), bbox),
             RAW / "imagesTr" / f"{case_id}_0000.nii.gz")                  # cropped image
    nib.save(crop(nib.load(lbl_path), bbox),
             RAW / "labelsTr" / f"{case_id}.nii.gz")                       # same bbox
```

**Step 2 — Run the standard nnUNet pipeline**

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
nnUNetv2_train 001 3d_fullres 0
```

**Step 3 — Inference on a new image**

The simplest approach uses the built-in `segment_onnx` / `segment_pt` functions (or their CLI equivalents) which handle detect → crop → infer → uncrop in one call. Install [nnunet-onnx](https://github.com/quentinRevillon/nnunet-onnx) for the inference backend:

```bash
pip install "nnunet-onnx @ git+https://github.com/quentinRevillon/nnunet-onnx.git" torch nnunetv2 onnxscript onnx
```

**Via CLI:**
```bash
# PyTorch checkpoint
sc-segment-pt -i new_subject.nii.gz -o seg.nii.gz \
    --checkpoint /path/to/fold_0/checkpoint_best.pth

# ONNX model (no nnunetv2 needed at runtime)
sc-segment-onnx -i new_subject.nii.gz -o seg.nii.gz \
    --model /path/to/model.onnx
```

**Via Python API:**
```python
from sc_crop import segment_pt, segment_onnx
import nibabel as nib

seg = segment_pt("new_subject.nii.gz", "/path/to/fold_0/checkpoint_best.pth")
seg = segment_onnx("new_subject.nii.gz", "/path/to/model.onnx")
nib.save(seg, "seg.nii.gz")
```

To convert a checkpoint to ONNX (plans are embedded in the file — no external json needed):
```bash
python -m nnunet_onnx.export \
    --checkpoint /path/to/fold_0/checkpoint_best.pth \
    --output model.onnx
```

---

## Examples

The example scripts are part of the repository — clone it to run them:

```bash
git clone https://github.com/ivadomed/sc-crop.git
cd sc-crop
```

**`examples/api_examples.py`** — Python API cookbook covering all usage patterns. If no image is provided, the SCT tutorial T2 is downloaded automatically:

```bash
python examples/api_examples.py                  # auto-download tutorial data
python examples/api_examples.py t2.nii.gz        # use your own image
python examples/api_examples.py t2.nii.gz --ex 3 # run example 3 only (padding)
```

| # | Pattern |
|---|---------|
| 1 | Basic `detect()` + `crop()` |
| 2 | Multi-volume: detect once, crop image + label with the same bbox |
| 3 | Padding variants: default / symmetric / mixed / full individual |
| 4 | `detect_and_crop()` one-liner |
| 5 | Fake segmentation model + `uncrop()` round-trip |
| 6 | GPU inference (`use_onnx=False`) |

**`examples/infer_with_sc_crop.py`** — Full inference pipeline template with a built-in fake model (center-cylinder segmentation). If no input is provided, the SCT tutorial T2 is downloaded automatically:

```bash
python examples/infer_with_sc_crop.py                        # auto-download tutorial data
python examples/infer_with_sc_crop.py -i t2.nii.gz -o seg.nii.gz
```

Replace `fake_sc_segmentation()` with your own model to use in production.

---

## Requirements

Python ≥ 3.8. Core dependencies installed automatically: `nibabel`, `numpy`, `pillow`, `pyyaml`, `onnxruntime`, `opencv-python-headless`, `scipy`.

`ultralytics` is optional — required only for GPU/PyTorch inference (`use_onnx=False`) and the debug panel (`debug=True`). Install with the `[yolo]` extra above.

---

## Training the detector

The detection model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
To publish a new model version, see the release procedure in [VERSIONS.md](VERSIONS.md#procédure-de-release).
