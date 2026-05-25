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

Add sc_crop to your existing dataset conversion loop (the step that converts image/label pairs to nnUNet format). It detects the spinal cord bbox and crops both image and label before saving — no GT mask required.

```python
from sc_crop import run as sc_crop_run
import nibabel as nib, numpy as np

def crop_nifti(img, xmin, xmax, ymin, ymax, zmin, zmax):
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    affine[:3, 3] = (img.affine @ [xmin, ymin, zmin, 1.0])[:3]
    return nib.Nifti1Image(data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1], affine, img.header)

# Inside your existing conversion loop (after reorienting to RPI):
result = sc_crop_run(image_rpi, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))
xmin, xmax = result['xmin'], result['xmax']
ymin, ymax = result['ymin'], result['ymax']
zmin, zmax = result['zmin'], result['zmax']

nib.save(crop_nifti(nib.load(image_rpi), xmin, xmax, ymin, ymax, zmin, zmax), out_image)
nib.save(crop_nifti(nib.load(label_rpi), xmin, xmax, ymin, ymax, zmin, zmax), out_label)
```

---

## Inference with a model trained with sc_crop

At inference time apply sc_crop with the **same padding as at training time**, run the model on the crop, then paste the segmentation back into the original image space.

```python
from sc_crop import run as sc_crop_run
import nibabel as nib, numpy as np

# 1. Detect bbox and crop
result = sc_crop_run(image_path, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))
xmin, xmax = result['xmin'], result['xmax']
ymin, ymax = result['ymin'], result['ymax']
zmin, zmax = result['zmin'], result['zmax']

original = nib.load(image_path)
data = np.asarray(original.dataobj)
affine = original.affine.copy()
affine[:3, 3] = (original.affine @ [xmin, ymin, zmin, 1.0])[:3]
crop = nib.Nifti1Image(data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1], affine)

# 2. Run your segmentation model on the crop → seg_crop (nib.Nifti1Image)
seg_crop = my_model(crop)

# 3. Restore segmentation to the full original image space
full = np.zeros(original.shape[:3], dtype=np.uint8)
full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = np.asarray(seg_crop.dataobj).astype(np.uint8)
nib.save(nib.Nifti1Image(full, original.affine, original.header), out_path)
```

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
