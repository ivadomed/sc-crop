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

### Minimal integration — add sc_crop to an existing conversion script

If you already have a script that converts your dataset to nnUNet format (iterating over image/label pairs), adding sc_crop is a **3-line change**: detect the bbox, crop image and label with it, save the crops.

```python
from sc_crop import run as sc_crop_run
import nibabel as nib
import numpy as np

def crop_nifti(img, xmin, xmax, ymin, ymax, zmin, zmax):
    data = np.asarray(img.dataobj)
    cropped = data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1]
    affine = img.affine.copy()
    affine[:3, 3] = (img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
    return nib.Nifti1Image(cropped, affine, img.header)

# In your existing conversion loop:
for image_path, label_path, out_image, out_label in pairs:

    # ── before sc_crop: reorient to RPI (required by nnUNet) ──────────────────
    os.system(f"sct_image -i {image_path} -setorient RPI -o {image_rpi}")
    os.system(f"sct_image -i {label_path} -setorient RPI -o {label_rpi}")

    # ── sc_crop: detect spinal cord bbox (no GT mask needed) ──────────────────
    result = sc_crop_run(image_rpi, padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))
    xmin, xmax = result['xmin'], result['xmax']
    ymin, ymax = result['ymin'], result['ymax']
    zmin, zmax = result['zmin'], result['zmax']

    # ── crop both image and label with the same bbox ───────────────────────────
    nib.save(crop_nifti(nib.load(image_rpi), xmin, xmax, ymin, ymax, zmin, zmax), out_image)
    nib.save(crop_nifti(nib.load(label_rpi), xmin, xmax, ymin, ymax, zmin, zmax), out_label)
```

This is the approach used to train the [contrast-agnostic spinal cord segmentation model v3.0](https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord).
The only change to the training script was replacing the standard `03_convert_msd_to_nnunet_reorient.py` with a version that inserts the sc_crop detection step between reorientation and saving.

### Install with GPU support (large datasets)

GPU batch inference is strongly recommended when preprocessing thousands of volumes. Install `ultralytics`:

```bash
pip install "sc-crop[yolo] @ git+https://github.com/ivadomed/sc-crop.git"
```

For single-image preprocessing (per-image loop as above), the default ONNX CPU backend is sufficient — no GPU required.

### Preprocessing your dataset (batch CLI)

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

### Running inference with a model trained with sc_crop

At inference time, **use the same padding as at training time**. The pipeline is:

```
input image  →  sc_crop detection  →  crop  →  segmentation model  →  restore to original space
```

The segmentation is returned in the exact same space (shape + affine) as the original input image.

#### Option A — PyTorch checkpoint (during development, requires nnunetv2)

```python
import tempfile, shutil, glob
import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from sc_crop import run as sc_crop_run

def segment(image_path: str, out_path: str, model_folder: str,
            pad_rl=10, pad_ap=15, pad_si=30):

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Reorient input to RPI (required by nnUNet)
        img_rpi = f"{tmp}/rpi.nii.gz"
        shutil.copyfile(image_path, f"{tmp}/orig.nii.gz")
        os.system(f"sct_image -i {tmp}/orig.nii.gz -setorient RPI -o {img_rpi}")
        original = nib.load(img_rpi)

        # 2. sc_crop: detect spinal cord bbox
        r = sc_crop_run(img_rpi, padding_rl_mm=pad_rl, padding_ap_mm=pad_ap,
                        padding_si_mm=(pad_si, pad_si))
        xmin, xmax, ymin, ymax, zmin, zmax = r['xmin'], r['xmax'], r['ymin'], r['ymax'], r['zmin'], r['zmax']

        # 3. Crop the image to the detected bbox
        data = np.asarray(original.dataobj)
        aff  = original.affine.copy()
        aff[:3, 3] = (original.affine @ [xmin, ymin, zmin, 1.0])[:3]
        crop = nib.Nifti1Image(data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1], aff)
        crop_path = f"{tmp}/crop.nii.gz"
        nib.save(crop, crop_path)

        # 4. nnUNet inference on cropped volume
        pred_dir = f"{tmp}/pred"
        os.makedirs(pred_dir)
        predictor = nnUNetPredictor(use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=False,
                                    device=torch.device('cpu'))
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=[0],
            checkpoint_name='checkpoint_final.pth',
        )
        predictor.predict_from_files([[crop_path]], pred_dir,
                                     save_probabilities=False, overwrite=True)
        seg_crop = nib.load(glob.glob(f"{pred_dir}/*.nii.gz")[0])

        # 5. Restore segmentation to original (full) image space
        full = np.zeros(original.shape[:3], dtype=np.uint8)
        full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = np.asarray(seg_crop.dataobj).astype(np.uint8)
        nib.save(nib.Nifti1Image(full, original.affine, original.header), out_path)
```

#### Option B — ONNX (deployment, CPU-only, no nnunetv2 needed)

ONNX inference is recommended for deployment: it is ~4× faster than PyTorch on CPU and has no dependency on nnunetv2.
Export your trained nnUNet checkpoint to ONNX once (see your segmentation framework's export script), then:

```python
import onnxruntime as ort
import json, numpy as np, nibabel as nib
from skimage.transform import resize as sk_resize
from sc_crop import run as sc_crop_run

def segment_onnx(image_path: str, out_path: str, onnx_model: str, plans_json: str,
                 pad_rl=10, pad_ap=15, pad_si=30):

    plans  = json.load(open(plans_json))
    target_spacing = plans['configurations']['3d_fullres']['spacing']   # [z, y, x]
    patch_size     = plans['configurations']['3d_fullres']['patch_size'] # [D, H, W]

    # 1. sc_crop: detect bbox and get cropped volume in RPI
    r = sc_crop_run(image_path, padding_rl_mm=pad_rl, padding_ap_mm=pad_ap,
                    padding_si_mm=(pad_si, pad_si))
    crop = nib.load(r['output'])   # already cropped by sc_crop --crop
    orig = nib.load(image_path)
    xmin, xmax, ymin, ymax, zmin, zmax = r['xmin'], r['xmax'], r['ymin'], r['ymax'], r['zmin'], r['zmax']

    # 2. Normalise and resample to training spacing
    data = crop.get_fdata(dtype=np.float32)
    data = (data - data.mean()) / (data.std() + 1e-8)
    zoom = [s / t for s, t in zip(crop.header.get_zooms()[:3], target_spacing[::-1])]
    data_rs = sk_resize(data, [int(round(s * z)) for s, z in zip(data.shape, zoom)],
                        order=3, preserve_range=True).astype(np.float32)

    # 3. Sliding-window inference (single patch for simplicity)
    session = ort.InferenceSession(onnx_model, providers=['CPUExecutionProvider'])
    patch   = data_rs[np.newaxis, np.newaxis]  # (1, 1, D, H, W)
    logits  = session.run(None, {session.get_inputs()[0].name: patch})[0]  # (1, C, D, H, W)
    seg_rs  = (logits[0, 1] > logits[0, 0]).astype(np.uint8)

    # 4. Resample back to crop space and restore to full image space
    seg_crop = sk_resize(seg_rs, data.shape, order=0, preserve_range=True).astype(np.uint8)
    full = np.zeros(orig.shape[:3], dtype=np.uint8)
    full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = seg_crop
    nib.save(nib.Nifti1Image(full, orig.affine, orig.header), out_path)
```

> **Note:** the ONNX example above uses a single patch. For volumes larger than the training patch size, implement sliding-window aggregation with Gaussian weighting (see `nnunetv2` source or the contrast-agnostic inference script for a complete implementation).

A complete minimal template is available in [`examples/infer_with_sc_crop.py`](examples/infer_with_sc_crop.py).

Models are downloaded automatically on first call and cached in `~/.cache/sc_crop/`. SHA256 is verified on every load.

---

## Training

The model was trained using [ivadomed/model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo).
The link between package versions, model versions, and training runs is documented in [VERSIONS.md](VERSIONS.md).
