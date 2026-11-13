# sc-crop — Spinal cord and canal detection and cropping

**sc-crop** detects the spinal canal (or the spinal cord) on a 3D image and crops the image around it, so that segmentation models only see the relevant region. Works on MRI and CT, across contrasts.

<img height="500" src="https://github.com/user-attachments/assets/18218977-f363-48a1-9462-642c2834f701" />

## Install

```bash
conda create -n sc_crop python=3.10 -y
conda activate sc_crop
pip install -r requirements.txt
```

## Crop an image

```bash
python sc_crop.py --image t2.nii.gz --model best.pt
```

| File | Content |
|---|---|
| `t2_crop.nii.gz` | Cropped image |
| `t2_cropbox.nii.gz` | Binary mask of the bounding box used (FSLeyes overlay) |

Check the crop with FSLeyes (example showed above):

```bash
fsleyes t2.nii.gz t2_cropbox.nii.gz -ot mask -mc 1 0 0 --outline -w 3 &
```

## Repository

- `sc_crop.py`: crops an image around the spinal canal or the spinal cord.
- `inference/`: preprocessing of an image into axial slices (`preprocess_image.py`) and detection on these slices (`predict_image.py`).
- `training/`: datasets, training and evaluation of the detection model (see [training/README.md](training/README.md)).
