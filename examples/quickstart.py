"""
sc-crop quickstart — detect, crop, restore.

Demonstrates the three core functions of sc-crop:
  1. detect()               — find the spinal cord bounding box
  2. crop()                 — crop image and label to that box
  3. uncrop() — restore a segmentation back to the original space

Downloads the test T2 image and SC segmentation automatically if not cached.

Requirements:
    pip install "sc-crop @ git+https://github.com/ivadomed/sc-crop.git"

Usage:
    python examples/quickstart.py
    python examples/quickstart.py -i t2.nii.gz -seg t2_seg.nii.gz
"""

import argparse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

from sc_crop import detect, crop, uncrop

_BASE_URL      = "https://github.com/ivadomed/sc-crop/releases/download/test-data"
_CACHE         = Path.home() / ".cache" / "sc_crop" / "tutorial"


def _download(fname: str) -> str:
    _CACHE.mkdir(parents=True, exist_ok=True)
    out = _CACHE / fname
    if not out.exists():
        print(f"Downloading {fname} …")
        urllib.request.urlretrieve(f"{_BASE_URL}/{fname}", out)
    return str(out)


def main():
    p = argparse.ArgumentParser(description="sc-crop quickstart: detect, crop, restore.")
    p.add_argument("-i",   default=None, help="Input T2 image (.nii.gz)")
    p.add_argument("-seg", default=None, help="SC segmentation label (.nii.gz)")
    args = p.parse_args()

    img_path = args.i   or _download("t2.nii.gz")
    seg_path = args.seg or _download("t2_seg.nii.gz")

    # ── 1. Detect ─────────────────────────────────────────────────────────────
    img  = nib.load(img_path)
    bbox = detect(img)                        # pass NIfTI to avoid loading twice
    print(f"bbox  : x[{bbox['xmin']}:{bbox['xmax']}] "
          f"y[{bbox['ymin']}:{bbox['ymax']}] "
          f"z[{bbox['zmin']}:{bbox['zmax']}]")

    # ── 2. Crop image and label with the same bbox ────────────────────────────
    crop_img = crop(img,      bbox)           # NIfTI already loaded
    crop_seg = crop(seg_path, bbox)           # path — loaded automatically
    assert crop_img.shape == crop_seg.shape
    print(f"crop  : {img.shape} → {crop_img.shape}")

    # ── 3. Restore to the original space ──────────────────────────────────────
    seg_full = uncrop(crop_seg, bbox)
    assert seg_full.shape == nib.load(img_path).shape[:3]
    assert np.array_equal(np.asarray(seg_full.dataobj),
                          np.asarray(nib.load(seg_path).dataobj))
    print(f"restore: {crop_seg.shape} → {seg_full.shape}  (round-trip ✓)")


if __name__ == "__main__":
    main()
