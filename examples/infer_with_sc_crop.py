"""
Inference pipeline template: sc-crop detection + segmentation model.

Shows the canonical 4-step pattern for running any segmentation model on
spinal cord MRI using sc-crop for automatic cropping and space restoration.

A runnable fake model is provided (center-cylinder mask) so the full pipeline
can be tested without a real model. Replace fake_sc_segmentation() with your
own model to use in production.

Requirements:
    pip install "sc-crop @ git+https://github.com/ivadomed/sc-crop.git"
    # + your model's requirements (nnunetv2, onnxruntime, etc.)

Usage:
    python examples/infer_with_sc_crop.py -i t2.nii.gz -o seg.nii.gz
    python examples/infer_with_sc_crop.py -i t2.nii.gz -o seg.nii.gz --fake
"""

import argparse

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

from sc_crop import detect_and_crop, restore_segmentation


# ── Orientation helpers ────────────────────────────────────────────────────────

def reorient_to_rpi(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI image to RPI (standard orientation for many SC models)."""
    rpi  = axcodes2ornt(("R", "P", "I"))
    curr = io_orientation(img.affine)
    return img.as_reoriented(ornt_transform(curr, rpi))


def reorient_back(img_rpi: nib.Nifti1Image,
                  original_orientation: np.ndarray) -> nib.Nifti1Image:
    """Reorient back from RPI to the original orientation."""
    rpi = axcodes2ornt(("R", "P", "I"))
    return img_rpi.as_reoriented(ornt_transform(rpi, original_orientation))


# ── Fake model (runnable demo — replace with your own) ────────────────────────

def fake_sc_segmentation(crop_rpi: nib.Nifti1Image) -> nib.Nifti1Image:
    """Demo fake model: returns a cylinder mask centered in the crop.

    After detect_and_crop() the spinal cord sits near the center of the RL
    and AP axes, so a small central cylinder approximates where a real model
    would place the SC segmentation. This lets you verify the full pipeline
    (crop → model → restore) in FSLeyes without a real model.

    Replace this function with your actual segmentation model.
    Input : crop_rpi — NIfTI image in RPI orientation, cropped around the SC.
    Output: binary segmentation NIfTI in the same space (RPI, same shape).
    """
    data = np.asarray(crop_rpi.dataobj)
    X, Y, Z = data.shape

    # Small cylinder centered in RL (axis 0) and AP (axis 1), full SI extent
    cx, cy = X / 2, Y / 2
    radius = max(1, min(X, Y) // 6)
    xx, yy = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    cylinder = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2)

    mask = np.zeros((X, Y, Z), dtype=np.uint8)
    mask[cylinder] = 1

    return nib.Nifti1Image(mask, crop_rpi.affine)


# ── Inference pipeline ─────────────────────────────────────────────────────────

def infer(input_path: str, output_path: str, use_fake: bool = False) -> None:
    """Full inference pipeline: detect → crop → model → restore.

    Step 1 — Detect SC and get cropped image in the original orientation.
    Step 2 — Reorient to RPI (skip if your model does not require it).
    Step 3 — Run your segmentation model (returns a binary NIfTI in RPI space).
    Step 4 — Reorient segmentation back and restore to the full image space.
    """
    # Step 1
    crop_nii, ctx        = detect_and_crop(input_path)
    original_orientation = io_orientation(crop_nii.affine)

    # Step 2
    crop_rpi = reorient_to_rpi(crop_nii)

    # Step 3 — swap fake_sc_segmentation for your real model here
    if use_fake:
        print("Using fake cylinder segmentation (demo mode)")
        seg_rpi = fake_sc_segmentation(crop_rpi)
    else:
        seg_rpi = your_model(crop_rpi)  # noqa: F821 — replace this

    # Step 4
    seg_crop = reorient_back(seg_rpi, original_orientation)
    seg_full = restore_segmentation(seg_crop, ctx)

    nib.save(seg_full, output_path)

    n = int(np.asarray(seg_full.dataobj).sum())
    print(f"Segmentation : {n} voxels  shape={seg_full.shape}  → {output_path}")
    print("Open in FSLeyes: fsleyes " + input_path + " " + output_path + " -cm red")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Spinal cord segmentation with sc-crop detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", required=True, help="Input NIfTI image (.nii or .nii.gz)")
    p.add_argument("-o", required=True, help="Output segmentation mask (.nii.gz)")
    p.add_argument("--fake", action="store_true",
                   help="Use the built-in fake cylinder model (demo mode, no real model needed)")
    args = p.parse_args()
    infer(args.i, args.o, use_fake=args.fake)


if __name__ == "__main__":
    main()
