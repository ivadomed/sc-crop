"""
Minimal inference template: sc-crop detection + your segmentation model.

This script shows the canonical 5-step pattern for running any segmentation model
on spinal cord MRI using sc-crop for automatic cropping and space restoration.

The pattern works for any model architecture (nnUNet, ONNX, PyTorch, etc.).

Requirements:
    pip install "sc-crop @ git+https://github.com/ivadomed/sc-crop.git"
    # + your model's requirements (nnunetv2, onnxruntime, etc.)

Usage:
    python infer_with_sc_crop.py -i t2.nii.gz -o seg.nii.gz
"""

import argparse
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from sc_crop import detect_and_crop, restore_segmentation


# ── Reorientation helpers ──────────────────────────────────────────────────────

def reorient_to_rpi(img):
    """Reorient a NIfTI image to RPI (required by nnUNet)."""
    rpi  = axcodes2ornt(('R', 'P', 'I'))
    curr = io_orientation(img.affine)
    return img.as_reoriented(ornt_transform(curr, rpi))


def reorient_back(img_rpi, original_orientation):
    """Reorient back from RPI to the original orientation."""
    rpi = axcodes2ornt(('R', 'P', 'I'))
    return img_rpi.as_reoriented(ornt_transform(rpi, original_orientation))


# ── Your model goes here ───────────────────────────────────────────────────────

def run_my_model(crop_rpi: nib.Nifti1Image) -> nib.Nifti1Image:
    """Replace this with your own segmentation model.

    Input : crop_rpi — NIfTI image in RPI orientation, cropped around the SC.
    Output: binary segmentation NIfTI in the same space (RPI, same shape as input).

    Example implementations:
        - ONNX : use onnxruntime.InferenceSession (no nnunetv2 needed)
        - PT   : use nnUNetPredictor from nnunetv2
    """
    raise NotImplementedError("Replace this with your actual model inference.")


# ── Main inference pipeline ───────────────────────────────────────────────────

def infer(input_path: str, output_path: str) -> None:
    """Full inference pipeline: detect → crop → model → restore."""

    # Step 1 — sc-crop: detect SC and get cropped image in original orientation
    crop_nii, ctx = detect_and_crop(input_path)
    original_orientation = io_orientation(crop_nii.affine)

    # Step 2 — Reorient crop to RPI (required by nnUNet; skip if your model doesn't need it)
    crop_rpi = reorient_to_rpi(crop_nii)

    # Step 3 — Run your segmentation model (returns segmentation in RPI, same shape as crop_rpi)
    seg_rpi = run_my_model(crop_rpi)

    # Step 4 — Reorient segmentation back to the original orientation
    seg_crop = reorient_back(seg_rpi, original_orientation)

    # Step 5 — Restore segmentation to the full original image space
    seg_full = restore_segmentation(seg_crop, ctx)
    nib.save(seg_full, output_path)
    print(f"Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Spinal cord segmentation with sc-crop detection")
    p.add_argument("-i", required=True, help="Input NIfTI image (.nii or .nii.gz)")
    p.add_argument("-o", required=True, help="Output segmentation mask (.nii.gz)")
    args = p.parse_args()
    infer(args.i, args.o)


if __name__ == "__main__":
    main()
