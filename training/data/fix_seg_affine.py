"""
Put the segmentations of one or several datasets on the grid of their image, and overwrite them. Used for the
masks whose affine differs from their image:
    canal masks of dcm-oklahoma and dcm-zurich : --suffix _label-canal_seg.nii.gz (default)
    SC masks of spider-challenge-2023          : --suffix _label-SC_seg.nii.gz

Each mask whose affine differs from its image (same tolerance as build_msd_dataset.py) is resampled onto its image
with sct_register_multimodal -identity 1 -x nn (no registration: the mask is placed in world space with its own
header, then resampled with nearest neighbour). After this, the masks have the shape and affine of their image.
Masks with the same affine as their image are left as they are.

In each dataset folder:
    Masks:  derivatives/labels/sub-*/.../<image name><suffix>
    Images: sub-*/.../<image name>.nii.gz

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description="Resample the segmentations of datasets onto their image "
                                                 "(sct_register_multimodal -identity 1) and overwrite them.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset-dirs", required=True, type=Path, nargs="+",
                        help="Dataset folders (e.g. <data>/dcm-oklahoma <data>/dcm-zurich).")
    parser.add_argument("--suffix", default="_label-canal_seg.nii.gz", help="Suffix of the masks to fix.")
    return parser


def main():
    args = get_parser().parse_args()
    # (dataset folder, mask) of all datasets
    masks = [(d, m) for d in args.dataset_dirs for m in sorted((d / "derivatives" / "labels").rglob(f"*{args.suffix}"))]

    fixed, failed = [], []
    for i, (dataset_dir, mask) in enumerate(masks, 1):
        labels_root = dataset_dir / "derivatives" / "labels"
        image = dataset_dir / mask.parent.relative_to(labels_root) / mask.name.replace(args.suffix, ".nii.gz")
        if not image.exists():
            failed.append(f"{mask} (no image {image})")
            continue
        # Same affine as the image (tolerance of build_msd_dataset.py): nothing to fix
        if np.allclose(nib.load(image).affine, nib.load(mask).affine, atol=1e-3):
            continue
        print(f"[{i}/{len(masks)}] different affine, resampling {mask}")
        # Outputs (resampled mask, warping fields) in a temporary folder next to the mask, then the mask is replaced
        with tempfile.TemporaryDirectory(dir=mask.parent) as tmp:
            result = subprocess.run(["sct_register_multimodal", "-i", str(mask), "-d", str(image), "-identity", "1",
                                     "-x", "nn", "-o", mask.name, "-ofolder", tmp], capture_output=True, text=True)
            output = Path(tmp) / mask.name
            if result.returncode != 0 or not output.exists():
                failed.append(f"{mask} ({result.stderr.strip().splitlines()[-1:] or 'no output'})")
                continue
            os.replace(output, mask)
            fixed.append(mask)

    print(f"Done: {len(masks)} segmentations, {len(fixed)} with a different affine overwritten, "
          f"{len(masks) - len(fixed) - len(failed)} already with the image affine left as they are, {len(failed)} failed.")
    for f in failed:
        print(f"  failed: {f}")


if __name__ == "__main__":
    main()
