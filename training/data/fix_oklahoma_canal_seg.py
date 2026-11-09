"""
Put the gold canal segmentations of dcm-oklahoma on the grid of their image, and overwrite them.

The canal masks of dcm-oklahoma have a different affine from their image. Each mask is resampled onto its
image with sct_register_multimodal -identity 1 -x nn (no registration: the mask is placed in world space with
its own header, then resampled with nearest neighbour). After this, the masks have the shape and affine of
their image.

Canal masks: derivatives/labels/sub-*/.../<image name>_label-canal_seg.nii.gz
Images:      sub-*/.../<image name>.nii.gz

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

CANAL_SUFFIX = "_label-canal_seg.nii.gz"


def get_parser():
    parser = argparse.ArgumentParser(description="Resample the canal segmentations of dcm-oklahoma onto their image "
                                                 "(sct_register_multimodal -identity 1) and overwrite them.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="dcm-oklahoma folder.")
    return parser


def main():
    args = get_parser().parse_args()
    labels_root = args.dataset_dir / "derivatives" / "labels"
    canals = sorted(labels_root.rglob(f"*{CANAL_SUFFIX}"))

    failed = []
    for i, canal in enumerate(canals, 1):
        image = args.dataset_dir / canal.parent.relative_to(labels_root) / canal.name.replace(CANAL_SUFFIX, ".nii.gz")
        print(f"[{i}/{len(canals)}] {canal}")
        if not image.exists():
            failed.append(f"{canal} (no image {image})")
            continue
        # Outputs (resampled mask, warping fields) in a temporary folder next to the mask, then the mask is replaced
        with tempfile.TemporaryDirectory(dir=canal.parent) as tmp:
            result = subprocess.run(["sct_register_multimodal", "-i", str(canal), "-d", str(image), "-identity", "1",
                                     "-x", "nn", "-o", canal.name, "-ofolder", tmp], capture_output=True, text=True)
            output = Path(tmp) / canal.name
            if result.returncode != 0 or not output.exists():
                failed.append(f"{canal} ({result.stderr.strip().splitlines()[-1:] or 'no output'})")
                continue
            os.replace(output, canal)

    print(f"Done: {len(canals) - len(failed)}/{len(canals)} canal segmentations overwritten.")
    for f in failed:
        print(f"  failed: {f}")


if __name__ == "__main__":
    main()
