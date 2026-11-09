"""
Segment the spinal cord (SC) of every image of the fullbody-totalsegmentator-mri dataset with SCT
(sct_deepseg spinalcord, on GPU) and save the predictions in the derivatives of the dataset.

Images: the ones with a TotalSegmentator segmentation (derivatives/labels/.../<image name>_label-body_dseg.nii.gz),
i.e. the images used by build_msd_dataset.py.

Outputs (in the dataset folder):
    derivatives/pred-sc-seg/<same path as the image>/<image name>_label-SC_seg.nii.gz
In OUTPUT_DIR:
    segment_sc_totalseg_mri_<YYYYMMDD_HHMMSS>.log : failed images, images with an empty SC segmentation, and
        images where more than 5% of the SC is outside the canal of the TotalSegmentator segmentation (label 46)

Predictions that already exist are not recomputed (unless --overwrite), so the script can be stopped and run again.

Author: Pierre-Louis Benveniste
"""
import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

DSEG_SUFFIX = "_label-body_dseg.nii.gz"
SC_SUFFIX = "_label-SC_seg.nii.gz"
OUTPUT_DIR = Path("derivatives") / "pred-sc-seg"
CANAL_LABEL = 46   # spinal canal in the TotalSegmentator MRI segmentations (canal_label_index in datasets.yaml)
# Images where more than this fraction of the SC is outside the canal are flagged in the log
MAX_SC_OUTSIDE_CANAL = 0.05


def get_parser():
    parser = argparse.ArgumentParser(description="Segment the SC of the fullbody-totalsegmentator-mri images.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="fullbody-totalsegmentator-mri folder.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the log is saved.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Segment the SC again even if the prediction already exists.")
    return parser


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    launched = datetime.now()
    labels_root = args.dataset_dir / "derivatives" / "labels"
    dsegs = sorted(labels_root.rglob(f"*{DSEG_SUFFIX}"))

    failed, empty, outside = [], [], []
    for i, dseg in enumerate(dsegs, 1):
        rel = dseg.parent.relative_to(labels_root)
        image = args.dataset_dir / rel / dseg.name.replace(DSEG_SUFFIX, ".nii.gz")
        output = args.dataset_dir / OUTPUT_DIR / rel / dseg.name.replace(DSEG_SUFFIX, SC_SUFFIX)
        if not image.exists():
            failed.append(f"{image} (image not found)")
            continue
        if output.exists() and not args.overwrite:
            print(f"[{i}/{len(dsegs)}] already done: {output}")
        else:
            print(f"[{i}/{len(dsegs)}] {image}")
            output.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(["sct_deepseg", "spinalcord", "-i", str(image), "-o", str(output)],
                                    env={**os.environ, "SCT_USE_GPU": "1"})
            if result.returncode != 0 or not output.exists():
                failed.append(str(image))
                continue

        # Checks: empty SC, SC outside the canal of the TotalSegmentator segmentation
        sc = np.asarray(nib.load(output).dataobj) > 0
        if not sc.any():
            empty.append(str(output))
            continue
        canal = np.asarray(nib.load(dseg).dataobj) == CANAL_LABEL
        if canal.shape == sc.shape:
            fraction = (sc & ~canal).sum() / sc.sum()
            if fraction > MAX_SC_OUTSIDE_CANAL:
                outside.append(f"{output}: {fraction:.1%} of the SC outside the canal")

    log = args.output_dir / f"segment_sc_totalseg_mri_{launched:%Y%m%d_%H%M%S}.log"
    log.write_text(f"Dataset: {args.dataset_dir.resolve()}\nLaunched: {launched:%Y-%m-%d %H:%M:%S}\n"
                   f"Images: {len(dsegs)}\n"
                   f"Failed images: {len(failed)}\n" + "".join(f"  {f}\n" for f in failed)
                   + f"Empty SC segmentations: {len(empty)}\n" + "".join(f"  {e}\n" for e in empty)
                   + f"SC outside the canal (> {MAX_SC_OUTSIDE_CANAL:.0%}): {len(outside)}\n"
                   + "".join(f"  {o}\n" for o in outside))
    print(f"Done. {len(dsegs)} images, {len(failed)} failed, {len(empty)} empty, {len(outside)} with the SC outside "
          f"the canal (see {log})")


if __name__ == "__main__":
    main()
