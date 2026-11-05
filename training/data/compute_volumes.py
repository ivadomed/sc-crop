"""
Compute the volumes of each image of the MSD dataset (with canal segmentations, from segment_canal.py):
volume of the image, of the SC segmentation and of the canal segmentation, and their ratios.

For each entry of the MSD json:
    image_volume_mm3 : volume covered by the image (number of voxels x voxel volume)
    sc_volume_mm3    : volume of the SC segmentation (voxels equal to sc_label_value), empty if no SC segmentation
    canal_volume_mm3 : volume of the canal: gold segmentation ("canal_seg", voxels equal to canal_label_value
                       for multi-class masks) if there is one, else silver segmentation ("canal_silver").
                       Empty if no canal segmentation.
    sc_ratio         : sc_volume / image_volume
    canal_ratio      : canal_volume / image_volume
Volumes use the voxel size of the image (the segmentations are on the image grid).

Outputs (in OUTPUT_FOLDER):
    volumes_<YYYYMMDD_HHMMSS>.csv : one row per entry
    volumes_<YYYYMMDD_HHMMSS>.log : per dataset (and overall): counts and mean of each value, and the entries
                                    that could not be read

Usage:
    python compute_volumes.py --msd <dataset_msd_..._canal_....json> --output-dir <output_folder>

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np

SPLITS = ("training", "validation", "test", "external_test")
VALUES = ("image_volume_mm3", "sc_volume_mm3", "canal_volume_mm3", "sc_ratio", "canal_ratio")


def get_parser():
    parser = argparse.ArgumentParser(description="Compute the image, SC and canal volumes of the MSD dataset.")
    parser.add_argument("--msd", required=True, type=Path,
                        help="MSD json file with the canal segmentations, built by segment_canal.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the csv and log are saved.")
    return parser


def mask_volume(path, value, voxel_volume):
    """Volume (mm3) of the voxels of a mask equal to `value` (non-zero voxels if value is None)."""
    data = np.asarray(nib.load(path).dataobj)
    return float(((data == value) if value else (data > 0)).sum() * voxel_volume)


def compute_entry(entry):
    """Return the row of the csv of one MSD entry."""
    img = nib.load(entry["image"])
    voxel_volume = float(np.prod(img.header.get_zooms()[:3]))
    image_volume = float(np.prod(img.shape[:3]) * voxel_volume)

    sc_volume = None
    if entry["sc_seg"]:
        sc_volume = mask_volume(entry["sc_seg"], entry["sc_label_value"], voxel_volume)

    # Gold canal segmentation if there is one, else the silver one
    canal_volume, canal_source = None, None
    if entry["canal_seg"]:
        canal_volume = mask_volume(entry["canal_seg"], entry.get("canal_label_value"), voxel_volume)
        canal_source = "gold"
    elif entry.get("canal_silver"):
        canal_volume = mask_volume(entry["canal_silver"], None, voxel_volume)
        canal_source = "silver"

    return {
        "image": entry["image"], "dataset": entry["dataset"], "split": entry["split"],
        "contrast": entry["contrast"], "canal_source": canal_source,
        "image_volume_mm3": image_volume, "sc_volume_mm3": sc_volume, "canal_volume_mm3": canal_volume,
        "sc_ratio": None if sc_volume is None else sc_volume / image_volume,
        "canal_ratio": None if canal_volume is None else canal_volume / image_volume,
    }


def format_stats(rows, failed):
    """Return, per dataset and overall, the counts and the mean of each value, as a text block."""
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    lines = [f"Entries: {len(rows)} ({len(failed)} could not be read)", ""]
    for dataset, dataset_rows in [*sorted(by_dataset.items()), ("ALL DATASETS", rows)]:
        n_gold = sum(r["canal_source"] == "gold" for r in dataset_rows)
        n_silver = sum(r["canal_source"] == "silver" for r in dataset_rows)
        lines.append(f"--- {dataset} ---")
        lines.append(f"  images: {len(dataset_rows)}, with SC segmentation: "
                     f"{sum(r['sc_volume_mm3'] is not None for r in dataset_rows)}, with canal segmentation: "
                     f"{n_gold + n_silver} ({n_gold} gold, {n_silver} silver)")
        for key in VALUES:
            values = [r[key] for r in dataset_rows if r[key] is not None]
            mean = f"{np.mean(values):.4g}" if values else "n/a"
            lines.append(f"  mean {key:<18} {mean:>12}   (n = {len(values)})")
        lines.append("")

    if failed:
        lines.append("--- Could not be read ---")
        lines.extend(f"  {f}" for f in failed)
    return "\n".join(lines)


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    msd = json.loads(args.msd.read_text())
    launched = datetime.now()

    entries = [e for split in SPLITS for e in msd.get(split, [])]
    rows, failed = [], []
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['image']}")
        try:
            rows.append(compute_entry(entry))
        except Exception as e:   # corrupted or missing image or segmentation
            failed.append(f"{entry['image']} ({e})")

    csv_path = args.output_dir / f"volumes_{launched:%Y%m%d_%H%M%S}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "dataset", "split", "contrast", "canal_source", *VALUES])
        writer.writeheader()
        writer.writerows(rows)

    stats = format_stats(rows, failed)
    log_path = args.output_dir / f"volumes_{launched:%Y%m%d_%H%M%S}.log"
    log_path.write_text(f"MSD dataset: {args.msd.resolve()}\nLaunched: {launched:%Y-%m-%d %H:%M:%S}\n\n{stats}\n")
    print(stats)
    print(f"\nSaved {csv_path}\nSaved {log_path}")


if __name__ == "__main__":
    main()
