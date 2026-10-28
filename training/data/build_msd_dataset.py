"""
Aggregate all spinal cord (SC) segmentations of the datasets listed in datasets.yaml into a single
MSD-style json file.

For each SC segmentation, the json stores the image, the label and the following metadata:
dataset, subject, site, pathology, contrast, resolution, image shape, acquisition (2D/3D),
orientation (axial/sagittal/coronal), field strength, region and the value of the SC in the label
(sc_label_value: 1 for binary masks, spinal_cord_label_index from datasets.yaml for multi-class masks).

Outputs (in OUTPUT_FOLDER):
    - sc_seg_msd_<YYYYMMDD>.json : the MSD dataset (absolute paths to images and labels)
    - sc_seg_msd_<YYYYMMDD>.log  : statistics of the dataset and list of skipped segmentations

Usage:
    python build_msd_dataset.py --data-dir <folder with all datasets> --output-dir <output_folder>

The data folder is the output folder of download_all_datasets.sh.

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

# Dataset registry of the repo, next to this script
CONFIG = Path(__file__).parent / "datasets.yaml"
# A volume is almost isotropic (→ 3D) when its thickest voxel side is less than this many times its thinnest one
ISOTROPY_RATIO = 1.2
# Contrasts that are always 3D acquisitions
CONTRASTS_3D = {"MP2RAGE", "UNIT1"}
# First letter of the axis code (from nibabel) → anatomical plane of the slices along that axis
PLANE_FROM_AXIS = {"R": "sagittal", "L": "sagittal", "A": "coronal", "P": "coronal", "S": "axial", "I": "axial"}


def get_parser():
    parser = argparse.ArgumentParser(description="Build an MSD dataset of all SC segmentations.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing all datasets.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the json and log are saved.")
    return parser


def find_image(mask_path, labels_root, dataset_root, mask_suffix, img_glob, image_dirs):
    """Return the image matching a mask, or None if it can't be found."""
    rel_dir = mask_path.parent.relative_to(labels_root)
    if img_glob:
        # Non-standard naming: look for the image in the same subject folder of the raw data
        matches = sorted((dataset_root / rel_dir).glob(img_glob))
        return matches[0] if len(matches) == 1 else None
    # BIDS naming: sub-01_T2w_label-SC_seg.nii.gz → sub-01_T2w.nii.gz, looked for in the raw data,
    # then in the extra image folders (e.g. derivatives/motion_correction for moco images)
    name = mask_path.name.replace(mask_suffix, ".nii.gz")
    for root in [dataset_root] + [dataset_root / d for d in image_dirs]:
        image = root / rel_dir / name
        # is_symlink() keeps git-annex files that are not downloaded, so they are reported as such
        if image.exists() or image.is_symlink():
            return image
    return None


def get_site(subject, dataset, multi_site):
    """Site of a subject: dataset name, or for multi-site datasets <dataset>__<subject name without trailing digits>."""
    if not multi_site:
        return dataset
    site = re.sub(r"\d+$", "", subject.removeprefix("sub-"))
    return f"{dataset}__{site}" if site else dataset


def read_pathologies(dataset_root):
    """Return {participant_id: pathology} from the 'pathology' column of participants.tsv, or {} if absent."""
    tsv = dataset_root / "participants.tsv"
    if not tsv.exists():
        return {}
    with open(tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # Column name may differ in case (pathology, Pathology)
    column = next((c for c in (rows[0] if rows else {}) if c.strip().lower() == "pathology"), None)
    if column is None:
        return {}
    return {r["participant_id"]: r[column].strip() for r in rows if r.get(column, "").strip() not in ("", "n/a")}


def get_image_metadata(image_path, contrast):
    """Read resolution, shape, acquisition (2D/3D), orientation and field strength of an image."""
    img = nib.load(image_path)   # only reads the header
    zooms = [round(float(z), 3) for z in img.header.get_zooms()[:3]]
    shape = [int(s) for s in img.shape[:3]]

    # 3D if almost isotropic or if the contrast is always 3D (MP2RAGE, UNIT1), 2D otherwise
    isotropic = max(zooms) / min(zooms) < ISOTROPY_RATIO
    acquisition = "3D" if isotropic or contrast.upper() in CONTRASTS_3D else "2D"

    # 2D: the highest resolution plane is perpendicular to the axis with the largest voxel size.
    # 3D: no orientation.
    orientation = None
    if acquisition == "2D":
        slice_axis = int(np.argmax(zooms))
        orientation = PLANE_FROM_AXIS[nib.aff2axcodes(img.affine)[slice_axis]]

    # Field strength from the BIDS json sidecar; a missing, empty or invalid sidecar is ignored
    sidecar = Path(str(image_path).replace(".nii.gz", ".json"))
    try:
        info = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        info = {}
    field_strength = info.get("MagneticFieldStrength", "unknown")

    return {"resolution": zooms, "shape": shape, "acquisition": acquisition,
            "orientation": orientation, "field_strength": field_strength}


def collect_dataset(cfg, data_dir, skipped):
    """Return one MSD entry per SC segmentation of a dataset. Problems are appended to `skipped`."""
    name = cfg["name"]
    dataset_root = data_dir / name
    labels_root = dataset_root / "derivatives" / (cfg.get("labels_dir") or "labels")
    if not labels_root.is_dir():
        skipped.append(f"{name}: no folder {labels_root}")
        return []

    # Pathology of each subject, from participants.tsv
    pathologies = read_pathologies(dataset_root)
    if not pathologies:
        skipped.append(f"{name}: no pathology in participants.tsv, set to unknown")

    entries = []
    for mask in sorted(labels_root.rglob(f"*{cfg['mask_suffix']}")):
        # Some datasets (e.g. dcm-brno) have extra derivatives outside anat/ that must not be used
        if cfg.get("anat_only") and mask.parent.name != "anat":
            continue
        # Other labels sharing the mask suffix (e.g. dcm-brno: _label-canal_seg.nii.gz vs _seg.nii.gz)
        if mask.name.endswith(tuple(cfg.get("exclude_suffixes", []))):
            continue
        image = find_image(mask, labels_root, dataset_root, cfg["mask_suffix"], cfg.get("img_glob"),
                           cfg.get("image_dirs", []))
        if image is None:
            skipped.append(f"{name}: no image found for {mask}")
            continue

        subject = re.search(r"sub-[A-Za-z0-9]+", image.name)
        subject = subject.group(0) if subject else "unknown"
        # BIDS suffix of the image: sub-01_acq-sag_T2w.nii.gz → T2w
        # (motion-corrected images: sub-101_run-1_dwi_moco.nii.gz → dwi)
        contrast = image.name.removesuffix(".nii.gz").removesuffix("_moco").split("_")[-1]
        try:
            metadata = get_image_metadata(image, contrast)
        except Exception as e:   # corrupted or partially downloaded image
            skipped.append(f"{name}: could not read {image} ({e})")
            continue

        entries.append({
            "image": str(image),
            "label": str(mask),
            "dataset": name,
            "subject": subject,
            "site": get_site(subject, name, cfg.get("multi_site", False)),
            "pathology": pathologies.get(subject, "unknown"),
            "contrast": contrast,
            "region": cfg.get("region"),
            # Value of the SC in the label: 1 for binary masks, the SC index for multi-class masks
            "sc_label_value": cfg.get("spinal_cord_label_index", 1),
            **metadata,
        })
    return entries


def format_stats(entries, skipped):
    """Return the dataset statistics as a text block."""
    lines = [f"Number of SC segmentations (without skipped) : {len(entries)}",
             f"Number of subjects (without skipped) : {len({(e['dataset'], e['subject']) for e in entries})}",
             f"Number of skipped segmentations/datasets: {len(skipped)}", ""]

    for key in ("dataset", "site", "pathology", "contrast", "acquisition", "orientation", "field_strength", "region"):
        lines.append(f"--- Per {key} ---")
        for value, count in Counter(str(e[key]) for e in entries).most_common():
            lines.append(f"  {value:<40} {count}")
        lines.append("")

    # Sites per dataset (useful to check the multi-site parsing)
    lines.append("--- Sites per dataset ---")
    sites = defaultdict(Counter)
    for e in entries:
        sites[e["dataset"]][e["site"]] += 1
    for dataset, counter in sorted(sites.items()):
        lines.append(f"  {dataset}: " + ", ".join(f"{s} ({c})" for s, c in counter.most_common()))
    lines.append("")

    # Resolution: smallest (in-plane) and largest (slice thickness) voxel size of each image
    if entries:
        lines.append("--- Resolution (mm) ---")
        for label, func in (("smallest voxel side", min), ("largest voxel side", max)):
            values = np.array([func(e["resolution"]) for e in entries])
            lines.append(f"  {label:<20} min {values.min():.2f}  median {np.median(values):.2f}  max {values.max():.2f}")
        lines.append("")

    if skipped:
        lines.append("--- Skipped ---")
        lines.extend(f"  {s}" for s in skipped)
    return "\n".join(lines)


def main():
    args = get_parser().parse_args()
    data_dir = args.data_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().strftime("%Y%m%d")

    with open(CONFIG) as f:
        datasets = yaml.safe_load(f)["datasets"]

    entries, skipped = [], []
    for cfg in datasets:
        print(f"Processing {cfg['name']}...")
        dataset_entries = collect_dataset(cfg, data_dir, skipped)
        print(f"  {len(dataset_entries)} SC segmentations")
        entries.extend(dataset_entries)

    msd = {
        "name": "sc_seg",
        "description": "Spinal cord segmentations aggregated from the datasets of datasets.yaml",
        "date": today,
        "data_dir": str(data_dir),
        # Most labels are binary SC masks; multi-class labels contain other structures too,
        # the SC being the voxels equal to each entry's "sc_label_value"
        "labels": {"0": "background", "1": "spinal cord"},
        "numData": len(entries),
        "data": entries,
    }
    json_path = args.output_dir / f"sc_seg_msd_{today}.json"
    json_path.write_text(json.dumps(msd, indent=4))

    stats = format_stats(entries, skipped)
    log_path = args.output_dir / f"sc_seg_msd_{today}.log"
    log_path.write_text(f"MSD dataset: {json_path}\nData folder: {data_dir}\n\n{stats}\n")

    print(stats)
    print(f"\nSaved {json_path}\nSaved {log_path}")


if __name__ == "__main__":
    main()
