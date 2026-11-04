"""
Convert the MSD dataset with canal predictions (from segment_canal.py) into 2D axial slices with YOLO
bounding boxes of the spinal cord (SC, class 0) and of the spinal canal (class 1).

Same processing as build_yolo_dataset.py, with the canal mask processed like the SC mask:
    1. 4D images (e.g. DWI, fMRI) are averaged over time into a 3D image
    2. Keep only the largest connected component of the SC mask and of the canal mask, then reorient
       image and masks to LAS (axis permutation/flip, no interpolation)
    3. Resample to 1 x 1 x 10 mm (RL x AP x SI): linear for the image, nearest neighbour for the masks
    4. Cut axial slices, from Superior (slice_000) to Inferior.
       Slice = data[:, :, z].T[::-1, :] → row 0 = Anterior, column 0 = patient Right
    5. Normalize intensities to uint8: clip to the 0.5-99.5 percentiles of the whole volume, then rescale
    6. Save each slice as a png (grayscale, or pseudo-RGB with --rgb)
    7. Save the boxes of each slice in YOLO format, one line per class present on the slice:
       "0 cx cy w h" for the SC and "1 cx cy w h" for the canal (normalized to [0, 1]).
       The text file is empty when the slice has neither SC nor canal.

Entries without canal prediction ("canal" field None: sct_deepseg failed) are skipped.
For the external_test split, the canal is the gold canal segmentation (from build_msd_dataset.py).
Canal-only datasets (e.g. CT, where the SC is not visible) have no SC label: their slices only have a canal box,
taken from the gold canal segmentation (voxels equal to "canal_label_value").
Images where more than 5% of the SC is outside the canal are flagged in the log (but kept in the dataset).

Outputs (YOLO layout, in OUTPUT_FOLDER):
    images/<split>/<dataset>__<image name>__slice_NNN.png
    labels/<split>/<dataset>__<image name>__slice_NNN.txt
    dataset.yaml : YOLO (ultralytics) dataset config
    build_yolo_dataset_sc_canal.log : skipped and failed images, images with the SC outside the canal, and
                                      masks with several connected components (only the largest one was kept)

Usage:
    python build_yolo_dataset_sc_canal.py --msd <dataset_msd_YYYYMMDD_canal.json> --output-dir <output_folder> [--rgb]

Author: Pierre-Louis Benveniste
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage
from nibabel.processing import resample_from_to

# Same processing as the SC-only dataset
from build_yolo_dataset import VOXEL_SIZE, normalize, reorient_to_las, target_grid, to_3d

SPLITS = ("training", "validation", "test", "external_test")
CLASSES = {0: "spinal_cord", 1: "canal"}
# Images where more than this fraction of the SC is outside the canal are flagged in the log
MAX_SC_OUTSIDE_CANAL = 0.05


def get_parser():
    parser = argparse.ArgumentParser(description="Convert the MSD dataset into 2D slices with YOLO bounding boxes "
                                                 "of the SC and of the canal.")
    parser.add_argument("--msd", required=True, type=Path,
                        help="MSD json file with the canal predictions, built by segment_canal.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the YOLO dataset is saved.")
    parser.add_argument("--rgb", action="store_true",
                        help="Save pseudo-RGB pngs (R = previous, G = current, B = next slice) "
                             "instead of grayscale pngs of the current slice.")
    return parser


def yolo_bbox(mask_slice, cls):
    """Return the YOLO line "<cls> cx cy w h" of the box around a 2D mask, or "" if the mask is empty."""
    rows, cols = np.nonzero(mask_slice)
    if rows.size == 0:
        return ""
    height, width = mask_slice.shape
    r1, r2, c1, c2 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return f"{cls} {(c1 + c2) / 2 / width:.6f} {(r1 + r2) / 2 / height:.6f} {(c2 - c1) / width:.6f} {(r2 - r1) / height:.6f}\n"


def prepare_mask(mask, affine, name, warnings):
    """Keep the largest connected component of a binary mask (voxels touching by a face, edge or corner),
    reorient it to LAS and resample it to VOXEL_SIZE. Masks with several components are added to `warnings`."""
    # Masks of single volume 4D images (e.g. canal of func images) have a 4th dimension of 1
    mask = mask.reshape(mask.shape[:3])
    components, n_components = ndimage.label(mask, structure=np.ones((3,) * mask.ndim))
    if n_components > 1:
        sizes = np.bincount(components.ravel())[1:]   # [1:] skips the background
        mask = (components == np.argmax(sizes) + 1).astype(np.uint8)
        warnings.append(f"{name}: {n_components} connected components, kept the largest")
    mask = reorient_to_las(nib.Nifti1Image(mask.astype(np.uint8), affine))
    return resample_from_to(mask, target_grid(mask, VOXEL_SIZE), order=0).get_fdata() > 0.5


def process_entry(entry, output_dir, rgb, warnings, sc_outside):
    """Write the png slices and YOLO label files of one MSD entry. Return the number of slices.
    Images where more than MAX_SC_OUTSIDE_CANAL of the SC is outside the canal are added to `sc_outside`."""
    img = reorient_to_las(to_3d(nib.load(entry["image"])))
    canal = nib.load(entry["canal"])
    # Canal = voxels equal to canal_label_value (multi-class gold masks, e.g. CT), else non-zero voxels
    canal_value = entry.get("canal_label_value")
    canal_mask = np.asarray(canal.dataobj) == canal_value if canal_value else np.asarray(canal.dataobj) > 0
    canal_data = prepare_mask(canal_mask, canal.affine, entry["canal"], warnings)

    if entry["label"] is None:
        # Canal-only datasets (e.g. CT, SC not visible): no SC box
        sc_data = np.zeros_like(canal_data)
    else:
        # SC = voxels equal to sc_label_value (multi-class labels)
        label = nib.load(entry["label"])
        sc_data = prepare_mask((np.asarray(label.dataobj) == entry["sc_label_value"]), label.affine,
                               entry["label"], warnings)
        # The SC should be inside the canal (both masks are on the same 1 x 1 x 10 mm grid)
        outside = (sc_data & ~canal_data).sum() / max(sc_data.sum(), 1)
        if outside > MAX_SC_OUTSIDE_CANAL:
            sc_outside.append(f"{entry['image']}: {outside:.1%} of the SC outside the canal")
    img_data = normalize(resample_from_to(img, target_grid(img, VOXEL_SIZE), order=1).get_fdata(dtype=np.float32))

    def get_slice(data, z):
        """Axial slice z (row 0 = Anterior, column 0 = Right), black if z is outside the volume."""
        if not 0 <= z < data.shape[2]:
            return np.zeros((data.shape[1], data.shape[0]), dtype=data.dtype)
        return data[:, :, z].T[::-1, :]

    name = f"{entry['dataset']}__{Path(entry['image']).name.removesuffix('.nii.gz')}"
    n_slices = img_data.shape[2]
    for i in range(n_slices):
        z = n_slices - 1 - i   # slice_000 = most superior slice
        if rgb:
            # R = previous (superior) slice, G = current, B = next (inferior)
            png = np.stack([get_slice(img_data, z + 1), get_slice(img_data, z), get_slice(img_data, z - 1)], axis=2)
        else:
            png = get_slice(img_data, z)
        stem = f"{name}__slice_{i:03d}"
        Image.fromarray(png).save(output_dir / "images" / entry["split"] / f"{stem}.png")
        (output_dir / "labels" / entry["split"] / f"{stem}.txt").write_text(
            yolo_bbox(get_slice(sc_data, z), 0) + yolo_bbox(get_slice(canal_data, z), 1))
    return n_slices


def main():
    args = get_parser().parse_args()
    msd = json.loads(args.msd.read_text())
    output_dir = args.output_dir.resolve()
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    skipped, failed, warnings, sc_outside = [], [], [], []
    for split in SPLITS:
        entries = msd.get(split, [])
        n_slices = 0
        for i, entry in enumerate(entries, 1):
            print(f"[{split} {i}/{len(entries)}] {entry['image']}")
            if not entry.get("canal"):
                skipped.append(entry["image"])
                continue
            try:
                n_slices += process_entry(entry, output_dir, args.rgb, warnings, sc_outside)
            except Exception as e:   # corrupted image, mask not matching the image, ...
                failed.append(f"{entry['image']} ({e})")
        print(f"{split}: {len(entries)} images, {n_slices} slices")

    # YOLO (ultralytics) dataset config
    yolo_cfg = {"path": str(output_dir), "train": "images/training", "val": "images/validation",
                "test": "images/test", "external_test": "images/external_test", "names": CLASSES}
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(yolo_cfg, sort_keys=False))

    log_path = output_dir / "build_yolo_dataset_sc_canal.log"
    log_path.write_text(f"MSD dataset: {args.msd.resolve()}\n"
                        f"Skipped images (no canal prediction): {len(skipped)}\n" + "".join(f"  {s}\n" for s in skipped)
                        + f"Failed images: {len(failed)}\n" + "".join(f"  {f}\n" for f in failed)
                        + f"SC outside the canal (> {MAX_SC_OUTSIDE_CANAL:.0%}): {len(sc_outside)}\n"
                        + "".join(f"  {s}\n" for s in sc_outside)
                        + f"Warnings: {len(warnings)}\n" + "".join(f"  {w}\n" for w in warnings))
    print(f"Done. {len(skipped)} skipped, {len(failed)} failed images, {len(sc_outside)} with SC outside the canal, "
          f"{len(warnings)} warnings (see {log_path})")


if __name__ == "__main__":
    main()
