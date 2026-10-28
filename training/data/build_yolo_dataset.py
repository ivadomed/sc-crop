"""
Convert the MSD dataset (from build_msd_dataset.py) into 2D axial slices with YOLO bounding boxes.

For each (image, label) pair of the MSD json:
    1. 4D images (e.g. DWI, fMRI) are averaged over time into a 3D image
    2. Reorient image and label to LAS (axis permutation/flip, no interpolation)
    3. Resample to 1 x 1 x 10 mm (RL x AP x SI): linear for the image, nearest neighbour for the SC mask
    4. Cut axial slices, from Superior (slice_000) to Inferior.
       Slice = data[:, :, z].T[::-1, :] → row 0 = Anterior, column 0 = patient Right
    5. Normalize intensities to uint8: clip to the 0.5-99.5 percentiles of the whole volume, then rescale
    6. Save each slice as a png:
       - default: grayscale (1 channel), the current slice only
       - with --rgb: pseudo-RGB, R = previous (superior) slice, G = current slice, B = next (inferior)
         slice. Missing neighbours (first/last slice) are black.
    7. Save the SC bounding box of each slice in YOLO format: "0 cx cy w h" (normalized to [0, 1]).
       The text file is empty when the slice has no SC.

Outputs (YOLO layout, in OUTPUT_FOLDER):
    images/<split>/<dataset>__<image name>__slice_NNN.png
    labels/<split>/<dataset>__<image name>__slice_NNN.txt
    dataset.yaml : YOLO (ultralytics) dataset config
    build_yolo_dataset.log : images that could not be processed, and warnings (still processed):
                             SC masks not made of one connected component, labels not on the image grid

Usage:
    python build_yolo_dataset.py --msd <dataset_msd_YYYYMMDD.json> --output-dir <output_folder> [--rgb]

Author: Pierre-Louis Benveniste
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nibabel.processing import resample_from_to
from PIL import Image
from scipy import ndimage

# Target voxel size in LAS (RL, AP, SI), in mm
VOXEL_SIZE = (1.0, 1.0, 10.0)
# Intensity percentiles used for the normalization
PERCENTILES = (0.5, 99.5)
SPLITS = ("training", "validation", "test")


def get_parser():
    parser = argparse.ArgumentParser(description="Convert the MSD dataset into 2D slices with YOLO bounding boxes.")
    parser.add_argument("--msd", required=True, type=Path, help="MSD json file built by build_msd_dataset.py.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the YOLO dataset is saved.")
    parser.add_argument("--rgb", action="store_true",
                        help="Save pseudo-RGB pngs (R = previous, G = current, B = next slice) "
                             "instead of grayscale pngs of the current slice.")
    return parser


def to_3d(img):
    """Average 4D images over time (last axis) into a 3D image."""
    if img.ndim == 4:
        return nib.Nifti1Image(img.get_fdata(dtype=np.float32).mean(axis=-1), img.affine)
    return img


def reorient_to_las(img):
    """Reorient to LAS by permuting/flipping the axes (no interpolation)."""
    transform = ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "A", "S")))
    return img.as_reoriented(transform)


def target_grid(img, voxel_size):
    """Return (shape, affine) of the grid covering `img` with `voxel_size`, keeping its orientation."""
    zooms = np.array(img.header.get_zooms()[:3])
    new_zooms = np.array(voxel_size)
    shape = np.maximum(np.round(np.array(img.shape[:3]) * zooms / new_zooms), 1).astype(int)
    affine = img.affine.copy()
    affine[:3, :3] = img.affine[:3, :3] * (new_zooms / zooms)   # scale each axis, keep the LAS directions
    # Shift the origin so that the new grid covers the same extent as the original one
    affine[:3, 3] = img.affine[:3, :3] @ ((new_zooms / zooms - 1) / 2) + img.affine[:3, 3]
    return tuple(shape), affine


def normalize(data):
    """Clip to the volume percentiles and rescale to uint8."""
    lo, hi = np.percentile(data, PERCENTILES)
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return ((np.clip(data, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def yolo_bbox(mask_slice):
    """Return the YOLO line "0 cx cy w h" of the SC in a 2D mask, or "" if there is no SC."""
    rows, cols = np.nonzero(mask_slice)
    if rows.size == 0:
        return ""
    height, width = mask_slice.shape
    r1, r2, c1, c2 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return f"0 {(c1 + c2) / 2 / width:.6f} {(r1 + r2) / 2 / height:.6f} {(c2 - c1) / width:.6f} {(r2 - r1) / height:.6f}\n"


def process_entry(entry, output_dir, rgb, warnings):
    """Write the png slices and YOLO label files of one MSD entry. Return the number of slices.
    SC masks that are not one connected component or not on the image grid are added to `warnings`
    (but still processed)."""
    img = reorient_to_las(to_3d(nib.load(entry["image"])))
    label = nib.load(entry["label"])
    # Keep only the SC (multi-class labels: SC = voxels equal to sc_label_value)
    sc = (np.asarray(label.dataobj) == entry["sc_label_value"]).astype(np.uint8)

    # The SC should be one connected component (voxels touching by a face, edge or corner)
    _, n_components = ndimage.label(sc, structure=np.ones((3,) * sc.ndim))
    if n_components != 1:
        warnings.append(f"{entry['label']}: {n_components} SC connected components")

    mask = reorient_to_las(nib.Nifti1Image(sc, label.affine))

    # Image and SC mask should be on the same grid (same shape and affine)
    if img.shape[:3] != mask.shape[:3]:
        warnings.append(f"{entry['label']}: image and label shapes differ ({img.shape[:3]} vs {mask.shape[:3]})")
    elif not np.allclose(img.affine, mask.affine, atol=1e-3):
        warnings.append(f"{entry['label']}: image and label affines differ")

    img_data = normalize(resample_from_to(img, target_grid(img, VOXEL_SIZE), order=1).get_fdata(dtype=np.float32))
    mask_data = resample_from_to(mask, target_grid(mask, VOXEL_SIZE), order=0).get_fdata() > 0.5

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
        (output_dir / "labels" / entry["split"] / f"{stem}.txt").write_text(yolo_bbox(get_slice(mask_data, z)))
    return n_slices


def main():
    args = get_parser().parse_args()
    msd = json.loads(args.msd.read_text())
    output_dir = args.output_dir.resolve()
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    failed, warnings = [], []
    for split in SPLITS:
        entries = msd.get(split, [])
        n_slices = 0
        for i, entry in enumerate(entries, 1):
            print(f"[{split} {i}/{len(entries)}] {entry['image']}")
            try:
                n_slices += process_entry(entry, output_dir, args.rgb, warnings)
            except Exception as e:   # corrupted image, label not matching the image, ...
                failed.append(f"{entry['image']} ({e})")
        print(f"{split}: {len(entries)} images, {n_slices} slices")

    # YOLO (ultralytics) dataset config
    yolo_cfg = {"path": str(output_dir), "train": "images/training", "val": "images/validation",
                "test": "images/test", "names": {0: "spinal_cord"}}
    (output_dir / "dataset.yaml").write_text(yaml.safe_dump(yolo_cfg, sort_keys=False))

    log_path = output_dir / "build_yolo_dataset.log"
    log_path.write_text(f"MSD dataset: {args.msd.resolve()}\n"
                        f"Failed images: {len(failed)}\n" + "".join(f"  {f}\n" for f in failed)
                        + f"Warnings: {len(warnings)}\n" + "".join(f"  {w}\n" for w in warnings))
    print(f"Done. {len(failed)} failed images, {len(warnings)} warnings (see {log_path})")


if __name__ == "__main__":
    main()
