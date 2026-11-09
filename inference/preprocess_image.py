"""
Preprocess a nifti image into the axial slices given to the YOLO model, exactly like the slices of the YOLO
dataset (training/data/build_yolo_dataset.py, whose functions are used):
    1. 4D images (e.g. DWI, fMRI) are averaged over time into a 3D image
    2. Reorient to LAS (axis permutation/flip, no interpolation)
    3. Resample to 1 x 1 x 10 mm (RL x AP x SI), linear interpolation
    4. Normalize intensities to uint8: clip to the 0.5-99.5 percentiles of the whole volume, then rescale
    5. Cut axial slices, from Superior (slice_000) to Inferior, row 0 = Anterior, column 0 = patient Right:
       grayscale, or pseudo-RGB with --rgb (R = previous (superior) slice, G = current, B = next (inferior),
       black outside the volume)

Outputs (in OUTPUT_FOLDER, when run as a script):
    slice_NNN.png : the slices (grayscale or RGB png, like the YOLO dataset)

Author: Pierre-Louis Benveniste
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PIL import Image

# Same preprocessing as the YOLO dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "data"))
from build_yolo_dataset import VOXEL_SIZE, normalize, reorient_to_las, target_grid, to_3d


def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess a nifti image into the axial slices given to YOLO.")
    parser.add_argument("--image", required=True, type=Path, help="Image (nii.gz).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the slices are saved.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (for models trained on RGB slices).")
    return parser


def preprocess(image, rgb=False):
    """Return the axial slices of a nifti image (from superior to inferior): list of (rows, columns, 3) uint8
    arrays in BGR order, as YOLO reads the pngs of build_yolo_dataset.py."""
    img = reorient_to_las(to_3d(nib.load(image)))
    data = normalize(resample_from_to(img, target_grid(img, VOXEL_SIZE), order=1).get_fdata(dtype=np.float32))
    # (slice, row, column): slice 0 = most superior, row 0 = Anterior, column 0 = patient Right
    axial = data.transpose(2, 1, 0)[::-1, ::-1, :]
    if rgb:
        # R = previous (superior) slice, G = current, B = next (inferior), black outside the volume
        padded = np.pad(axial, ((1, 1), (0, 0), (0, 0)))
        slices = np.stack([padded[2:], padded[1:-1], padded[:-2]], axis=-1)   # BGR
    else:
        slices = np.repeat(axial[..., None], 3, axis=-1)
    return list(slices)


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slices = preprocess(args.image, args.rgb)
    for i, s in enumerate(slices):
        # BGR → RGB png, or one channel for grayscale
        Image.fromarray(s[:, :, ::-1] if args.rgb else s[:, :, 0]).save(args.output_dir / f"slice_{i:03d}.png")
    print(f"{len(slices)} slices saved in {args.output_dir}")


if __name__ == "__main__":
    main()
