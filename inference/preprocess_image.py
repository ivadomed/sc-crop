"""
Preprocess a nifti image into the axial slices given to the YOLO model, like the slices of the YOLO dataset
(training/data/build_yolo_dataset.py, whose functions are used):
    1. 4D images (e.g. DWI, fMRI) are averaged over time into a 3D image
    2. Reorient to LAS (axis permutation/flip, no interpolation)
    3. Resample to 1 x 1 x SI_RES mm (RL x AP x SI), linear interpolation. SI_RES is 1 mm by default (--si-res).
       The training slices (YOLO dataset) are 1 x 1 x 10 mm (--si-res 10): the slices have the same in-plane size
       (1 x 1 mm) whatever SI_RES, there are just more of them along SI with a smaller SI_RES, so the SI extent
       of the structure is found with a SI_RES precision.
    4. Normalize intensities to uint8: clip to the 0.5-99.5 percentiles of the whole volume, then rescale
    5. Cut axial slices, from Superior (slice_000) to Inferior, row 0 = Anterior, column 0 = patient Right:
       grayscale, or pseudo-RGB with --rgb (R = superior slice, G = current, B = inferior slice, black outside
       the volume). The R and B slices are ~10 mm away from the current one, like in the YOLO dataset (e.g. 10
       slices away at 1 mm, see rgb_offset).

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

# Default SI voxel size (slice thickness) in mm of the images given to the model: 1 mm iso. The in-plane voxel
# size is the one of the training slices (VOXEL_SIZE of build_yolo_dataset.py: 1 x 1 x 10 mm).
SI_RES = 1.0


def positive_float(value):
    """argparse type: float > 0."""
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return value


def add_si_res_argument(parser):
    """Add the --si-res option (SI voxel size of the resampling) to `parser`."""
    parser.add_argument("--si-res", type=positive_float, default=SI_RES,
                        help=f"SI voxel size (slice thickness, mm) of the resampling: 1 x 1 x SI_RES mm (RL AP SI). "
                             f"The training slices are 1 x 1 x {VOXEL_SIZE[2]:g} mm.")


def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess a nifti image into the axial slices given to YOLO.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Image (nii.gz).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the slices are saved.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (for models trained on RGB slices).")
    add_si_res_argument(parser)
    return parser


def resample_image(image, si_res=SI_RES):
    """Return the nifti image averaged over time (4D), reoriented to LAS, resampled to 1 x 1 x `si_res` mm and
    normalized to uint8 (steps 1 to 4), as a nibabel image (its affine gives the position of the slices in the
    image, its zooms the voxel size)."""
    img = reorient_to_las(to_3d(nib.load(image)))
    shape, affine = target_grid(img, (VOXEL_SIZE[0], VOXEL_SIZE[1], si_res))
    data = normalize(resample_from_to(img, (shape, affine), order=1).get_fdata(dtype=np.float32))
    return nib.Nifti1Image(data, affine)


def rgb_offset(voxel_size):
    """Return the number of slices between the current slice and the R / B slices of the pseudo-RGB slices, so
    that they are about as far apart (in mm) as in the YOLO dataset: 10 at 1 mm iso, 1 at 1 x 1 x 10 mm."""
    return max(1, round(VOXEL_SIZE[2] / voxel_size[2]))


def cut_slices(data, rgb=False, offset=1):
    """Return the axial slices of LAS data (step 5), from superior to inferior: list of (rows, columns, 3) uint8
    arrays in BGR order, as YOLO reads the pngs of build_yolo_dataset.py. With `rgb`, the R and B channels are
    the slices `offset` slices above and below (see rgb_offset)."""
    # (slice, row, column): slice 0 = most superior, row 0 = Anterior, column 0 = patient Right
    axial = data.transpose(2, 1, 0)[::-1, ::-1, :]
    if rgb:
        # R = superior slice, G = current, B = inferior slice, black outside the volume
        n = len(axial)
        padded = np.pad(axial, ((offset, offset), (0, 0), (0, 0)))
        slices = np.stack([padded[2 * offset:], padded[offset:offset + n], padded[:n]], axis=-1)   # BGR
    else:
        slices = np.repeat(axial[..., None], 3, axis=-1)
    return list(slices)


def image_slices(resampled, rgb=False):
    """Return the axial slices (step 5, see cut_slices) of an image returned by resample_image."""
    return cut_slices(np.asarray(resampled.dataobj), rgb, rgb_offset(resampled.header.get_zooms()[:3]))


def preprocess(image, rgb=False, si_res=SI_RES):
    """Return the axial slices of a nifti image (steps 1 to 5), see resample_image and cut_slices."""
    return image_slices(resample_image(image, si_res), rgb)


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slices = preprocess(args.image, args.rgb, args.si_res)
    for i, s in enumerate(slices):
        # BGR → RGB png, or one channel for grayscale
        Image.fromarray(s[:, :, ::-1] if args.rgb else s[:, :, 0]).save(args.output_dir / f"slice_{i:03d}.png")
    print(f"{len(slices)} slices saved in {args.output_dir}")


if __name__ == "__main__":
    main()
