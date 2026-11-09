"""
Crop a nifti image around the spinal canal (default) or the spinal cord (SC) detected by a trained YOLO model.

    1. Preprocess the image into axial slices (inference/preprocess_image.py) and detect the structure on each
       slice (inference/predict_image.py)
    2. 3D box = smallest box containing the boxes of all the slices where the structure is detected
    3. Bring the box back to the voxels of the original image and crop the image

The prediction runs on the GPU if there is one, else on the CPU.

Outputs (in OUTPUT_FOLDER, default: folder of the image):
    <name>_crop.nii.gz    : cropped image
    <name>_cropbox.nii.gz : binary mask of the bounding box used, on the grid of the image (FSLeyes overlay)

Author: Pierre-Louis Benveniste
"""
import argparse
import itertools
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from ultralytics import YOLO

# Inference scripts (preprocessing of an image, prediction of its slices)
sys.path.insert(0, str(Path(__file__).resolve().parent / "inference"))
from predict_image import get_thresholds, predict_slices
from preprocess_image import VOXEL_SIZE, cut_slices, resample_image

# Class of each structure in the YOLO model
STRUCTURES = {"sc": 0, "canal": 1}


def get_parser():
    parser = argparse.ArgumentParser(description="Crop a nifti image around the spinal canal or the spinal cord.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Image (nii.gz).")
    parser.add_argument("--model", required=True, type=Path, help="YOLO weights (best.pt of a training).")
    parser.add_argument("--structure", choices=STRUCTURES, default="canal", help="Structure to crop around.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (model trained on RGB slices).")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25],
                        help="Confidence threshold: one value for all classes, or one value per class "
                             "(SC canal, e.g. --conf 0.3 0.5).")
    parser.add_argument("--margin", type=float, nargs="+", default=[0],
                        help="Margin added on both sides of the box (mm): one value for all axes, or one value per "
                             "axis (RL AP SI, e.g. --margin 5 5 20).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder (default: folder of the image).")
    return parser


def get_margins(margin):
    """Return the margin (mm) of each axis (RL, AP, SI) from the --margin values: 1 value (all axes) or 3 values."""
    margin = np.atleast_1d(np.asarray(margin, dtype=float))
    if len(margin) not in (1, 3):
        raise SystemExit(f"--margin: give 1 value or 3 values (RL AP SI), got {len(margin)}")
    return np.broadcast_to(margin, 3)


def find_box(preds, cls, shape, margin):
    """Return the (lower, upper) corners of the 3D box around the boxes of class `cls` of all the slices, plus
    `margin` (mm, see get_margins), in voxel coordinates of the resampled LAS image of shape `shape` (voxel
    centers at integers). Return None if the class is not detected on any slice."""
    detected = [(i, p[cls][0]) for i, p in enumerate(preds) if cls in p]
    if not detected:
        return None
    n_x, n_y, n_z = shape
    x1 = min(box[0] for _, box in detected)
    y1 = min(box[1] for _, box in detected)
    x2 = max(box[2] for _, box in detected)
    y2 = max(box[3] for _, box in detected)
    first, last = detected[0][0], detected[-1][0]
    # Slice i = z (n_z - 1 - i), pixel column u = x (u - 0.5), pixel row v = y (n_y - v - 0.5) (see cut_slices,
    # pixel edges at integers): box edges in voxel coordinates
    lower = np.array([x1 - 0.5, n_y - y2 - 0.5, n_z - 1 - last - 0.5])
    upper = np.array([x2 - 0.5, n_y - y1 - 0.5, n_z - 1 - first + 0.5])
    margin_voxels = get_margins(margin) / np.array(VOXEL_SIZE)
    return lower - margin_voxels, upper + margin_voxels


def to_image_voxels(lower, upper, resampled_affine, image):
    """Return the slices (x, y, z) of the voxels of `image` whose center is inside the box (lower, upper) given in
    voxel coordinates of the resampled image (affine `resampled_affine`)."""
    corners = np.array(list(itertools.product(*zip(lower, upper))))   # 8 corners of the box
    world = nib.affines.apply_affine(resampled_affine, corners)
    voxels = nib.affines.apply_affine(np.linalg.inv(image.affine), world)
    shape = np.array(image.shape[:3])
    # 1e-3: rounding errors of the affines
    start = np.clip(np.ceil(voxels.min(axis=0) - 1e-3), 0, shape).astype(int)
    stop = np.clip(np.floor(voxels.max(axis=0) + 1e-3) + 1, 0, shape).astype(int)
    return tuple(slice(a, b) for a, b in zip(start, stop))


def sc_crop(image, model, output_dir=None, structure="canal", rgb=False, conf=(0.25,), margin=0):
    """Crop `image` around the `structure` ("canal" or "sc") detected by the YOLO weights `model`, and save the
    cropped image and the mask of the box used. Return (cropped image path, mask path)."""
    get_margins(margin)   # check the margin before the prediction
    image = Path(image)
    output_dir = Path(output_dir) if output_dir else image.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    name = image.name.removesuffix(".gz").removesuffix(".nii")

    model = YOLO(model)
    resampled = resample_image(image)
    preds = predict_slices(model, cut_slices(np.asarray(resampled.dataobj), rgb), get_thresholds(conf, model.names))
    box = find_box(preds, STRUCTURES[structure], resampled.shape, margin)
    if box is None:
        raise SystemExit(f"No {structure} detected on {image}: nothing cropped")

    img = nib.load(image)
    crop = to_image_voxels(*box, resampled.affine, img)
    if any(s.stop <= s.start for s in crop):
        raise SystemExit(f"The {structure} box is outside {image}: nothing cropped")
    mask = np.zeros(img.shape[:3], dtype=np.uint8)
    mask[crop] = 1

    crop_path = output_dir / f"{name}_crop.nii.gz"
    mask_path = output_dir / f"{name}_cropbox.nii.gz"
    nib.save(img.slicer[crop], crop_path)
    nib.save(nib.Nifti1Image(mask, img.affine), mask_path)
    n_detected = sum(STRUCTURES[structure] in p for p in preds)
    print(f"{structure} detected on {n_detected}/{len(preds)} slices. Box (voxels of the image): "
          + ", ".join(f"{a}: {s.start}-{s.stop - 1}" for a, s in zip("xyz", crop)))
    print(f"Saved {crop_path} and {mask_path}. To check:\n"
          f"fsleyes {image} {mask_path} -ot mask -mc 1 0 0 --outline -w 2 &")
    return crop_path, mask_path


def main():
    args = get_parser().parse_args()
    sc_crop(args.image, args.model, args.output_dir, args.structure, args.rgb, args.conf, args.margin)


if __name__ == "__main__":
    main()
