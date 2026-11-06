"""
Train a YOLO model to detect the spinal cord (SC, class 0) and the spinal canal (class 1) on axial slices.

Input: the YOLO dataset built by data/build_yolo_dataset.py (its dataset.yaml), with
images/{training,validation,test,external_test} and one label file per slice (one "cls cx cy w h" line per class
present on the slice, empty if neither).

Augmentations: YOLO built-in ones (get_augmentations: brightness, rotation, scale, translation, left-right
flip). With --extra-augment, MRI intensity augmentations are added (get_extra_augmentations: downscale,
blur, noise, gamma, bias field, inversion), in place of the default albumentations of YOLO.

At the end of the training, the confidence threshold of each class is chosen on the validation set (best
slice-level F1), then the best model is evaluated with these thresholds on the training, validation, test
and external test sets, with the metrics of each class (see evaluate_detection.py).

Outputs (in RUN_DIR):
    checkpoints/weights/best.pt : best model on the validation set (highest mAP50-95)
    checkpoints/weights/last.pt : model at the last epoch
    checkpoints/                : training curves, metrics and validation plots from YOLO
    evaluation/training_curves.png : losses and validation metrics of each epoch
    evaluation/thresholds.json  : confidence threshold of each class (chosen on the validation set)
    evaluation/thresholds.png   : sensitivity, specificity, precision and F1 vs the threshold, per class
    evaluation/<split>/metrics.json : metrics of all classes (split = training, validation, test, external_test)
    evaluation/<split>/<class>/     : metrics, plots, worst slices and worst volumes of the best model, per class
    training.log                : messages of YOLO and of this script

Inspired from: the train.py script of the previous version of sc-crop

Author: Pierre-Louis Benveniste
"""
import argparse
import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER

from evaluate_detection import evaluate, plot_training_curves, read_dataset, select_thresholds


def get_parser():
    parser = argparse.ArgumentParser(description="Train a YOLO model to detect the spinal cord and canal on axial slices.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="dataset.yaml built by build_yolo_dataset.py.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Folder where checkpoints and logs are saved.")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model to start from (yolo26{n,s,m,l,x}.pt).")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=320, help="Image size used by YOLO (pixels).")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1: as large as the GPU memory allows).")
    parser.add_argument("--device", default="0", help="GPU id(s) (e.g. 0 or 0,1) or cpu.")
    parser.add_argument("--patience", type=int, default=100, help="Epochs without improvement before stopping.")
    parser.add_argument("--workers", type=int, default=2, help="Data loading workers.")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--conf", type=float, default=None,
                        help="Same confidence threshold for all classes in the final evaluation "
                             "(default: one threshold per class, chosen on the validation set).")
    parser.add_argument("--extra-augment", action="store_true",
                        help="Add the MRI intensity augmentations of get_extra_augmentations (albumentations).")
    return parser


def get_augmentations():
    """Return the YOLO built-in augmentations used for training (arguments of model.train).
    No hue/saturation (grayscale or pseudo-RGB slices), no mosaic, no up-down flip (anterior always at the top)."""
    return dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,   # brightness only
        degrees=15.0, scale=0.2, translate=0.1, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=0.0, copy_paste=0.0,
    )


class BiasField(A.ImageOnlyTransform):
    """Multiplicative polynomial bias field, simulating MRI B1 inhomogeneity (e.g. surface coils)."""

    def __init__(self, coef_range=(-0.4, 0.4), order=3, p=0.2):
        super().__init__(p=p)
        self.coef_range = coef_range
        self.order = order

    def apply(self, img, **params):
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        field = np.ones((h, w), dtype=np.float32)
        for i in range(self.order):
            for j in range(self.order - i):
                field += self.random_generator.uniform(*self.coef_range) * x ** i * y ** j
        field = np.clip(field, 0.1, 3.0)
        if img.ndim == 3:
            field = field[:, :, None]   # same field for the 3 channels of pseudo-RGB slices
        return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("coef_range", "order")


class RandomInvert(A.ImageOnlyTransform):
    """Intensity inversion (255 - image), simulating a T1 <-> T2 contrast swap (bright <-> dark CSF)."""

    def apply(self, img, **params):
        return 255 - img

    def get_transform_init_args_names(self):
        return ()


def get_extra_augmentations():
    """Return the extra MRI augmentations (albumentations), used with --extra-augment.
    Intensity only (no geometric transform), so the boxes are unchanged."""
    return [
        # Low in-plane resolution (e.g. DWI, fMRI, thick 2D slices upsampled to 1 mm) and partial volume
        A.Downscale(scale_range=(0.25, 1.0),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=0.25),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # Noisy acquisitions (DWI, fMRI, low field)
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        # Contrast between tissues
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
        BiasField(coef_range=(-0.4, 0.4), order=3, p=0.2),
        # Rare: also makes the background (and black neighbour channels) white
        RandomInvert(p=0.15),
    ]


def use_extra_augmentations():
    """Make YOLO use get_extra_augmentations instead of its default albumentations (tiny blur, gray, CLAHE).
    Must be called before the training, so that the data loading workers inherit it."""
    init = Albumentations.__init__

    def patched_init(self, p=1.0, transforms=None):
        init(self, p=p, transforms=get_extra_augmentations() if transforms is None else transforms)

    Albumentations.__init__ = patched_init


def main():
    args = get_parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    # Save the YOLO messages (and the ones of this script) in a log file of the run
    file_handler = logging.FileHandler(args.run_dir / "training.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(f"Arguments: {vars(args)}")
    LOGGER.info(f"Augmentations: {get_augmentations()}")
    if args.extra_augment:
        use_extra_augmentations()

    model = YOLO(args.model)
    results = model.train(
        data=str(args.dataset.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        project=str(args.run_dir.resolve()),
        name="checkpoints",
        exist_ok=True,
        **get_augmentations(),
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    LOGGER.info(f"Best weights: {best_weights}")

    # Evaluate the best model on every split of the dataset (splits without slices are skipped)
    eval_dir = args.run_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    plot_training_curves(save_dir / "results.csv", eval_dir / "training_curves.png")
    best_model = YOLO(best_weights)
    root = Path(yaml.safe_load(args.dataset.read_text())["path"])
    # Confidence threshold of each class: chosen on the validation set (best F1), or --conf for all classes
    if args.conf is None:
        thresholds = select_thresholds(best_model, args.dataset, eval_dir, args.imgsz, args.device)
    else:
        thresholds = {cls: args.conf for cls in read_dataset(args.dataset, "validation")[2]}
    for split in ("training", "validation", "test", "external_test"):
        evaluate(best_model, args.dataset, split, eval_dir, thresholds, args.imgsz, args.device)


if __name__ == "__main__":
    main()
