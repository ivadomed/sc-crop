"""
Train a YOLO model to detect the spinal cord (SC) on axial MRI slices.

Input: the YOLO dataset built by data/build_yolo_dataset.py (its dataset.yaml), with
images/{training,validation,test} and one label file per slice ("0 cx cy w h", empty if no SC).

Augmentations:
    - YOLO built-in: brightness (hsv_v=0.15), rotation (15°), scale (0.2), translation (0.1),
      left-right and up-down flips (0.5). No mosaic, no colour changes (hue/saturation).
    - with --extra-augment: MRI-specific augmentations added to the YOLO pipeline (albumentations):
      zoom, Gaussian noise, blur, downscaling, gamma, bias field, contrast inversion, contrast.
    - with --no-augment: no augmentation at all.

Outputs (in RUN_DIR):
    checkpoints/weights/best.pt : best model on the validation set
                                  (YOLO fitness = 0.1 * mAP50 + 0.9 * mAP50-95)
    checkpoints/weights/last.pt : model at the last epoch
    checkpoints/                : training curves, metrics and validation plots from YOLO

Usage:
    python train_detection.py --dataset <yolo_dataset>/dataset.yaml --run-dir <run_dir> [--extra-augment] [--wandb]

Inspired from: the train.py script of the previous version of sc-crop

Author: Pierre-Louis Benveniste
"""
import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def get_parser():
    parser = argparse.ArgumentParser(description="Train a YOLO model to detect the spinal cord on axial slices.",
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
    parser.add_argument("--no-augment", action="store_true", help="Disable all augmentations.")
    parser.add_argument("--extra-augment", action="store_true", help="Add MRI-specific augmentations.")
    parser.add_argument("--wandb", action="store_true", help="Log the training to Weights & Biases.")
    parser.add_argument("--wandb-project", default="sc_crop_detection", help="Weights & Biases project.")
    return parser


class BiasField(A.ImageOnlyTransform):
    """Smooth multiplicative intensity field, simulating MRI coil inhomogeneity."""

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
                field += np.random.uniform(*self.coef_range) * x**i * y**j
        field = np.clip(field, 0.1, 3.0)
        if img.ndim == 3:
            field = field[:, :, None]
        return np.clip(img.astype(np.float32) * field, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("coef_range", "order")


class RandomInvert(A.ImageOnlyTransform):
    """Invert intensities (255 - img), simulating a T1w <-> T2w contrast swap."""

    def apply(self, img, **params):
        return 255 - img

    def get_transform_init_args_names(self):
        return ()


def add_mri_augmentations():
    """Add MRI-specific augmentations to the albumentations step of the YOLO training pipeline.
    Must be called before the model is created, so that data loading workers use it."""
    from ultralytics.data.augment import Albumentations

    extra = [
        A.Affine(scale=(0.25, 4.0), translate_percent=(-0.3, 0.3), p=0.2),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.Downscale(scale_range=(0.25, 1.0),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=0.25),
        A.RandomGamma(gamma_limit=(80, 120), p=0.1),
        BiasField(p=0.2),
        RandomInvert(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.3, 0.3), p=0.15),
    ]

    original_init = Albumentations.__init__

    def init_with_mri_augmentations(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        existing = list(self.transform.transforms) if self.transform is not None else []
        # bbox_params: boxes are moved along with the spatial transforms (Affine)
        self.transform = A.Compose(existing + extra,
                                   bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
        self.contains_spatial = True

    Albumentations.__init__ = init_with_mri_augmentations


def main():
    args = get_parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Weights & Biases: YOLO logs to the run opened here (named after the run folder)
    SETTINGS["wandb"] = args.wandb
    if args.wandb:
        import importlib
        import ultralytics.utils.callbacks.wb as wb_callback
        import wandb
        importlib.reload(wb_callback)   # the callback checks SETTINGS["wandb"] when it is imported
        wandb.init(project=args.wandb_project, name=args.run_dir.name,
                   config={k: str(v) for k, v in vars(args).items()})

    if args.extra_augment and not args.no_augment:
        add_mri_augmentations()

    # YOLO built-in augmentations. No hue/saturation (grayscale or pseudo-RGB slices), no mosaic.
    if args.no_augment:
        augmentations = dict(hsv_v=0.0, degrees=0.0, scale=0.0, translate=0.0, fliplr=0.0, flipud=0.0)
    else:
        augmentations = dict(hsv_v=0.15, degrees=15.0, scale=0.2, translate=0.1, fliplr=0.5, flipud=0.5)

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
        hsv_h=0.0, hsv_s=0.0, shear=0.0, perspective=0.0, mosaic=0.0, copy_paste=0.0,
        **augmentations,
    )

    save_dir = Path(results.save_dir) if results is not None else Path(model.trainer.save_dir)
    print(f"\nBest weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
