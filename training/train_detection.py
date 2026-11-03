"""
Train a YOLO model to detect the spinal cord (SC) on axial MRI slices.

Input: the YOLO dataset built by data/build_yolo_dataset.py (its dataset.yaml), with
images/{training,validation,test} and one label file per slice ("0 cx cy w h", empty if no SC).

Outputs (in RUN_DIR):
    checkpoints/weights/best.pt : best model on the validation set
                                  (YOLO fitness = 0.1 * mAP50 + 0.9 * mAP50-95)
    checkpoints/weights/last.pt : model at the last epoch
    checkpoints/                : training curves, metrics and validation plots from YOLO

Inspired from: the train.py script of the previous version of sc-crop

Author: Pierre-Louis Benveniste
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


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
    return parser


def get_augmentations():
    """Return the YOLO built-in augmentations used for training (arguments of model.train).
    No hue/saturation (grayscale or pseudo-RGB slices), no mosaic."""
    return dict(
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.15,   # brightness only
        degrees=15.0, scale=0.2, translate=0.1, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.5,
        mosaic=0.0, copy_paste=0.0,
    )


def main():
    args = get_parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"\nBest weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
