"""
Train a YOLO model to detect the spinal cord (SC, class 0) and the spinal canal (class 1) on axial slices.

Input: the YOLO dataset built by data/build_yolo_dataset.py (its dataset.yaml), with
images/{training,validation,test} and one label file per slice (one "cls cx cy w h" line per class
present on the slice, empty if neither).

At the end of the training, the best model is evaluated on the training and validation sets, with the
metrics of each class (see evaluate_detection.py).

Outputs (in RUN_DIR):
    checkpoints/weights/best.pt : best model on the validation set (highest mAP50-95)
    checkpoints/weights/last.pt : model at the last epoch
    checkpoints/                : training curves, metrics and validation plots from YOLO
    evaluation/training_curves.png : losses and validation metrics of each epoch
    evaluation/{training,validation}/<class>/ : metrics, plots and worst slices of the best model, per class
    training.log                : messages of YOLO and of this script

Inspired from: the train.py script of the previous version of sc-crop

Author: Pierre-Louis Benveniste
"""
import argparse
import logging
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from evaluate_detection import evaluate, plot_training_curves


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
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence of a predicted box for the final evaluation.")
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

    # Save the YOLO messages (and the ones of this script) in a log file of the run
    file_handler = logging.FileHandler(args.run_dir / "training.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    LOGGER.addHandler(file_handler)
    LOGGER.info(f"Arguments: {vars(args)}")

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

    # Evaluate the best model on the training and validation sets
    eval_dir = args.run_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    plot_training_curves(save_dir / "results.csv", eval_dir / "training_curves.png")
    best_model = YOLO(best_weights)
    for split in ("training", "validation"):
        evaluate(best_model, args.dataset, split, eval_dir, args.conf, args.imgsz, args.device)


if __name__ == "__main__":
    main()
