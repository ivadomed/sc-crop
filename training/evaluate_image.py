"""
Run the detection model on one image and compare its boxes to the ground truth (GT) boxes of the SC and canal
segmentations: metrics (same as evaluate_detection.py) and plots of each axial slice and of the sagittal view.

    1. Preprocess the image and predict the boxes of all its axial slices (functions of ../inference/)
    2. GT boxes: the masks (non-zero voxels) are processed like in data/build_yolo_dataset.py (largest connected
       component, LAS, 1 x 1 x 10 mm), then the box of each slice is taken around the mask
    3. Metrics of each class: slice, box and volume level (see evaluate_detection.py)

The masks must have the same shape and affine as the image (see data/fix_seg_affine.py).

Outputs (in OUTPUT_DIR):
    metrics.json        : metrics of each class (with its threshold)
    slices.csv          : outcome, confidence, IoU, GT and predicted boxes of each slice and class
    axial/slice_NNN.png : each axial slice with the GT (solid) and predicted (dashed) boxes, SC green, canal red
    overview.png        : all axial slices in one figure
    sagittal.png        : sagittal views, one row per class (SC, canal): GT boxes (left) and predicted boxes
                          (right), as their AP extent on each slice, at the center (RL) of the GT boxes

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed (servers)
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from ultralytics import YOLO

# Inference scripts (preprocessing, prediction) and dataset building (processing of the masks)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "training" / "data"))
from build_yolo_dataset import prepare_mask
from evaluate_detection import compare_boxes, compute_metrics, evaluate_volumes
from predict_image import get_thresholds, predict_slices
from preprocess_image import cut_slices, resample_image

# Color and short name of each class (0: SC, 1: canal)
COLORS = {0: "lime", 1: "red"}
SHORT_NAMES = {0: "SC", 1: "canal"}


def get_parser():
    parser = argparse.ArgumentParser(description="Predict the boxes of one image and compare them to its GT boxes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Image (nii.gz).")
    parser.add_argument("--model", required=True, type=Path, help="Model weights (e.g. checkpoints/weights/best.pt).")
    parser.add_argument("--canal-seg", required=True, type=Path, help="Canal segmentation (nii.gz).")
    parser.add_argument("--sc-seg", type=Path, default=None, help="SC segmentation (nii.gz). None: canal only.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the results are saved.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (for the models trained on RGB slices).")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25],
                        help="Confidence threshold: one value for all classes, or one per class (SC canal).")
    return parser


def read_gt_boxes(mask_path, image, warnings):
    """Return the GT box (x1, y1, x2, y2) of each axial slice (None if the mask is empty on the slice), with the
    mask processed like in build_yolo_dataset.py. Masks with several components are added to `warnings`."""
    mask = nib.load(mask_path)
    if mask.shape[:3] != image.shape[:3] or not np.allclose(mask.affine, image.affine, atol=1e-3):
        raise SystemExit(f"{mask_path} does not have the shape and affine of the image (see data/fix_seg_affine.py)")
    data = prepare_mask(np.asarray(mask.dataobj) > 0, mask.affine, mask_path, warnings)
    boxes = []
    for s in data.transpose(2, 1, 0)[::-1, ::-1, :]:   # axial slices, cut like the image slices (cut_slices)
        rows, cols = np.nonzero(s)
        boxes.append((float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
                     if rows.size else None)
    return boxes


def draw_box(ax, x, y, width, height, cls, predicted):
    ax.add_patch(Rectangle((x, y), width, height, fill=False, color=COLORS[cls], linewidth=1.5,
                           linestyle="--" if predicted else "-"))


def legend(ax, names):
    ax.legend(handles=[Line2D([], [], color=COLORS[cls], linestyle=ls, label=f"{names[cls]} {source}")
                       for cls in names for ls, source in (("-", "GT"), ("--", "prediction"))],
              loc="lower right", fontsize=7)


def plot_axial(ax, image, records, i):
    """Plot axial slice `i` with the GT (solid) and predicted (dashed) boxes of each class."""
    # extent: pixel i covers [i, i + 1], like the box coordinates
    ax.imshow(image, cmap="gray", extent=(0, image.shape[1], image.shape[0], 0))
    ious = []
    for cls, class_records in records.items():
        r = class_records[i]
        for box, predicted in ((r["gt_box"], False), (r["pred_box"], True)):
            if box:
                draw_box(ax, box[0], box[1], box[2] - box[0], box[3] - box[1], cls, predicted)
        if r["outcome"] != "TN":
            ious.append(f"{SHORT_NAMES[cls]} {r['outcome']} IoU {r['iou']:.2f}")
    ax.set_title(f"slice {i:03d}  " + "  ".join(ious), fontsize=8)
    ax.axis("off")


def plot_sagittal(images, records, names, metrics, path):
    """Save the sagittal views: one row per class (SC, then canal), with the GT boxes (left) and the predicted
    boxes (right) drawn as their AP extent on each slice. Both views of a class are at the center (RL) of its GT
    boxes (else of its predicted boxes)."""
    axial = np.stack(images)   # (slice, AP, RL), from superior to inferior
    fig, axes = plt.subplots(len(records), 2, figsize=(10, 8 * len(records)), squeeze=False)
    for row, (cls, class_records) in zip(axes, records.items()):
        centers = ([(r["gt_box"][0] + r["gt_box"][2]) / 2 for r in class_records if r["gt_box"]]
                   or [(r["pred_box"][0] + r["pred_box"][2]) / 2 for r in class_records if r["pred_box"]]
                   or [axial.shape[2] / 2])
        x = min(int(np.median(centers)), axial.shape[2] - 1)
        m = metrics[names[cls]]
        iou_3d = "n/a" if m["median_iou_3d"] is None else f"{m['median_iou_3d']:.2f}"
        for ax, key, source in ((row[0], "gt_box", "GT"), (row[1], "pred_box", "prediction")):
            # extent: slice z covers [z, z + 1], AP pixel i covers [i, i + 1], like the box coordinates
            ax.imshow(axial[:, :, x], cmap="gray", extent=(0, axial.shape[1], axial.shape[0], 0), aspect=10)
            for z, r in enumerate(class_records):
                if r[key]:
                    draw_box(ax, r[key][1], z, r[key][3] - r[key][1], 1, cls, False)
            ax.set_title(f"{names[cls]} {source} (3D IoU {iou_3d})\nsagittal at RL = {x} mm", fontsize=10)
            ax.axis("off")
    fig.suptitle("Superior at the top, anterior on the left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    args = get_parser().parse_args()
    (args.output_dir / "axial").mkdir(parents=True, exist_ok=True)

    # 1. Prediction
    slices = cut_slices(np.asarray(resample_image(args.image).dataobj), args.rgb)
    model = YOLO(args.model)
    names = model.names
    thresholds = get_thresholds(args.conf, names)
    preds = predict_slices(model, slices, thresholds)

    # 2. GT boxes of each class with a segmentation
    image = nib.load(args.image)
    masks = {cls: path for cls, path in ((0, args.sc_seg), (1, args.canal_seg)) if path is not None}
    warnings = []
    records = {}
    for cls, mask_path in masks.items():
        gt_boxes = read_gt_boxes(mask_path, image, warnings)
        records[cls] = []
        for i, (gt, pred) in enumerate(zip(gt_boxes, preds)):
            box, conf = pred.get(cls, (None, None))
            records[cls].append({"volume": args.image.name, "dataset": "image", "contrast": "", "image": None,
                                 "slice": i, "gt_box": gt, "pred_box": box, "conf": conf, **compare_boxes(gt, box)})

    # 3. Metrics
    metrics = {names[cls]: {"threshold": thresholds[cls], **compute_metrics(r, evaluate_volumes(r))}
               for cls, r in records.items()}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))
    with open(args.output_dir / "slices.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "class", "outcome", "conf", "iou", "gt_box", "pred_box"])
        for cls, class_records in records.items():
            for r in class_records:
                writer.writerow([r["slice"], names[cls], r["outcome"],
                                 *(None if v is None else round(v, 4) for v in (r["conf"], r["iou"])),
                                 *(None if b is None else [round(float(v), 1) for v in b] for b in (r["gt_box"], r["pred_box"]))])

    # Plots: G channel = current slice (pseudo-RGB slices are BGR, like the dataset pngs)
    images = [s[:, :, 1] for s in slices]
    for i, im in enumerate(images):
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_axial(ax, im, records, i)
        legend(ax, {cls: names[cls] for cls in records})
        fig.tight_layout()
        fig.savefig(args.output_dir / "axial" / f"slice_{i:03d}.png", dpi=100)
        plt.close(fig)

    n_cols = min(8, len(images))
    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, im) in enumerate(zip(axes.flat, images)):
        plot_axial(ax, im, records, i)
    fig.suptitle(f"{args.image.name}: {len(images)} slices (superior to inferior), GT solid, prediction dashed, "
                 "SC green, canal red")
    fig.tight_layout()
    fig.savefig(args.output_dir / "overview.png", dpi=100)
    plt.close(fig)

    plot_sagittal(images, records, names, metrics, args.output_dir / "sagittal.png")

    for name, m in metrics.items():
        print(f"{name} (threshold {m['threshold']}): sensitivity {m['sensitivity']}, specificity {m['specificity']}, "
              f"median IoU {m['median_iou']}, 3D IoU {m['median_iou_3d']}, 3D GT coverage {m['median_gt_coverage_3d']}")
    print(f"Warnings: {warnings or 'none'}\nSaved in {args.output_dir} (metrics.json, slices.csv, overview.png, "
          f"sagittal.png, axial/)")


if __name__ == "__main__":
    main()
