"""
Evaluate a YOLO spinal cord (SC) detection model on one split of the YOLO dataset built by
data/build_yolo_dataset.py, with metrics that reflect what sc-crop needs: finding the SC on each slice
and a 3D box around the whole SC.

Each slice has at most one SC box. The predicted box of a slice is its most confident box, if its
confidence is >= --conf. Slices are at 1 x 1 mm in-plane, so 1 pixel = 1 mm.

Metrics:
    Slice level (is there SC on this slice?):
        sensitivity : SC slices with a predicted box / SC slices
        specificity : empty slices without predicted box / empty slices
        precision   : slices with a predicted box that contain SC / slices with a predicted box
    Box level, on SC slices (a missed SC slice counts as IoU = 0):
        IoU          : intersection over union of the predicted and ground truth (GT) boxes
        IoU >= 0.5   : % of SC slices with a correctly placed box
        center error : distance between the predicted and GT box centers (mm), on detected SC slices
        GT coverage  : fraction of the GT box inside the predicted box, on detected SC slices
                       (= fraction of the SC a crop around the predicted box would keep)
    Volume level (the 3D box used to crop): union of the slice boxes, from the first to the last slice
    with a box. On volumes with SC:
        3D IoU, 3D GT coverage, and SI coverage (fraction of the GT SC slices inside the predicted range)

Outputs (in OUTPUT_DIR/<split>/):
    metrics.json  : all metrics, overall and per dataset
    slices.csv    : results of each slice
    volumes.csv   : results of each volume
    summary.png   : plots of the metrics
    worst_slices/ : the N worst slices (lowest IoU): image with the GT (green) and predicted (red) boxes
    worst_volumes/: the N worst volumes (lowest 3D IoU): sagittal view at the center of the GT 3D box,
                    with the GT (green) and predicted (red) 3D boxes

Usage:
    python evaluate_detection.py --model <run_dir>/checkpoints/weights/best.pt \
        --dataset <yolo_dataset>/dataset.yaml --split test --output-dir <run_dir>/evaluation

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed (servers)
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

# Logger of YOLO: during training, its messages are also saved in the training log file
LOGGER = logging.getLogger("ultralytics")
OUTCOMES = ("TP", "FN", "FP", "TN")   # SC detected, SC missed, box on an empty slice, empty slice left empty


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO SC detection model on one split of the dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Model weights (e.g. checkpoints/weights/best.pt).")
    parser.add_argument("--dataset", required=True, type=Path, help="dataset.yaml built by build_yolo_dataset.py.")
    parser.add_argument("--split", default="test", choices=["training", "validation", "test"])
    parser.add_argument("--output-dir", required=True, type=Path, help="Results are saved in <output-dir>/<split>/.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence of a predicted box.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default: the one used for training).")
    parser.add_argument("--device", default="0", help="GPU id or cpu.")
    parser.add_argument("--n-worst", type=int, default=100, help="Number of worst slices and worst volumes to save as png.")
    return parser


# ── Boxes ─────────────────────────────────────────────────────────────────────

def read_gt_box(label_path, width, height):
    """Return the GT box (x1, y1, x2, y2) in pixels from a YOLO label file, or None if the slice has no SC."""
    values = label_path.read_text().split() if label_path.exists() else []
    if not values:
        return None
    cx, cy, w, h = (float(v) for v in values[1:5])
    return ((cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


# ── Predictions and metrics ───────────────────────────────────────────────────

def predict(model, images_dir, conf, imgsz, device):
    """Return {image name: ((x1, y1, x2, y2), confidence)} with the most confident box of each image of
    `images_dir`, or None for images without box."""
    preds = {}
    # The source must be the folder, not a list of paths: YOLO loads a list in memory as one single batch
    results = model.predict(source=str(images_dir), stream=True, batch=32,
                            conf=conf, imgsz=imgsz, device=device, verbose=False)
    for r in results:
        if len(r.boxes):
            i = int(r.boxes.conf.argmax())
            preds[Path(r.path).stem] = (tuple(float(v) for v in r.boxes.xyxy[i]), float(r.boxes.conf[i]))
        else:
            preds[Path(r.path).stem] = None
    return preds


def evaluate_slices(images_dir, labels_dir, preds):
    """Return one record per slice with its GT and predicted boxes, outcome and metrics."""
    records = []
    for image in sorted(images_dir.glob("*.png")):
        width, height = Image.open(image).size
        gt = read_gt_box(labels_dir / f"{image.stem}.txt", width, height)
        pred, conf = preds.get(image.stem) or (None, None)
        # Slice name: <dataset>__<image name>__slice_NNN
        volume, slice_index = image.stem.rsplit("__slice_", 1)
        record = {"name": image.stem, "image": str(image), "volume": volume, "dataset": volume.split("__")[0],
                  "contrast": volume.split("_")[-1], "slice": int(slice_index), "gt_box": gt, "pred_box": pred,
                  "conf": conf, "iou": None, "center_error_mm": None, "gt_coverage": None}
        if gt and pred:
            inter = intersection(gt, pred)
            record.update(outcome="TP", iou=inter / (area(gt) + area(pred) - inter), gt_coverage=inter / area(gt),
                          center_error_mm=float(np.hypot((gt[0] + gt[2] - pred[0] - pred[2]) / 2,
                                                         (gt[1] + gt[3] - pred[1] - pred[3]) / 2)))
        elif gt:
            record.update(outcome="FN", iou=0.0)
        elif pred:
            record.update(outcome="FP", iou=0.0)
        else:
            record.update(outcome="TN")
        records.append(record)
    return records


def box_3d(slice_boxes):
    """3D box (x1, y1, z1, x2, y2, z2) around [(slice index, 2D box)], or None. z in slices, z2 excluded."""
    if not slice_boxes:
        return None
    zs, boxes = zip(*slice_boxes)
    boxes = np.array(boxes)
    return (*boxes[:, :2].min(axis=0), min(zs), *boxes[:, 2:].max(axis=0), max(zs) + 1)


def volume_3d(box):
    return float(np.prod([max(0.0, box[i + 3] - box[i]) for i in range(3)]))


def evaluate_volumes(records):
    """Return one record per volume with SC, comparing the predicted and GT 3D boxes."""
    by_volume = defaultdict(list)
    for r in records:
        by_volume[r["volume"]].append(r)

    volumes = []
    for volume, slices in sorted(by_volume.items()):
        gt = box_3d([(r["slice"], r["gt_box"]) for r in slices if r["gt_box"]])
        if gt is None:
            continue
        pred = box_3d([(r["slice"], r["pred_box"]) for r in slices if r["pred_box"]])
        record = {"volume": volume, "dataset": slices[0]["dataset"], "contrast": slices[0]["contrast"],
                  "n_sc_slices": sum(r["gt_box"] is not None for r in slices),
                  "iou_3d": 0.0, "gt_coverage_3d": 0.0, "si_coverage": 0.0,
                  "gt_box_3d": gt, "pred_box_3d": pred,
                  "images": [r["image"] for r in sorted(slices, key=lambda r: r["slice"])]}
        if pred:
            inter = volume_3d((*np.maximum(gt[:3], pred[:3]), *np.minimum(gt[3:], pred[3:])))
            record.update(iou_3d=inter / (volume_3d(gt) + volume_3d(pred) - inter),
                          gt_coverage_3d=inter / volume_3d(gt),
                          si_coverage=max(0, min(gt[5], pred[5]) - max(gt[2], pred[2])) / (gt[5] - gt[2]))
        volumes.append(record)
    return volumes


def compute_metrics(records, volumes):
    """Return the slice, box and volume metrics of a set of slices and volumes."""
    counts = {o: sum(r["outcome"] == o for r in records) for o in OUTCOMES}
    tp, fn, fp, tn = (counts[o] for o in OUTCOMES)
    ious = [r["iou"] for r in records if r["gt_box"]]
    detected = [r for r in records if r["outcome"] == "TP"]

    def ratio(a, b):
        return a / b if b else None

    def median(values):
        return float(np.median(values)) if values else None

    return {
        "n_slices": len(records), "n_sc_slices": tp + fn, "n_empty_slices": fp + tn, **counts,
        "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp), "precision": ratio(tp, tp + fp),
        "mean_iou": float(np.mean(ious)) if ious else None, "median_iou": median(ious),
        "iou_above_0.5": ratio(sum(i >= 0.5 for i in ious), len(ious)),
        "iou_above_0.75": ratio(sum(i >= 0.75 for i in ious), len(ious)),
        "median_center_error_mm": median([r["center_error_mm"] for r in detected]),
        "median_gt_coverage": median([r["gt_coverage"] for r in detected]),
        "n_volumes": len(volumes),
        "median_iou_3d": median([v["iou_3d"] for v in volumes]),
        "median_gt_coverage_3d": median([v["gt_coverage_3d"] for v in volumes]),
        "gt_coverage_3d_above_0.95": ratio(sum(v["gt_coverage_3d"] >= 0.95 for v in volumes), len(volumes)),
        "median_si_coverage": median([v["si_coverage"] for v in volumes]),
    }


# ── Outputs ───────────────────────────────────────────────────────────────────

def save_csv(rows, columns, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (np.round(v, 4).tolist() if isinstance(v, (float, tuple)) else v)
                             for k, v in row.items()})


def plot_summary(records, volumes, metrics, split, path):
    """Plot the slice, box and volume metrics of one split in one figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{split}: {metrics['n_slices']} slices, {metrics['n_volumes']} volumes with SC", fontsize=14)

    def fmt(value):
        return "n/a" if value is None else f"{value:.3f}"

    # Slice outcomes
    ax = axes[0, 0]
    colors = {"TP": "tab:green", "FN": "tab:red", "FP": "tab:orange", "TN": "tab:gray"}
    labels = {"TP": "SC detected", "FN": "SC missed", "FP": "box on\nempty slice", "TN": "empty slice\nleft empty"}
    bars = ax.bar([labels[o] for o in OUTCOMES], [metrics[o] for o in OUTCOMES], color=[colors[o] for o in OUTCOMES])
    ax.bar_label(bars)
    ax.set_title(f"Slices: sensitivity {fmt(metrics['sensitivity'])}, specificity {fmt(metrics['specificity'])}")
    ax.set_ylabel("Number of slices")

    # IoU distribution on SC slices
    ax = axes[0, 1]
    ious = [r["iou"] for r in records if r["gt_box"]]
    ax.hist(ious, bins=np.linspace(0, 1, 21), color="tab:blue")
    for threshold in (0.5, 0.75):
        ax.axvline(threshold, color="k", linestyle="--", linewidth=1)
    ax.set_title(f"IoU on SC slices (missed = 0): median {fmt(metrics['median_iou'])}, "
                 f"{fmt(metrics['iou_above_0.5'])} >= 0.5")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Number of slices")

    # IoU per dataset
    ax = axes[0, 2]
    per_dataset = defaultdict(list)
    for r in records:
        if r["gt_box"]:
            per_dataset[r["dataset"]].append(r["iou"])
    names = sorted(per_dataset, key=lambda d: np.median(per_dataset[d]))
    if names:
        ax.boxplot([per_dataset[d] for d in names], vert=False, showfliers=False)
        ax.set_yticks(range(1, len(names) + 1), [f"{d} ({len(per_dataset[d])})" for d in names], fontsize=8)
    ax.set_title("IoU on SC slices per dataset (number of slices)")
    ax.set_xlabel("IoU")
    ax.set_xlim(0, 1)

    # Center error on detected SC slices
    ax = axes[1, 0]
    errors = [r["center_error_mm"] for r in records if r["outcome"] == "TP"]
    ax.hist(errors, bins=np.linspace(0, max(errors + [10]), 31), color="tab:purple")
    ax.set_title(f"Distance between box centers (detected SC): median {fmt(metrics['median_center_error_mm'])} mm")
    ax.set_xlabel("Center error (mm)")
    ax.set_ylabel("Number of slices")

    # GT coverage on detected SC slices
    ax = axes[1, 1]
    ax.hist([r["gt_coverage"] for r in records if r["outcome"] == "TP"], bins=np.linspace(0, 1, 21),
            color="tab:cyan")
    ax.set_title(f"Fraction of the GT box inside the predicted box: median {fmt(metrics['median_gt_coverage'])}")
    ax.set_xlabel("GT coverage")
    ax.set_ylabel("Number of slices")

    # Volume level
    ax = axes[1, 2]
    bins = np.linspace(0, 1, 21)
    ax.hist([v["gt_coverage_3d"] for v in volumes], bins=bins, alpha=0.6, label="GT coverage (3D)")
    ax.hist([v["iou_3d"] for v in volumes], bins=bins, alpha=0.6, label="IoU (3D)")
    ax.set_title(f"Volumes: median 3D IoU {fmt(metrics['median_iou_3d'])}, "
                 f"{fmt(metrics['gt_coverage_3d_above_0.95'])} with GT coverage >= 0.95")
    ax.set_xlabel("Value")
    ax.set_ylabel("Number of volumes")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_slice(record, rank, path):
    """Save the slice with its GT box (green) and predicted box (red)."""
    image = np.array(Image.open(record["image"]))
    fig, ax = plt.subplots(figsize=(6, 6))
    # extent: pixel i covers [i, i + 1], like the box coordinates
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None, extent=(0, image.shape[1], image.shape[0], 0))
    for box, color, label in ((record["gt_box"], "lime", "GT"), (record["pred_box"], "red", "prediction")):
        if box:
            ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, color=color,
                                   linewidth=1.5, label=label))
    if record["gt_box"] or record["pred_box"]:
        ax.legend(loc="lower right", fontsize=8)
    conf = "no box" if record["conf"] is None else f"conf {record['conf']:.2f}"
    ax.set_title(f"#{rank} {record['outcome']}  IoU {record['iou']:.2f}  {conf}\n{record['name']}", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_volume(volume, rank, path):
    """Save the sagittal view of the volume at the center (RL) of the GT 3D box, with the GT 3D box (green)
    and predicted 3D box (red). Rows = slices (superior at the top, 10 mm), columns = AP (1 mm)."""
    # Axial slices stacked from superior to inferior; pseudo-RGB pngs: G = current slice
    axial = [np.array(Image.open(image)) for image in volume["images"]]
    axial = np.stack([a[:, :, 1] if a.ndim == 3 else a for a in axial])   # (slice, AP, RL)
    gt, pred = volume["gt_box_3d"], volume["pred_box_3d"]
    x = min(int((gt[0] + gt[3]) / 2), axial.shape[2] - 1)
    sagittal = axial[:, :, x]   # (slice, AP)

    fig, ax = plt.subplots(figsize=(6, 8))
    # extent: slice z covers [z, z + 1], AP pixel i covers [i, i + 1], like the box coordinates
    ax.imshow(sagittal, cmap="gray", extent=(0, sagittal.shape[1], sagittal.shape[0], 0), aspect=10)
    for box, color, label in ((gt, "lime", "GT"), (pred, "red", "prediction")):
        if box:   # (x1, y1, z1, x2, y2, z2): y = AP, z = slice
            ax.add_patch(Rectangle((box[1], box[2]), box[4] - box[1], box[5] - box[2], fill=False, color=color,
                                   linewidth=1.5, label=label))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"#{rank}  3D IoU {volume['iou_3d']:.2f}  GT coverage {volume['gt_coverage_3d']:.2f}  "
                 f"SI coverage {volume['si_coverage']:.2f}\n{volume['volume']} (RL = {x} mm)", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_results(records, split, output_dir, n_worst):
    """Compute the metrics of one split and save metrics, csv files, plots and worst slices. Return the metrics."""
    out = output_dir / split
    (out / "worst_slices").mkdir(parents=True, exist_ok=True)
    volumes = evaluate_volumes(records)

    metrics = {"overall": compute_metrics(records, volumes), "per_dataset": {}}
    for dataset in sorted({r["dataset"] for r in records}):
        metrics["per_dataset"][dataset] = compute_metrics([r for r in records if r["dataset"] == dataset],
                                                          [v for v in volumes if v["dataset"] == dataset])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=4))

    save_csv(records, ["name", "dataset", "contrast", "volume", "slice", "outcome", "conf", "iou",
                       "center_error_mm", "gt_coverage", "gt_box", "pred_box"], out / "slices.csv")
    save_csv(volumes, ["volume", "dataset", "contrast", "n_sc_slices", "iou_3d", "gt_coverage_3d", "si_coverage"],
             out / "volumes.csv")
    plot_summary(records, volumes, metrics["overall"], split, out / "summary.png")

    # Worst slices: lowest IoU first (missed SC and boxes on empty slices have IoU 0),
    # then the most confident wrong boxes first
    candidates = sorted((r for r in records if r["outcome"] != "TN"), key=lambda r: (r["iou"], -(r["conf"] or 0)))
    for rank, record in enumerate(candidates[:n_worst], 1):
        plot_slice(record, rank, out / "worst_slices" / f"{rank:03d}_{record['name']}.png")

    # Worst volumes: lowest 3D IoU first, then the lowest GT coverage
    (out / "worst_volumes").mkdir(exist_ok=True)
    worst_volumes = sorted(volumes, key=lambda v: (v["iou_3d"], v["gt_coverage_3d"]))
    for rank, volume in enumerate(worst_volumes[:n_worst], 1):
        plot_volume(volume, rank, out / "worst_volumes" / f"{rank:03d}_{volume['volume']}.png")
    return metrics


def evaluate(model, dataset_yaml, split, output_dir, conf=0.25, imgsz=320, device="0", n_worst=100):
    """Predict and evaluate one split of the YOLO dataset. Return the metrics."""
    root = Path(yaml.safe_load(dataset_yaml.read_text())["path"])
    images_dir, labels_dir = root / "images" / split, root / "labels" / split
    images = sorted(images_dir.glob("*.png"))
    LOGGER.info(f"Evaluating {split}: {len(images)} slices")
    preds = predict(model, images_dir, conf, imgsz, device)
    metrics = save_results(evaluate_slices(images_dir, labels_dir, preds), split, output_dir, n_worst)

    m = metrics["overall"]
    LOGGER.info(f"{split}: sensitivity {m['sensitivity']}, specificity {m['specificity']}, median IoU {m['median_iou']}, "
          f"median 3D IoU {m['median_iou_3d']} (see {output_dir / split})")
    return metrics


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(results_csv, path):
    """Plot the losses (training vs validation) and validation metrics of each epoch from YOLO's results.csv."""
    with open(results_csv) as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    epochs = [r["epoch"] for r in rows]
    losses = [k.split("/", 1)[1] for k in rows[0] if k.startswith("train/") and f"val/{k.split('/', 1)[1]}" in rows[0]]
    metrics = [k for k in rows[0] if k.startswith("metrics/")]
    # best.pt is the epoch with the highest mAP50-95 (YOLO fitness)
    best = max(rows, key=lambda r: r.get("metrics/mAP50-95(B)", 0))["epoch"]

    # First row: losses (training vs validation). Second row: one plot per validation metric.
    n_cols = max(len(losses), len(metrics))
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9), squeeze=False)
    for ax, loss in zip(axes[0], losses):
        ax.plot(epochs, [r[f"train/{loss}"] for r in rows], label="training")
        ax.plot(epochs, [r[f"val/{loss}"] for r in rows], label="validation")
        ax.set_title(f"Loss: {loss}")
    for ax, metric in zip(axes[1], metrics):
        values = [r[metric] for r in rows]
        name = metric.removeprefix("metrics/").removesuffix("(B)")
        ax.plot(epochs, values, color="tab:green", label=name)
        ax.set_title(f"Validation {name}: best epoch {next(r[metric] for r in rows if r['epoch'] == best):.3f}, "
                     f"max {max(values):.3f}")
    for ax in axes.flat:
        if not ax.lines:   # unused plot
            ax.axis("off")
            continue
        ax.axvline(best, color="k", linestyle="--", linewidth=1, label=f"best epoch ({int(best)})")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    args = get_parser().parse_args()
    from ultralytics import YOLO   # imported here: the functions above do not need it

    model = YOLO(args.model)
    imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 320)
    evaluate(model, args.dataset, args.split, args.output_dir, args.conf, imgsz, args.device, args.n_worst)


if __name__ == "__main__":
    main()
