"""
Evaluate a previous YOLO model detecting only the spinal cord (SC, one class) on the YOLO dataset built by
data/build_yolo_dataset.py, with the same metrics as evaluate_detection.py, to compare it with the SC and canal models.

The boxes of the model (its only class) are compared with the SC boxes of the dataset (class 0); the canal is not
evaluated. Like for the SC and canal models, the confidence threshold is chosen on the validation set (best
slice-level F1), unless it is set with --conf.

With --compare-with (evaluation folder of an SC and canal model, e.g. <run_dir>/evaluation from train_detection.py),
the main SC metrics of both models are saved side by side, overall and per dataset.

Outputs (in OUTPUT_DIR):
    thresholds.json, thresholds.png : SC threshold chosen on the validation set (without --conf)
    <split>/spinal_cord/            : metrics, csv, plots, worst slices and worst volumes (see evaluate_detection.py)
    comparison_<split>.csv          : with --compare-with: main SC metrics of both models, overall and per dataset

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
from pathlib import Path

from evaluate_detection import evaluate, read_dataset, select_thresholds

SC_CLASS = 0   # class of the SC in the dataset
# Metrics of the comparison (see evaluate_detection.py)
METRICS = ("n_volumes", "sensitivity", "specificity", "precision", "median_iou", "median_center_error_mm",
           "median_iou_3d", "median_gt_coverage_3d", "gt_coverage_3d_above_0.95", "median_si_coverage")


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model detecting only the SC on the YOLO dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, type=Path, help="Weights of the SC-only model (.pt).")
    parser.add_argument("--dataset", required=True, type=Path, help="dataset.yaml built by build_yolo_dataset.py.")
    parser.add_argument("--splits", nargs="+", default=["test", "external_test"],
                        choices=["training", "validation", "test", "external_test"], help="Splits to evaluate.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the results are saved.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold (default: chosen on the validation set).")
    parser.add_argument("--compare-with", type=Path, default=None,
                        help="Evaluation folder of an SC and canal model (<run_dir>/evaluation), to compare with.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (default: the one used for training).")
    parser.add_argument("--device", default="0", help="GPU id or cpu.")
    parser.add_argument("--n-worst", type=int, default=100, help="Number of worst slices and worst volumes to save as png.")
    return parser


def save_comparison(metrics, other_metrics_path, path):
    """Save the main SC metrics of the SC-only model (`metrics`) and of the SC and canal model (its metrics.json),
    overall and per dataset, one row per dataset and model."""
    other = json.loads(other_metrics_path.read_text())
    rows = []
    for dataset in ["overall", *sorted(metrics["per_dataset"])]:
        for model, m in (("sc_only", metrics), ("sc_and_canal", other)):
            values = m["overall"] if dataset == "overall" else m["per_dataset"].get(dataset)
            if values is None:
                continue
            rows.append({"dataset": dataset, "model": model, "threshold": m["threshold"],
                         **{k: round(values[k], 4) if isinstance(values[k], float) else values[k] for k in METRICS}})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    args = get_parser().parse_args()
    from ultralytics import YOLO   # imported here: the functions above do not need it

    model = YOLO(args.model)
    if len(model.names) != 1:
        raise SystemExit(f"{args.model} detects {len(model.names)} classes ({model.names}): expected only the SC")
    imgsz = args.imgsz or model.ckpt.get("train_args", {}).get("imgsz", 320)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sc_name = read_dataset(args.dataset, "validation")[2][SC_CLASS]

    # The only class of the model (0) is compared with the SC of the dataset (class 0)
    if args.conf is not None:
        thresholds = {SC_CLASS: args.conf}
    else:
        thresholds = select_thresholds(model, args.dataset, args.output_dir, imgsz, args.device, class_ids=[SC_CLASS])

    for split in args.splits:
        images_dir = read_dataset(args.dataset, split)[0]
        if not any(images_dir.glob("*.png")):
            print(f"{split}: no slices in {images_dir}, skipped")
            continue
        metrics = evaluate(model, args.dataset, split, args.output_dir, thresholds, imgsz, args.device,
                           args.n_worst)[sc_name]
        if args.compare_with is not None:
            other = args.compare_with / split / sc_name / "metrics.json"
            if not other.exists():
                print(f"{split}: no {other}, no comparison")
                continue
            rows = save_comparison(metrics, other, args.output_dir / f"comparison_{split}.csv")
            print(f"{split}, {sc_name} (overall):")
            for row in rows[:2]:
                print(f"  {row['model']:<13} " + ", ".join(f"{k} {row[k]}" for k in METRICS))


if __name__ == "__main__":
    main()
