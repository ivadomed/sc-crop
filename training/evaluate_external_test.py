"""
Evaluate all the models trained by run_experiments.sh on the external test set, e.g. after rebuilding the
YOLO datasets with fixed external test segmentations.

For each training folder <date>_<configuration>_<model>/ of EXPERIMENTS_DIR:
    - model: checkpoints/weights/best.pt
    - thresholds: evaluation/thresholds.json (chosen on the validation set at the end of the training)
    - dataset: the RGB dataset for the rgb configurations, else the grayscale one
    - evaluate_detection.py on the external_test split (metrics of each class, overall and per dataset)
Models are evaluated one after the other.

Outputs (in OUTPUT_DIR):
    <configuration>_<model>/external_test/ : results of evaluate_detection.py (metrics, csv, plots, worst slices)
    external_test_summary.csv              : main metrics of each model, class and dataset (overall = all datasets)

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from evaluate_detection import evaluate, read_thresholds

# Metrics of the summary (see evaluate_detection.py)
METRICS = ("n_volumes", "sensitivity", "specificity", "precision", "median_iou", "median_center_error_mm",
           "median_iou_3d", "median_gt_coverage_3d", "gt_coverage_3d_above_0.95", "median_si_coverage")


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate the models of run_experiments.sh on the external test set.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiments-dir", required=True, type=Path, help="Output folder of run_experiments.sh.")
    parser.add_argument("--gray-dataset", required=True, type=Path, help="dataset.yaml of the grayscale YOLO dataset.")
    parser.add_argument("--rgb-dataset", required=True, type=Path, help="dataset.yaml of the RGB YOLO dataset.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the results are saved.")
    parser.add_argument("--device", default="0", help="GPU id or cpu.")
    parser.add_argument("--n-worst", type=int, default=100, help="Number of worst slices and worst volumes to save as png.")
    return parser


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for weights in sorted(args.experiments_dir.glob("*/checkpoints/weights/best.pt")):
        run_dir = weights.parents[2]
        name = run_dir.name.split("_", 2)[-1]   # <configuration>_<model>, without the date
        thresholds_path = run_dir / "evaluation" / "thresholds.json"
        if not thresholds_path.exists():
            print(f"{name}: no {thresholds_path}, skipped")
            continue
        dataset = args.rgb_dataset if name.startswith("rgb") else args.gray_dataset
        print(f"{name}: {weights} on {dataset}")

        model = YOLO(weights)
        imgsz = model.ckpt.get("train_args", {}).get("imgsz", 320)
        thresholds = read_thresholds(thresholds_path, dataset)
        metrics = evaluate(model, dataset, "external_test", args.output_dir / name, thresholds, imgsz, args.device,
                           args.n_worst)

        for cls_name, m in metrics.items():
            for dataset_name, values in {"overall": m["overall"], **m["per_dataset"]}.items():
                rows.append({"model": name, "class": cls_name, "threshold": m["threshold"], "dataset": dataset_name,
                             **{k: values[k] for k in METRICS}})

    if not rows:
        raise SystemExit(f"No model evaluated (no */checkpoints/weights/best.pt with thresholds in {args.experiments_dir})")
    summary = args.output_dir / "external_test_summary.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()} for row in rows)
    print(f"{len({r['model'] for r in rows})} models evaluated. Summary: {summary}")


if __name__ == "__main__":
    main()
