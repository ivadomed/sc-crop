"""
Compare the metrics of trained models with and without some datasets (by default spider-challenge-2023), to see
how much these datasets lower the overall performance.

The metrics are recomputed from the results of each slice (slices.csv) saved by evaluate_detection.py (e.g. at the
end of train_detection.py in <run_dir>/evaluation/, or by evaluate_sc_only_model.py): no new prediction, same
confidence thresholds. For each evaluation folder, split and class, the metrics are computed on all the slices and
on the slices without the excluded datasets.

Output:
    OUTPUT_CSV : one row per run, split, class and subset ("all" or "without <datasets>"): threshold and main
                 metrics (see evaluate_detection.py)

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
from pathlib import Path

from evaluate_detection import compute_metrics, evaluate_volumes

# Metrics saved in the output csv (see evaluate_detection.py)
METRICS = ("n_slices", "n_volumes", "sensitivity", "specificity", "precision", "median_iou", "iou_above_0.5",
           "median_center_error_mm", "median_gt_coverage", "median_iou_3d", "median_gt_coverage_3d",
           "gt_coverage_3d_above_0.95", "median_si_coverage")


def get_parser():
    parser = argparse.ArgumentParser(description="Compare the metrics of models with and without some datasets.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval-dirs", required=True, type=Path, nargs="+",
                        help="Evaluation folders (e.g. <experiments_dir>/*/evaluation).")
    parser.add_argument("--output", required=True, type=Path, help="csv file where the metrics are saved.")
    parser.add_argument("--exclude", nargs="+", default=["spider-challenge-2023"], help="Datasets to exclude.")
    parser.add_argument("--splits", nargs="+", default=["external_test", "test"],
                        choices=["training", "validation", "test", "external_test"], help="Splits to evaluate.")
    return parser


def read_slices(path):
    """Return the records of slices.csv (written by evaluate_detection.py) with their numbers and boxes parsed."""
    records = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("conf", "iou", "center_error_mm", "gt_coverage"):
                row[key] = float(row[key]) if row[key] else None
            for key in ("gt_box", "pred_box"):
                row[key] = tuple(json.loads(row[key])) if row[key] else None
            row["slice"] = int(row["slice"])
            row["image"] = None   # only used for the plots
            records.append(row)
    return records


def main():
    args = get_parser().parse_args()
    subset_name = f"without {' '.join(args.exclude)}"

    rows = []
    for eval_dir in args.eval_dirs:
        # Name of the run: <run_dir> of <run_dir>/evaluation, else the folder itself
        run = eval_dir.parent.name if eval_dir.name == "evaluation" else eval_dir.name
        for split in args.splits:
            for slices_csv in sorted((eval_dir / split).glob("*/slices.csv")):
                name = slices_csv.parent.name
                threshold = json.loads((slices_csv.parent / "metrics.json").read_text())["threshold"]
                records = read_slices(slices_csv)
                kept = [r for r in records if r["dataset"] not in args.exclude]
                for subset, subset_records in (("all", records), (subset_name, kept)):
                    metrics = compute_metrics(subset_records, evaluate_volumes(subset_records))
                    rows.append({"run": run, "split": split, "class": name, "subset": subset,
                                 "n_excluded_slices": len(records) - len(kept) if subset != "all" else 0,
                                 "threshold": threshold,
                                 **{k: round(metrics[k], 4) if isinstance(metrics[k], float) else metrics[k]
                                    for k in METRICS}})
                for row in rows[-2:]:
                    print(f"{run:<30} {split:<13} {name:<12} {row['subset']:<35} sensitivity {row['sensitivity']}, "
                          f"specificity {row['specificity']}, median IoU {row['median_iou']}, "
                          f"median 3D IoU {row['median_iou_3d']}, 3D GT coverage >= 0.95 {row['gt_coverage_3d_above_0.95']}")

    if not rows:
        raise SystemExit(f"No <split>/<class>/slices.csv found in {[str(d) for d in args.eval_dirs]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows saved in {args.output}")


if __name__ == "__main__":
    main()
