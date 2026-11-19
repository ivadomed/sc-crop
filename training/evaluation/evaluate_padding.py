"""
Evaluate the crops of trained models when the predicted boxes are padded: by default 0x0x0, 0x0x10, 5x5x20 and
10x10x30 mm (RL x AP x SI, added on each side of the box), for each class (SC and canal).

The metrics are recomputed from the results of each slice (slices.csv) saved by evaluate_detection.py (e.g. at the
end of train_detection.py in <run_dir>/evaluation/): no new prediction, same confidence thresholds.
Slices are 1 x 1 x 10 mm (RL x AP x SI), so an SI padding of 10 mm adds 1 slice above and 1 below the box. The SI
padding stops at the first and last slice of the volume; the RL/AP padding is not limited to the image.

Metrics, for each padding (missed volumes and slices count as 0):
    Volume level (3D box = crop), on volumes with the structure:
        3D IoU, 3D GT coverage (fraction of the GT 3D box inside the padded box), SI coverage,
        % of volumes with 3D GT coverage >= 0.95 and = 1 (whole GT box in the crop),
        crop / GT volume ratio (size of the crop compared to the GT box)
    Slice level, on slices with the structure and a predicted box (RL/AP padding only):
        GT coverage, % of slices with GT coverage >= 0.95 and = 1

Output:
    OUTPUT_CSV : one row per run, split, class, padding and dataset ("overall", then each dataset)

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_detection import area, evaluate_volumes, intersection, volume_3d
from evaluate_without_dataset import read_slices

# Voxel size of the slices (RL, AP, SI) in mm, as in data/build_yolo_dataset.py
VOXEL_SIZE = (1.0, 1.0, 10.0)


def get_parser():
    parser = argparse.ArgumentParser(description="Evaluate the crops of models with padded predicted boxes.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--eval-dirs", required=True, type=Path, nargs="+",
                        help="Evaluation folders (e.g. <experiments_dir>/*/evaluation).")
    parser.add_argument("--output", required=True, type=Path, help="csv file where the metrics are saved.")
    parser.add_argument("--paddings", nargs="+", default=["0x0x0", "0x0x10", "5x5x10", "5x5x20", "10x10x30"],
                        help="Paddings RLxAPxSI in mm, added on each side of the predicted boxes.")
    parser.add_argument("--splits", nargs="+", default=["test", "external_test"],
                        choices=["training", "validation", "test", "external_test"], help="Splits to evaluate.")
    return parser


def pad_volume(box, padding, n_slices):
    """Pad a 3D box (x1, y1, z1, x2, y2, z2) by `padding` (x, y, z) on each side; z stays in [0, n_slices]."""
    px, py, pz = padding
    return (box[0] - px, box[1] - py, max(box[2] - pz, 0), box[3] + px, box[4] + py, min(box[5] + pz, n_slices))


def padding_metrics(records, volumes, n_slices, padding):
    """Return the volume and slice metrics of `records` (slices of one class) with the predicted boxes padded by
    `padding` (x, y, z) in voxels. `volumes`: from evaluate_volumes(records); `n_slices`: {volume: number of slices}."""
    ious, coverages, si_coverages, ratios = [], [], [], []
    for v in volumes:
        gt = v["gt_box_3d"]
        if v["pred_box_3d"] is None:   # structure missed on the whole volume
            ious.append(0.0)
            coverages.append(0.0)
            si_coverages.append(0.0)
            continue
        pred = pad_volume(v["pred_box_3d"], padding, n_slices[v["volume"]])
        inter = volume_3d((*np.maximum(gt[:3], pred[:3]), *np.minimum(gt[3:], pred[3:])))
        ious.append(inter / (volume_3d(gt) + volume_3d(pred) - inter))
        coverages.append(inter / volume_3d(gt))
        si_coverages.append(max(0, min(gt[5], pred[5]) - max(gt[2], pred[2])) / (gt[5] - gt[2]))
        ratios.append(volume_3d(pred) / volume_3d(gt))

    # Slice level: in-plane padding of the predicted box of each detected slice
    px, py, _ = padding
    slice_coverages = [intersection(r["gt_box"], (r["pred_box"][0] - px, r["pred_box"][1] - py,
                                                  r["pred_box"][2] + px, r["pred_box"][3] + py)) / area(r["gt_box"])
                       for r in records if r["outcome"] == "TP"]

    def median(values):
        return round(float(np.median(values)), 4) if values else None

    def fraction(values, threshold):
        return round(float(np.mean(np.array(values) >= threshold)), 4) if values else None

    return {"n_volumes": len(volumes), "median_iou_3d": median(ious), "median_gt_coverage_3d": median(coverages),
            "gt_coverage_3d_above_0.95": fraction(coverages, 0.95), "gt_coverage_3d_full": fraction(coverages, 1.0),
            "median_si_coverage": median(si_coverages), "median_crop_to_gt_ratio": median(ratios),
            "n_detected_slices": len(slice_coverages), "median_slice_gt_coverage": median(slice_coverages),
            "slice_gt_coverage_above_0.95": fraction(slice_coverages, 0.95),
            "slice_gt_coverage_full": fraction(slice_coverages, 1.0)}


def main():
    args = get_parser().parse_args()
    # Paddings in mm → in voxels (x = RL, y = AP, z = SI)
    paddings = {p: tuple(round(float(mm) / size) for mm, size in zip(p.split("x"), VOXEL_SIZE)) for p in args.paddings}

    rows = []
    for eval_dir in args.eval_dirs:
        # Name of the run: <run_dir> of <run_dir>/evaluation, else the folder itself
        run = eval_dir.parent.name if eval_dir.name == "evaluation" else eval_dir.name
        for split in args.splits:
            for slices_csv in sorted((eval_dir / split).glob("*/slices.csv")):
                name = slices_csv.parent.name
                threshold = json.loads((slices_csv.parent / "metrics.json").read_text())["threshold"]
                records = read_slices(slices_csv)
                n_slices = defaultdict(int)
                for r in records:
                    n_slices[r["volume"]] += 1
                by_dataset = defaultdict(list)
                for r in records:
                    by_dataset[r["dataset"]].append(r)

                for padding_name, padding in paddings.items():
                    for dataset, dataset_records in [("overall", records), *sorted(by_dataset.items())]:
                        metrics = padding_metrics(dataset_records, evaluate_volumes(dataset_records), n_slices, padding)
                        rows.append({"run": run, "split": split, "class": name, "threshold": threshold,
                                     "padding_mm": padding_name, "dataset": dataset, **metrics})
                        if dataset == "overall":
                            print(f"{run:<30} {split:<13} {name:<12} padding {padding_name:<9} "
                                  f"3D GT coverage: median {metrics['median_gt_coverage_3d']}, "
                                  f">= 0.95 {metrics['gt_coverage_3d_above_0.95']}, full {metrics['gt_coverage_3d_full']}; "
                                  f"slice GT coverage >= 0.95 {metrics['slice_gt_coverage_above_0.95']}; "
                                  f"median 3D IoU {metrics['median_iou_3d']}, "
                                  f"crop / GT volume {metrics['median_crop_to_gt_ratio']}")

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
