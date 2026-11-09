"""
Compare the inference speed of all the models trained by run_experiments.sh, on the same 50 3D volumes of the
test set (MSD json), selected at random.

For each model (checkpoints/weights/best.pt of each training folder <date>_<configuration>_<model>/):
    1. load the model on the device
    2. for each volume: preprocess the image and predict all its slices in one batch
Each step is timed. Models and volumes are processed one after the other, on the GPU if there is one else on the CPU

Outputs (in OUTPUT_FOLDER):
    inference_speed_<YYYYMMDD_HHMMSS>.csv         : preprocessing and prediction time of each volume, for each model
    inference_speed_summary_<YYYYMMDD_HHMMSS>.csv : per model: number of parameters, loading time, mean
                                                    preprocessing, prediction and total time per volume (ms)

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# Inference scripts (preprocessing of an image, prediction of its slices)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from predict_image import IMAGE_SIZE, predict_slices
from preprocess_image import preprocess

SEED = 42    # random selection of the volumes
CONF = 0.25  # confidence threshold of all classes


def get_parser():
    parser = argparse.ArgumentParser(description="Compare the inference speed of the models of run_experiments.sh.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiments-dir", required=True, type=Path, help="Output folder of run_experiments.sh.")
    parser.add_argument("--msd", required=True, type=Path, help="MSD json (its test images are used).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the results are saved.")
    parser.add_argument("--n-volumes", type=int, default=50, help="Number of test volumes, selected at random.")
    return parser


def now():
    """Return the current time (s), once the GPU (if there is one) has finished its work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inference_speed_{datetime.now():%Y%m%d_%H%M%S}"

    images = sorted({entry["image"] for entry in json.loads(args.msd.read_text())["test"]})
    volumes = sorted(random.Random(SEED).sample(images, args.n_volumes))
    # Read every image once, so that the first model does not also time the first reading of the files
    for v in volumes:
        Path(v).read_bytes()

    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")

    rows, summary = [], []
    for weights in sorted(args.experiments_dir.glob("*/checkpoints/weights/best.pt")):
        name = weights.parents[2].name.split("_", 2)[-1]   # <configuration>_<model>, without the date
        rgb = name.startswith("rgb")

        # YOLO moves the model to the GPU (if there is one) at its first prediction: one empty slice is predicted
        start = now()
        model = YOLO(weights)
        model.predict(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8), verbose=False)
        loading = now() - start

        model_rows = []
        for v in volumes:
            start = now()
            slices = preprocess(v, rgb)
            preprocessed = now()
            predict_slices(model, slices, {cls: CONF for cls in model.names})
            end = now()
            model_rows.append({"model": name, "volume": v, "n_slices": len(slices),
                               "preprocessing_ms": round(1000 * (preprocessed - start), 2),
                               "prediction_ms": round(1000 * (end - preprocessed), 2),
                               "total_ms": round(1000 * (end - start), 2)})
        rows += model_rows

        summary.append({"model": name, "n_parameters": sum(p.numel() for p in model.model.parameters()),
                        "loading_ms": round(1000 * loading, 2)})
        for key in ("preprocessing_ms", "prediction_ms", "total_ms"):
            summary[-1][f"mean_{key}"] = round(float(np.mean([r[key] for r in model_rows])), 2)
        print(summary[-1])

    for path, table in ((args.output_dir / f"{stem}.csv", rows),
                        (args.output_dir / f"{stem.replace('speed', 'speed_summary')}.csv", summary)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(f"{len(summary)} models, {len(volumes)} volumes. Saved in {args.output_dir} ({stem}*.csv)")


if __name__ == "__main__":
    main()
