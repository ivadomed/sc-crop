"""
Detect the spinal cord (SC, class 0) and the spinal canal (class 1) on one nifti image with a trained YOLO model.

The image is preprocessed into axial slices (preprocess_image.py), then all the slices are predicted in one
batch. On each slice, the most confident box of each class is kept, if its confidence is >= the threshold of the
class (--conf): one value for all classes (e.g. --conf 0.25), or one value per class, in the order of the
classes: SC then canal (e.g. --conf 0.3 0.5, such as the thresholds chosen on the validation set at the end of
the training, in <run_dir>/evaluation/thresholds.json).

The prediction runs on the GPU if there is one, else on the CPU.

Outputs:
    OUTPUT_CSV : one row per kept box: slice (0 = most superior), class, class name, confidence and box
                 (x1, y1, x2, y2) in pixels of the slice (1 x 1 mm, row 0 = Anterior, column 0 = patient Right)

Author: Pierre-Louis Benveniste
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

from preprocess_image import preprocess

IMAGE_SIZE = 320  # size of the slices used for training and prediction (pixels)


def get_parser():
    parser = argparse.ArgumentParser(description="Detect the SC and the canal on one nifti image.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True, type=Path, help="Image (nii.gz).")
    parser.add_argument("--model", required=True, type=Path, help="YOLO weights (best.pt of a training).")
    parser.add_argument("--output", required=True, type=Path, help="csv file where the boxes are saved.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (model trained on RGB slices).")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25],
                        help="Confidence threshold: one value for all classes, or one value per class "
                             "(SC canal, e.g. --conf 0.3 0.5).")
    return parser


def get_thresholds(conf, names):
    """Return {class: threshold} from the --conf values: one for all classes, or one per class (in the order
    of `names` = {class: name})."""
    if len(conf) == 1:
        return {cls: conf[0] for cls in names}
    if len(conf) != len(names):
        raise SystemExit(f"--conf: give 1 value or {len(names)} values ({', '.join(names.values())}), got {len(conf)}")
    return dict(zip(sorted(names), conf))


def predict_slices(model, slices, thresholds, imgsz=IMAGE_SIZE, device=None):
    """Predict all the slices in one batch. Return, for each slice, {class: ((x1, y1, x2, y2), confidence)}
    with the most confident box of each class whose confidence is >= the threshold of the class"""
    results = model.predict(source=slices, imgsz=imgsz, device=device, conf=min(thresholds.values()),
                            batch=len(slices), verbose=False)
    preds = []
    for r in results:
        best = {}
        for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            cls = int(cls)
            if conf >= thresholds[cls] and (cls not in best or conf > best[cls][1]):
                best[cls] = (tuple(box), conf)
        preds.append(best)
    return preds


def predict_image(image, model, output, rgb=False, conf=(0.25,)):
    """Detect the SC and the canal on a nifti image with the YOLO weights `model`
    and save the kept boxes in the csv file `output`"""
    model = YOLO(model)
    thresholds = get_thresholds(conf, model.names)
    print("Thresholds: " + ", ".join(f"{model.names[cls]} {t}" for cls, t in thresholds.items()))
    preds = predict_slices(model, preprocess(image, rgb), thresholds)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "class", "name", "confidence", "x1", "y1", "x2", "y2"])
        for i, boxes in enumerate(preds):
            for cls, (box, box_conf) in sorted(boxes.items()):
                writer.writerow([i, cls, model.names[cls], round(box_conf, 4)] + [round(x, 1) for x in box])
    found = ", ".join(f"{name} on {sum(cls in boxes for boxes in preds)}" for cls, name in model.names.items())
    print(f"{len(preds)} slices, boxes: {found}. Saved in {output}")
    return preds


def main():
    args = get_parser().parse_args()
    predict_image(args.image, args.model, args.output, args.rgb, args.conf)


if __name__ == "__main__":
    main()
