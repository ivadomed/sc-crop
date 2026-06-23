"""
Slice-by-slice comparison: YOLO.pt vs cached ONNX vs freshly exported ONNX.

REF  : YOLO("best.pt").predict()        — ground truth (same as evaluate.py)
OLD  : YOLO("model.onnx").predict()     — cached ONNX from release
NEW  : YOLO("<fresh>.onnx").predict()   — re-exported from best.pt on the fly

Usage:
    python scripts/compare_inference.py \\
        --png-dir processed/10mm_SI_1mm_axial_3ch_normslice_all/basel-mp2rage/sub-C001_UNIT1/png \\
        --pt      runs/20260528_192341/checkpoints/weights/best.pt \\
        --onnx    ~/.cache/sc_crop/v0.0.10/model.onnx \\
        --conf    0.1 \\
        [--out    compare_out.csv]
"""

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from ultralytics import YOLO


def best_box(results, conf_thresh: float):
    """Return best box as (cx, cy, w, h, conf) normalised, or None."""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    mask = boxes.conf >= conf_thresh
    if not mask.any():
        return None
    best_idx = int(boxes.conf[mask].argmax())
    cx, cy, w, h = boxes.xywhn[mask][best_idx].tolist()
    conf = float(boxes.conf[mask][best_idx])
    return (cx, cy, w, h, conf)


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO.pt.predict() vs YOLO.onnx.predict() per slice.")
    parser.add_argument("--png-dir", required=True, help="Directory of input PNGs")
    parser.add_argument("--pt",      required=True, help="Path to best.pt (training reference)")
    parser.add_argument("--onnx",    required=True, help="Path to model.onnx")
    parser.add_argument("--conf",   type=float, default=0.1)
    parser.add_argument("--imgsz", type=int,   default=320, help="Inference imgsz (must match training)")
    parser.add_argument("--out",   default=None, help="Output CSV path")
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    pngs    = sorted(png_dir.glob("slice_*.png"))
    assert pngs, f"No slice_*.png found in {png_dir}"

    # Export fresh ONNX from best.pt into a temp dir
    pt_path = Path(args.pt)
    with tempfile.TemporaryDirectory() as tmp:
        import shutil
        tmp_pt = Path(tmp) / pt_path.name
        shutil.copy2(pt_path, tmp_pt)
        print(f"Exporting fresh ONNX from {pt_path.name} (dynamic=True, opset=19) …")
        tmp_model = YOLO(str(tmp_pt))
        tmp_model.export(format="onnx", imgsz=320, opset=19, dynamic=True)
        fresh_onnx = tmp_pt.with_suffix(".onnx")
        assert fresh_onnx.exists()
        print(f"Fresh ONNX: {fresh_onnx}")

        imgsz = args.imgsz

        pt_model    = YOLO(args.pt)
        old_model   = YOLO(args.onnx)
        fresh_model = YOLO(str(fresh_onnx))

        rows = []
        header = ["slice",
                  "pt", "old_onnx", "fresh_onnx",
                  "old_dcx", "old_dcy", "old_dconf",
                  "fresh_dcx", "fresh_dcy", "fresh_dconf"]

        print(f"\n{'slice':<12} {'pt':>3} {'old':>4} {'new':>4}  "
              f"{'old_Δcx':>8} {'old_Δcy':>8} {'old_Δconf':>9}  "
              f"{'new_Δcx':>8} {'new_Δcy':>8} {'new_Δconf':>9}")
        print("─" * 95)

        for png in pngs:
            ref   = best_box(pt_model.predict(str(png),    conf=args.conf, imgsz=imgsz, verbose=False), args.conf)
            old   = best_box(old_model.predict(str(png),   conf=args.conf, imgsz=imgsz, verbose=False), args.conf)
            fresh = best_box(fresh_model.predict(str(png), conf=args.conf, imgsz=imgsz, verbose=False), args.conf)

            def _delta(a, b):
                if a and b:
                    return f"{b[0]-a[0]:>8.4f} {b[1]-a[1]:>8.4f} {b[4]-a[4]:>9.4f}"
                elif a and not b:
                    return "   MISS   MISS      MISS"
                elif not a and b:
                    return "  EXTRA  EXTRA     EXTRA"
                return "      -       -          -"

            ref_d = "Y" if ref else "N"
            old_d = "Y" if old else "N"
            frh_d = "Y" if fresh else "N"
            old_str   = _delta(ref, old)
            fresh_str = _delta(ref, fresh)

            row = [png.name, ref_d, old_d, frh_d,
                   *([f"{old[0]-ref[0]:.4f}", f"{old[1]-ref[1]:.4f}", f"{old[4]-ref[4]:.4f}"]
                     if ref and old else ["", "", ""]),
                   *([f"{fresh[0]-ref[0]:.4f}", f"{fresh[1]-ref[1]:.4f}", f"{fresh[4]-ref[4]:.4f}"]
                     if ref and fresh else ["", "", ""])]
            rows.append(row)
            print(f"{png.name:<12}  {ref_d}   {old_d}   {frh_d}  {old_str}  {fresh_str}")

        print(f"\n{'─' * 95}")
        # col=0 → old_onnx (cols 4,5,6), col=1 → fresh_onnx (cols 7,8,9)
        for label, det_col, dcx_col in [("old_onnx", 2, 4), ("fresh_onnx", 3, 7)]:
            mm  = sum(1 for r in rows if r[det_col] != r[1])
            dcx = [float(r[dcx_col]) for r in rows if r[1] == "Y" and r[det_col] == "Y" and r[dcx_col] != ""]
            if dcx:
                print(f"{label:<12} mismatches={mm}  mean|Δcx|={np.mean(np.abs(dcx)):.4f}  max|Δcx|={max(np.abs(dcx)):.4f}")
            else:
                print(f"{label:<12} mismatches={mm}")

        if args.out:
            with open(args.out, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"\nCSV → {args.out}")


if __name__ == "__main__":
    main()
