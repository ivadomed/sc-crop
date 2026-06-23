"""
Slice-by-slice comparison for the classifier: YOLO.pt vs ONNX (onnxruntime).

The PT model uses LetterboxClsTransform (pickled in checkpoint).
The ONNX model uses onnxruntime directly with the same letterbox preprocessing.

Usage:
    python scripts/compare_cls_inference.py \
        --png-dir processed/10mm_SI_1mm_axial_3ch_normslice_all/basel-mp2rage/sub-C001_UNIT1/png \
        --pt      runs/20260529_090157/checkpoints_cls/weights/loss_best.pt \
        --onnx    ~/.cache/sc_crop/v0.0.10/cls_model.onnx \
        --imgsz   320
"""

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import onnxruntime as ort


def _inject_letterbox_cls_transform() -> None:
    """Inject LetterboxClsTransform into sys.modules['train'] so torch.load() can unpickle the PT checkpoint."""
    import cv2 as _cv2

    class LetterboxClsTransform:
        def __init__(self, imgsz, augment, hsv_v, scale, degrees, translate, fliplr, flipud):
            self.imgsz = imgsz; self.augment = augment; self.hsv_v = hsv_v
            self.scale = scale; self.degrees = degrees; self.translate = translate
            self.fliplr = fliplr; self.flipud = flipud; self.transforms = [self]
        def __getattr__(self, name):
            if name == "transforms": return [self]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        def __call__(self, pil_img):
            img = np.array(pil_img, dtype=np.uint8)
            h, w = img.shape[:2]; r = self.imgsz / max(h, w)
            nh, nw = round(h * r), round(w * r)
            if nh != h or nw != w:
                img = _cv2.resize(img, (nw, nh), interpolation=_cv2.INTER_LINEAR)
            ph, pw = self.imgsz - nh, self.imgsz - nw
            img = _cv2.copyMakeBorder(img, ph//2, ph-ph//2, pw//2, pw-pw//2,
                                      _cv2.BORDER_CONSTANT, value=(114, 114, 114))
            import torchvision.transforms.functional as F
            return F.to_tensor(img)

    module = types.ModuleType("train")
    module.LetterboxClsTransform = LetterboxClsTransform
    sys.modules["train"] = module


def _letterbox(img: np.ndarray, imgsz: int) -> np.ndarray:
    """Square letterbox — matches LetterboxClsTransform from training. Returns float32 NCHW [1,3,H,W]."""
    rgb = img if img.ndim == 3 else np.stack([img] * 3, axis=2)
    h, w = rgb.shape[:2]
    r = imgsz / max(h, w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = imgsz - nh, imgsz - nw
    padded = cv2.copyMakeBorder(rgb, ph//2, ph-ph//2, pw//2, pw-pw//2,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0


def sc_prob_pt(results, sc_idx: int):
    r = results[0]
    if r.probs is None:
        return None
    return float(r.probs.data[sc_idx])


def get_sc_idx(model) -> int:
    for idx, name in model.names.items():
        if name == "sc":
            return int(idx)
    raise ValueError(f"Class 'sc' not found: {model.names}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--pt",      required=True, help="loss_best.pt (training reference)")
    parser.add_argument("--onnx",    required=True, help="cls_model.onnx")
    parser.add_argument("--imgsz",   type=int, default=320)
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()

    _inject_letterbox_cls_transform()

    png_dir = Path(args.png_dir)
    pngs    = sorted(png_dir.glob("slice_*.png"))
    assert pngs, f"No slice_*.png found in {png_dir}"

    pt_model  = YOLO(args.pt)
    cls_sess  = ort.InferenceSession(args.onnx)
    sc_idx    = get_sc_idx(pt_model)

    print(f"SC class index: {sc_idx}  |  imgsz={args.imgsz}")
    print(f"{'slice':<12} {'pt_prob':>8} {'onnx_prob':>9} {'Δprob':>8}")
    print("─" * 45)

    diffs = []
    for png in pngs:
        ref_prob = sc_prob_pt(pt_model.predict(str(png), imgsz=args.imgsz, verbose=False), sc_idx)

        img = np.array(Image.open(png).convert("RGB"), dtype=np.uint8)
        out = cls_sess.run(None, {"images": _letterbox(img, args.imgsz)})[0][0]
        onnx_prob = float(out[1])  # sc index = 1

        if ref_prob is None:
            print(f"{png.name:<12}  pt probs unavailable")
            continue

        d = onnx_prob - ref_prob
        diffs.append(d)
        flag = "  ← MISMATCH" if abs(d) > 1e-4 else ""
        print(f"{png.name:<12} {ref_prob:>8.4f} {onnx_prob:>9.4f} {d:>8.4f}{flag}")

    if diffs:
        print(f"\n{'─' * 45}")
        print(f"Max |Δprob| : {max(abs(d) for d in diffs):.6f}")
        print(f"Mean |Δprob|: {np.mean(np.abs(diffs)):.6f}")


if __name__ == "__main__":
    main()
