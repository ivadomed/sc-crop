"""
ONNX Runtime inference for sc_crop.

Detection model  : output [1, 300, 6] — [x1, y1, x2, y2, conf, class_id] in px coords (0-320)
Classification model: output [1, 2]  — [prob_no_sc, prob_sc] (softmax already applied)

Both models expect input [1, 3, 320, 320] float32 in [0, 1].
"""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

_IMGSZ   = 320
_CLS_SC_IDX = 1   # alphabetical: 0=no_sc, 1=sc


def load_session(model_path: str):
    """Load an ONNX InferenceSession (CPU)."""
    import onnxruntime as ort
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _letterbox(sl: np.ndarray) -> tuple:
    """Letterbox-resize to _IMGSZ×_IMGSZ, preserving aspect ratio.

    Returns (input_tensor [1,3,H,W], scale, pad_x, pad_y) where pad_x/pad_y are
    the pixel offsets added on each side (left/top respectively). Matches the
    preprocessing applied by YOLO's PyTorch predict() so ONNX and PyTorch see
    identical inputs.
    """
    H, W = sl.shape[:2]
    scale = min(_IMGSZ / H, _IMGSZ / W)
    new_H, new_W = int(round(H * scale)), int(round(W * scale))
    rgb = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=2)
    resized = np.array(PILImage.fromarray(rgb).resize((new_W, new_H), PILImage.BILINEAR))
    pad_x = (_IMGSZ - new_W) / 2
    pad_y = (_IMGSZ - new_H) / 2
    canvas = np.zeros((_IMGSZ, _IMGSZ, 3), dtype=np.uint8)
    x0, y0 = int(pad_x), int(pad_y)
    canvas[y0:y0 + new_H, x0:x0 + new_W] = resized
    arr = canvas.astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis], scale, pad_x, pad_y


def infer_slices_onnx(sess, slices: list, las_idxs: list, conf_thresh: float) -> dict:
    """Run ONNX detection inference slice by slice.

    Returns {las_idx: (cx, cy, w, h)} normalised to [0, 1] in original slice space.
    """
    preds: dict = {}
    for las_idx, sl in zip(las_idxs, slices):
        H, W = sl.shape[:2]
        inp, scale, pad_x, pad_y = _letterbox(sl)
        out = sess.run(None, {"images": inp})[0][0]   # [300, 6]
        mask  = (out[:, 4] >= conf_thresh) & (out[:, 5] == 0)
        valid = out[mask]
        if len(valid) == 0:
            continue
        best = valid[valid[:, 4].argmax()]
        # de-pad and de-scale from letterboxed 320×320 space to original slice space
        x1 = (best[0] - pad_x) / scale / W
        y1 = (best[1] - pad_y) / scale / H
        x2 = (best[2] - pad_x) / scale / W
        y2 = (best[3] - pad_y) / scale / H
        preds[las_idx] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return preds


def cls_comp_filter_onnx(preds: dict, slices: list, las_idxs: list,
                          sess, cls_conf: float) -> dict:
    """Connected-component cls regularization using ONNX runtime.

    Processes components from most superior (lowest z). Runs cls slice by slice
    within each component, stops at first positive (prob_sc >= cls_conf).
    Returns all preds with z >= min_z of the first validated component.
    Fallback: returns all preds if no component is validated.
    """
    from .crop import _si_connected_components
    comps     = _si_connected_components(preds)
    slice_map = {idx: sl for idx, sl in zip(las_idxs, slices)}

    for comp in comps:
        for z in comp:
            if z not in slice_map:
                continue
            out = sess.run(None, {"images": _letterbox(slice_map[z])[0]})[0][0]  # [2]
            if float(out[_CLS_SC_IDX]) >= cls_conf:
                return {z_: b for z_, b in preds.items() if z_ >= min(comp)}
    return preds
