"""
ONNX Runtime inference for sc_crop.

Detection model  : dynamic input [1, 3, 320, W] — rectangular letterbox (pad to a
                   multiple of 32, like YOLO's predict()). Output [1, 300, 6] —
                   [x1, y1, x2, y2, conf, class_id] in letterboxed px coords.
Classification model: fixed input [1, 3, 320, 320] — square letterbox. Output [1, 2]
                   — [prob_no_sc, prob_sc] (softmax already applied).

Both letterbox functions replicate the exact training/predict preprocessing:
cv2.resize INTER_LINEAR + grey 114 padding. This is bit-exact with YOLO PyTorch.
"""

from __future__ import annotations

import cv2
import numpy as np

_IMGSZ   = 320
_STRIDE  = 32
_CLS_SC_IDX = 1   # alphabetical: 0=no_sc, 1=sc


def load_session(model_path: str):
    """Load an ONNX InferenceSession (CPU)."""
    import onnxruntime as ort
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _letterbox_detect(sl: np.ndarray) -> tuple:
    """Rectangular letterbox for detection — bit-exact with YOLO's predict().

    Resizes preserving aspect ratio so the longest side is _IMGSZ, then pads the
    short side with grey (114) up to the next multiple of _STRIDE (rect/auto mode).
    Returns (input_tensor [1,3,H',W'], scale, pad_x, pad_y) — pad_x/pad_y are the
    left/top pixel offsets used to de-letterbox the predicted boxes.
    """
    rgb = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=2)
    h0, w0 = rgb.shape[:2]
    scale = min(_IMGSZ / h0, _IMGSZ / w0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    dw = ((_IMGSZ - new_w) % _STRIDE) / 2
    dh = ((_IMGSZ - new_h) % _STRIDE) / 2
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    inp = padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
    return inp, scale, left, top


def _letterbox_cls(sl: np.ndarray) -> np.ndarray:
    """Square 320×320 letterbox for the classifier — matches its training transform.

    Resize longest side to _IMGSZ, pad short side with grey (114) to a square,
    centred with floor/ceil split (ph//2). Returns input tensor [1,3,320,320].
    """
    rgb = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=2)
    h, w = rgb.shape[:2]
    r = _IMGSZ / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = _IMGSZ - nh, _IMGSZ - nw
    padded = cv2.copyMakeBorder(rgb, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0


def infer_slices_onnx(sess, slices: list, las_idxs: list, conf_thresh: float) -> dict:
    """Run ONNX detection inference slice by slice.

    Returns {las_idx: (cx, cy, w, h)} normalised to [0, 1] in original slice space.
    """
    preds: dict = {}
    for las_idx, sl in zip(las_idxs, slices):
        H, W = sl.shape[:2]
        inp, scale, pad_x, pad_y = _letterbox_detect(sl)
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
            out = sess.run(None, {"images": _letterbox_cls(slice_map[z])})[0][0]  # [2]
            if float(out[_CLS_SC_IDX]) >= cls_conf:
                return {z_: b for z_, b in preds.items() if z_ >= min(comp)}
    return preds
