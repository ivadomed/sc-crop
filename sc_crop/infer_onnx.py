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


def _preprocess(sl: np.ndarray) -> np.ndarray:
    """Resize slice to _IMGSZ×_IMGSZ, normalize to [0,1], return [1,3,H,W] float32."""
    rgb = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=2)
    img = PILImage.fromarray(rgb).resize((_IMGSZ, _IMGSZ), PILImage.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]   # [1, 3, H, W]


def infer_slices_onnx(sess, slices: list, las_idxs: list, conf_thresh: float) -> dict:
    """Run ONNX detection inference slice by slice.

    Returns {las_idx: (cx, cy, w, h)} normalised to [0, 1].
    """
    preds: dict = {}
    for las_idx, sl in zip(las_idxs, slices):
        out = sess.run(None, {"images": _preprocess(sl)})[0][0]   # [300, 6]
        mask  = (out[:, 4] >= conf_thresh) & (out[:, 5] == 0)
        valid = out[mask]
        if len(valid) == 0:
            continue
        best       = valid[valid[:, 4].argmax()]
        x1, y1, x2, y2 = best[:4] / _IMGSZ
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
            out = sess.run(None, {"images": _preprocess(slice_map[z])})[0][0]  # [2]
            if float(out[_CLS_SC_IDX]) >= cls_conf:
                return {z_: b for z_, b in preds.items() if z_ >= min(comp)}
    return preds
