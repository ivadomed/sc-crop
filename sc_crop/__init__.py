"""
sc-crop — Spinal cord detection and cropping.

Public API:
    detect(img_path, ...)                  → bbox              — detect SC bbox, return context
    crop(img, bbox)                         → cropped NIfTI    — crop any volume (image or label)
    uncrop(img, bbox)          → full-space NIfTI — restore any cropped volume to original space
    detect_and_crop(img_path, ...)         → (crop_nii, bbox)  — convenience: detect + crop image
    crop_nifti(img, bbox)                   → cropped NIfTI    — alias for crop() (backward compat)
    download()                             → pre-download ONNX models (auto on first use)
    download_pt()                          → pre-download PyTorch models (GPU batch)
    ensure_model()                         → path to model.onnx
    ensure_cls_model()                     → path to cls_model.onnx
    ensure_pt_model()                      → path to model.pt
    load_config()                          → loaded config dict

Version constants:
    __version__       → sc-crop package version  (e.g. "0.0.5")
    __model_version__ → detection model version   (e.g. "v0.0.3")
    MODEL_VERSION     → alias for __model_version__
"""

from .crop import crop, crop_nifti, detect, detect_and_crop, load_config, uncrop
from .download import (
    download,
    download_pt,
    ensure_cls_model,
    ensure_model,
    ensure_pt_model,
    _MODEL_TAG,
)

__version__       = "0.1.0"
__model_version__ = _MODEL_TAG
MODEL_VERSION     = _MODEL_TAG
