"""
sc-crop — Spinal cord detection and cropping.

Public API:
    detect_and_crop(img_path, ...)         → (crop_nii, ctx)  — crop in memory for inference
    restore_segmentation(seg_nii, ctx)     → full-space NIfTI — restore after model inference
    run(input_path, ...)                   → detect SC bbox (+ optional crop file)
    preprocess_dataset(...)                → batch nnUNet preprocessing on GPU
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

from .crop import detect_and_crop, load_config, restore_segmentation, run
from .download import (
    download,
    download_pt,
    ensure_cls_model,
    ensure_model,
    ensure_pt_model,
    _MODEL_TAG,
)
from .nnunet import preprocess_dataset

__version__       = "0.0.5"
__model_version__ = _MODEL_TAG
MODEL_VERSION     = _MODEL_TAG
