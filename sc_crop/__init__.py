"""
sc-crop — Spinal cord detection, cropping, and segmentation.

Public API:
    detect(img_path, ...)                  → bbox              — detect SC bbox, return context
    crop(img, bbox)                         → cropped NIfTI    — crop any volume (image or label)
    uncrop(img, bbox)          → full-space NIfTI — restore any cropped volume to original space
    detect_and_crop(img_path, ...)         → (crop_nii, bbox)  — convenience: detect + crop image
    crop_nifti(img, bbox)                   → cropped NIfTI    — alias for crop() (backward compat)
    segment_onnx(img, model_path, ...)     → binary NIfTI     — detect + ONNX inference + restore
    segment_pt(img, checkpoint, ...)       → binary NIfTI     — detect + PyTorch inference + restore
    download()                             → pre-download ONNX models (auto on first use)
    download_pt()                          → pre-download PyTorch models (GPU batch)
    ensure_model()                         → path to model.onnx
    ensure_cls_model()                     → path to cls_model.onnx
    ensure_pt_model()                      → path to model.pt
    load_config()                          → loaded config dict

Quality control:
    check_label_crop(label, bbox)          → dict              — check no SC voxels lost after crop
    save_bbox_nifti(bbox, ref_nii, path)   → None              — save crop box as binary NIfTI mask
    check_seg_truncation(seg_nii, bbox)    → list[str]         — pad_* faces where seg touches crop boundary
    CropReport                                                  — accumulate and save per-volume QC

Version constants:
    __version__       → sc-crop package version  (e.g. "0.0.5")
    __model_version__ → detection model version   (e.g. "v0.0.3")
    MODEL_VERSION     → alias for __model_version__
"""

from .crop import crop, crop_nifti, detect, detect_and_crop, load_config, uncrop
from .qc import check_label_crop, CropReport, save_bbox_nifti, check_seg_truncation
from .download import (
    download,
    download_pt,
    ensure_cls_model,
    ensure_model,
    ensure_pt_model,
    _MODEL_TAG,
)
from .segment import segment_onnx, segment_pt

__version__       = "0.5.0"
__model_version__ = _MODEL_TAG
MODEL_VERSION     = _MODEL_TAG
