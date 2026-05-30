"""
End-to-end SC segmentation: sc_crop detection + nnunet_onnx inference.

Requires nnunet-onnx (optional dependency):
    pip install "nnunet-onnx @ git+https://github.com/ivadomed/nnunet-onnx.git"

Public API:
    segment_onnx(img, model_path, ...)  → binary NIfTI — ONNX Runtime, no PyTorch
    segment_pt(img, checkpoint, ...)    → binary NIfTI — PyTorch nnUNet checkpoint
"""

from __future__ import annotations

import time
from pathlib import Path

import nibabel as nib

from .crop import detect_and_crop, uncrop


def _require_nnunet_onnx():
    try:
        import nnunet_onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "nnunet-onnx is required for segmentation.\n"
            "Install with: pip install 'nnunet-onnx @ git+https://github.com/ivadomed/nnunet-onnx.git'"
        )


def segment_onnx(img, model_path: str, tile_step: float = 0.5,
                 threads: int | None = None) -> nib.Nifti1Image:
    """Detect SC, crop, run ONNX inference, restore to full image space.

    Args:
        img:        Path to NIfTI image or nibabel NIfTI object.
        model_path: Path to .onnx model exported with nnunet_onnx.export.
        tile_step:  Sliding window step as fraction of patch size (default 0.5).
        threads:    ONNX Runtime intra-op threads (default: auto).

    Returns:
        Binary segmentation NIfTI in the same space as the input image.
    """
    _require_nnunet_onnx()
    from nnunet_onnx.inference import infer_onnx

    input_path = str(img) if isinstance(img, Path) else img
    crop_nii, bbox = detect_and_crop(input_path)
    seg_crop = infer_onnx(crop_nii, model_path, tile_step=tile_step, threads=threads)
    return uncrop(seg_crop, bbox)


def segment_pt(img, checkpoint_path: str, device: str = 'cpu',
               use_mirroring: bool = False) -> nib.Nifti1Image:
    """Detect SC, crop, run PyTorch nnUNet inference, restore to full image space.

    Args:
        img:             Path to NIfTI image or nibabel NIfTI object.
        checkpoint_path: Path to fold_N/checkpoint_best.pth or checkpoint_final.pth.
        device:          'cpu', 'cuda', or 'mps' (default: 'cpu').
        use_mirroring:   Enable test-time augmentation (8x slower, default: False).

    Returns:
        Binary segmentation NIfTI in the same space as the input image.
    """
    _require_nnunet_onnx()
    from nnunet_onnx.inference import infer_pt

    input_path = str(img) if isinstance(img, Path) else img
    crop_nii, bbox = detect_and_crop(input_path)
    seg_crop = infer_pt(crop_nii, checkpoint_path, device=device, use_mirroring=use_mirroring)
    return uncrop(seg_crop, bbox)


# ── CLI entry points ───────────────────────────────────────────────────────────

def _cli_segment_onnx():
    """Entry point for: sc-segment-onnx"""
    import argparse

    p = argparse.ArgumentParser(description='SC segmentation via ONNX (no PyTorch required).')
    p.add_argument('-i',          required=True,             help='Input NIfTI image (.nii.gz)')
    p.add_argument('-o',          required=True,             help='Output segmentation mask (.nii.gz)')
    p.add_argument('--model',     required=True,             help='Path to .onnx model')
    p.add_argument('--tile-step', type=float, default=0.5,   help='Sliding window step (default: 0.5)')
    p.add_argument('--threads',   type=int,   default=None,  help='ONNX Runtime threads (default: auto)')
    args = p.parse_args()

    SEP = '─' * 38
    t0 = time.perf_counter()
    crop_nii, bbox = detect_and_crop(args.i)
    t1 = time.perf_counter()

    _require_nnunet_onnx()
    from nnunet_onnx.inference import infer_onnx
    seg_crop = infer_onnx(crop_nii, args.model, tile_step=args.tile_step, threads=args.threads)
    t2 = time.perf_counter()

    seg_full = uncrop(seg_crop, bbox)
    nib.save(seg_full, args.o)
    t3 = time.perf_counter()

    print(SEP)
    print(f'  detect + crop : {t1-t0:.1f}s')
    print(f'  inference     : {t2-t1:.1f}s')
    print(f'  restore + save: {t3-t2:.1f}s')
    print(SEP)
    print(f'  total         : {t3-t0:.1f}s  →  {args.o}')


def _cli_segment_pt():
    """Entry point for: sc-segment-pt"""
    import argparse

    p = argparse.ArgumentParser(description='SC segmentation via PyTorch nnUNet checkpoint.')
    p.add_argument('-i',           required=True,            help='Input NIfTI image (.nii.gz)')
    p.add_argument('-o',           required=True,            help='Output segmentation mask (.nii.gz)')
    p.add_argument('--checkpoint', required=True,
                   help='Path to fold_N/checkpoint_best.pth or checkpoint_final.pth')
    p.add_argument('--device',     default='cpu', choices=['cpu', 'cuda', 'mps'],
                   help='Inference device (default: cpu)')
    p.add_argument('--tta',        action='store_true',
                   help='Enable mirroring test-time augmentation (8x slower)')
    args = p.parse_args()

    SEP = '─' * 38
    t0 = time.perf_counter()
    crop_nii, bbox = detect_and_crop(args.i)
    t1 = time.perf_counter()

    _require_nnunet_onnx()
    from nnunet_onnx.inference import infer_pt
    seg_crop = infer_pt(crop_nii, args.checkpoint, device=args.device, use_mirroring=args.tta)
    t2 = time.perf_counter()

    seg_full = uncrop(seg_crop, bbox)
    nib.save(seg_full, args.o)
    t3 = time.perf_counter()

    print(SEP)
    print(f'  detect + crop : {t1-t0:.1f}s')
    print(f'  inference     : {t2-t1:.1f}s')
    print(f'  restore + save: {t3-t2:.1f}s')
    print(SEP)
    print(f'  total         : {t3-t0:.1f}s  →  {args.o}')
