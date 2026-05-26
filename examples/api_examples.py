"""
sc-crop API examples — runnable cookbook.

Shows all major usage patterns for the Python API, from the simplest
to the most advanced. Each example is self-contained and can be run
independently by calling it directly.

Requirements:
    pip install "sc-crop @ git+https://github.com/ivadomed/sc-crop.git"

Usage:
    python examples/api_examples.py image.nii.gz
    python examples/api_examples.py image.nii.gz --all     # run all examples
    python examples/api_examples.py image.nii.gz --ex 3    # run example 3 only
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from sc_crop import detect, crop, detect_and_crop, restore_segmentation


# ─── Example 1 — Basic: detect + crop ────────────────────────────────────────

def example_basic(img_path: str) -> None:
    """Detect the SC bbox, then crop image and label with the same bbox."""
    print("\n── Example 1: Basic detect + crop ──────────────────────────────")

    # detect() is pure: no files written, returns a bbox context dict
    ctx = detect(img_path)
    print(f"  bbox: x[{ctx['xmin']}:{ctx['xmax']}]  "
          f"y[{ctx['ymin']}:{ctx['ymax']}]  "
          f"z[{ctx['zmin']}:{ctx['zmax']}]")
    print(f"  orientation: {ctx['original_axcodes']}")

    # crop() works on ANY volume in the same space (image, label, …)
    crop_img = crop(nib.load(img_path), ctx)
    print(f"  original shape : {nib.load(img_path).shape}")
    print(f"  cropped shape  : {crop_img.shape}")

    out = Path(img_path).with_name(Path(img_path).name.replace(".nii", "_ex1_crop.nii"))
    nib.save(crop_img, out)
    print(f"  → {out}")


# ─── Example 2 — Multi-volume: image + label with the same bbox ──────────────

def example_multi_volume(img_path: str) -> None:
    """Detect once, crop both image and label — they share the same bbox."""
    print("\n── Example 2: Multi-volume (image + label) ──────────────────────")

    # Simulate a label by binarizing the image
    img         = nib.load(img_path)
    fake_label  = nib.Nifti1Image((np.asarray(img.dataobj) > 0).astype(np.uint8),
                                   img.affine, img.header)

    ctx         = detect(img_path)
    crop_img    = crop(img,        ctx)
    crop_label  = crop(fake_label, ctx)

    assert crop_img.shape == crop_label.shape, "image and label must have the same cropped shape"
    print(f"  cropped image shape : {crop_img.shape}")
    print(f"  cropped label shape : {crop_label.shape}  (identical bbox ✓)")

    stem = Path(img_path).name.replace(".nii.gz", "").replace(".nii", "")
    parent = Path(img_path).parent
    nib.save(crop_img,   parent / f"{stem}_ex2_crop.nii.gz")
    nib.save(crop_label, parent / f"{stem}_ex2_label_crop.nii.gz")
    print(f"  → {parent / f'{stem}_ex2_crop.nii.gz'}")
    print(f"  → {parent / f'{stem}_ex2_label_crop.nii.gz'}")


# ─── Example 3 — Padding variants ────────────────────────────────────────────

def example_padding(img_path: str) -> None:
    """Show the 3 padding styles and how priority works."""
    print("\n── Example 3: Padding variants ──────────────────────────────────")

    # Defaults: superior=40, inferior=60, left=right=10, anterior=posterior=15
    ctx_default = detect(img_path)

    # Symmetric SI shorthand (common for focused cervical datasets)
    ctx_si = detect(img_path, pad_si=30)

    # Shorthand + override one face (highest priority)
    ctx_mix = detect(img_path, pad_si=30, pad_inferior=60)

    # Full individual control (all 6 faces)
    ctx_full = detect(img_path,
                      pad_superior=40, pad_inferior=60,
                      pad_left=10,     pad_right=10,
                      pad_anterior=15, pad_posterior=15)

    def _z(ctx): return ctx["zmax"] - ctx["zmin"]
    def _x(ctx): return ctx["xmax"] - ctx["xmin"]

    print(f"  {'style':<30} {'z_range':>8}  {'x_range':>8}")
    print(f"  {'─'*50}")
    print(f"  {'default (sup=40, inf=60)':<30} {_z(ctx_default):>8}  {_x(ctx_default):>8}")
    print(f"  {'pad_si=30 (symmetric)':<30} {_z(ctx_si):>8}  {_x(ctx_si):>8}")
    print(f"  {'pad_si=30 + pad_inf=60':<30} {_z(ctx_mix):>8}  {_x(ctx_mix):>8}")
    print(f"  {'full individual':<30} {_z(ctx_full):>8}  {_x(ctx_full):>8}")

    # Priority rule: pad_si=30 gives inf=30, but pad_inferior=60 overrides → bigger z range
    assert _z(ctx_mix) >= _z(ctx_si), "pad_inferior=60 should give >= padding than pad_si=30"
    print("  priority rule verified: individual > shorthand ✓")


# ─── Example 4 — detect_and_crop convenience wrapper ─────────────────────────

def example_detect_and_crop(img_path: str) -> None:
    """Single-call convenience when only cropping one volume."""
    print("\n── Example 4: detect_and_crop() one-liner ───────────────────────")

    crop_nii, ctx = detect_and_crop(img_path)
    print(f"  cropped shape : {crop_nii.shape}")
    print(f"  ctx keys      : {[k for k in ctx if not k.startswith('_')]}")

    out = Path(img_path).with_name(Path(img_path).name.replace(".nii", "_ex4_crop.nii"))
    nib.save(crop_nii, out)
    print(f"  → {out}")


# ─── Example 5 — restore_segmentation ────────────────────────────────────────

def example_restore(img_path: str) -> None:
    """Fake model (center cylinder) → restore segmentation to the original space.

    After sc-crop the spinal cord sits near the center of the RL/AP axes, so a
    small central cylinder approximates where a real model would place the SC
    segmentation. Open the output in FSLeyes alongside the original to verify
    the segmentation lands at the correct anatomical location.
    """
    print("\n── Example 5: fake model + restore_segmentation() ──────────────")

    crop_nii, ctx = detect_and_crop(img_path)

    # Fake model: cylinder centered in RL/AP, full SI extent
    X, Y, Z = crop_nii.shape[:3]
    cx, cy   = X / 2, Y / 2
    radius   = max(1, min(X, Y) // 6)
    xx, yy   = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    cylinder = ((xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2)
    mask     = np.zeros((X, Y, Z), dtype=np.uint8)
    mask[cylinder] = 1
    fake_seg = nib.Nifti1Image(mask, crop_nii.affine)

    # Restore to the full original image space
    seg_full   = restore_segmentation(fake_seg, ctx)
    orig_shape = nib.load(img_path).shape[:3]
    assert seg_full.shape == orig_shape, f"expected {orig_shape}, got {seg_full.shape}"

    n = int(np.asarray(seg_full.dataobj).sum())
    print(f"  crop shape          : {crop_nii.shape}  (cylinder r={radius}px)")
    print(f"  restored seg shape  : {seg_full.shape}  ({n} non-zero voxels)")

    out = Path(img_path).with_name(Path(img_path).name.replace(".nii", "_ex5_seg.nii"))
    nib.save(seg_full, out)
    print(f"  → {out}")
    print(f"  verify: fsleyes {img_path} {out} -cm red")


# ─── Example 6 — GPU inference (PyTorch, no ONNX) ────────────────────────────

def example_gpu(img_path: str, device: str = "cuda") -> None:
    """Use PyTorch inference on GPU instead of the default ONNX Runtime."""
    print("\n── Example 6: GPU inference (--no-onnx) ─────────────────────────")
    print("  (requires: pip install 'sc-crop[yolo]'  and  CUDA/MPS device)")
    try:
        ctx = detect(img_path, use_onnx=False, device=device)
        print(f"  GPU ({device}) inference bbox: x[{ctx['xmin']}:{ctx['xmax']}]")
    except Exception as exc:
        print(f"  skipped — {exc}")


# ─── Runner ───────────────────────────────────────────────────────────────────

EXAMPLES = {
    1: ("Basic detect + crop",                  example_basic),
    2: ("Multi-volume: image + label",           example_multi_volume),
    3: ("Padding variants",                      example_padding),
    4: ("detect_and_crop() one-liner",           example_detect_and_crop),
    5: ("restore_segmentation() round-trip",     example_restore),
    6: ("GPU inference (PyTorch, skip if no GPU)", example_gpu),
}


def main():
    p = argparse.ArgumentParser(
        description="sc-crop API cookbook — runnable examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Input NIfTI volume (.nii or .nii.gz)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Run all examples (default)")
    g.add_argument("--ex",  type=int, choices=list(EXAMPLES), metavar="N",
                   help=f"Run only example N ({', '.join(str(k) for k in EXAMPLES)})")
    args = p.parse_args()

    print(f"\nsc-crop API examples — {args.input}")
    for idx, (label, fn) in EXAMPLES.items():
        if args.ex and args.ex != idx:
            continue
        fn(args.input)

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
