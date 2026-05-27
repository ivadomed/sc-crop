"""
sc-crop API examples — runnable cookbook.

Shows all major usage patterns for the Python API, from the simplest
to the most advanced. Each example is self-contained and can be run
independently by calling it directly.

If no image is provided, a T2 image and its SC segmentation are downloaded
automatically (~5 MB) and cached in ~/.cache/sc_crop/tutorial/.

Requirements:
    pip install "sc-crop @ git+https://github.com/ivadomed/sc-crop.git"

Usage:
    python examples/api_examples.py                        # auto-download tutorial data
    python examples/api_examples.py image.nii.gz           # use your own image
    python examples/api_examples.py image.nii.gz --ex 3   # run example 3 only
"""

import argparse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

from sc_crop import detect, crop, detect_and_crop, uncrop

_BASE_URL      = "https://github.com/ivadomed/sc-crop/releases/download/test-data"
_TUTORIAL_CACHE = Path.home() / ".cache" / "sc_crop" / "tutorial"
_TUTORIAL_IMG  = _TUTORIAL_CACHE / "t2.nii.gz"
_TUTORIAL_SEG  = _TUTORIAL_CACHE / "t2_seg.nii.gz"


def _get_tutorial_data() -> tuple[str, str]:
    """Download and cache the tutorial T2 image and SC segmentation. Returns (img_path, seg_path)."""
    _TUTORIAL_CACHE.mkdir(parents=True, exist_ok=True)
    for fname in ("t2.nii.gz", "t2_seg.nii.gz"):
        out = _TUTORIAL_CACHE / fname
        if not out.exists():
            url = f"{_BASE_URL}/{fname}"
            print(f"Downloading {fname} → {out}")
            urllib.request.urlretrieve(url, out)
    return str(_TUTORIAL_IMG), str(_TUTORIAL_SEG)


# ─── Example 1 — Basic: detect + crop ────────────────────────────────────────

def example_basic(img_path: str, _seg_path: str) -> None:
    """Detect the SC bbox, then crop the image."""
    print("\n── Example 1: Basic detect + crop ──────────────────────────────")

    bbox = detect(img_path)
    print(f"  bbox: x[{bbox['xmin']}:{bbox['xmax']}]  "
          f"y[{bbox['ymin']}:{bbox['ymax']}]  "
          f"z[{bbox['zmin']}:{bbox['zmax']}]")
    print(f"  orientation: {bbox['original_axcodes']}")

    crop_img = crop(nib.load(img_path), bbox)
    print(f"  original shape : {nib.load(img_path).shape}")
    print(f"  cropped shape  : {crop_img.shape}")

    out = Path(img_path).with_name(Path(img_path).stem.replace(".nii", "") + "_ex1_crop.nii.gz")
    nib.save(crop_img, out)
    print(f"  → {out}")


# ─── Example 2 — Multi-volume: image + label with the same bbox ──────────────

def example_multi_volume(img_path: str, seg_path: str) -> None:
    """Detect once, crop both image and label — they share the same bbox."""
    print("\n── Example 2: Multi-volume (image + label) ──────────────────────")

    img   = nib.load(img_path)
    label = nib.load(seg_path)

    bbox        = detect(img_path)
    crop_img   = crop(img,   bbox)
    crop_label = crop(label, bbox)

    assert crop_img.shape == crop_label.shape, "image and label must have the same cropped shape"
    print(f"  cropped image shape : {crop_img.shape}")
    print(f"  cropped label shape : {crop_label.shape}  (identical bbox ✓)")

    stem   = Path(img_path).name.replace(".nii.gz", "").replace(".nii", "")
    parent = Path(img_path).parent
    nib.save(crop_img,   parent / f"{stem}_ex2_crop.nii.gz")
    nib.save(crop_label, parent / f"{stem}_ex2_label_crop.nii.gz")
    print(f"  → {parent / f'{stem}_ex2_crop.nii.gz'}")
    print(f"  → {parent / f'{stem}_ex2_label_crop.nii.gz'}")


# ─── Example 3 — Padding variants ────────────────────────────────────────────

def example_padding(img_path: str, _seg_path: str) -> None:
    """Show the 3 padding styles and how priority works."""
    print("\n── Example 3: Padding variants ──────────────────────────────────")

    # Defaults: superior=40, inferior=60, left=right=10, anterior=posterior=15
    bbox_default = detect(img_path)

    # Symmetric SI (common for focused cervical datasets)
    bbox_si = detect(img_path, pad_si=30)

    # Symmetric + override one face (highest priority)
    bbox_mix = detect(img_path, pad_si=30, pad_inferior=60)

    # Full individual control (all 6 faces)
    bbox_full = detect(img_path,
                      pad_superior=40, pad_inferior=60,
                      pad_left=10,     pad_right=10,
                      pad_anterior=15, pad_posterior=15)

    def _z(bbox): return bbox["zmax"] - bbox["zmin"]
    def _x(bbox): return bbox["xmax"] - bbox["xmin"]

    print(f"  {'style':<30} {'z_range':>8}  {'x_range':>8}")
    print(f"  {'─'*50}")
    print(f"  {'default (sup=40, inf=60)':<30} {_z(bbox_default):>8}  {_x(bbox_default):>8}")
    print(f"  {'pad_si=30 (symmetric)':<30} {_z(bbox_si):>8}  {_x(bbox_si):>8}")
    print(f"  {'pad_si=30 + pad_inf=60':<30} {_z(bbox_mix):>8}  {_x(bbox_mix):>8}")
    print(f"  {'full individual':<30} {_z(bbox_full):>8}  {_x(bbox_full):>8}")

    assert _z(bbox_mix) >= _z(bbox_si), "pad_inferior=60 should give >= padding than pad_si=30"
    print("  priority rule verified: individual > symmetric ✓")


# ─── Example 4 — detect_and_crop convenience wrapper ─────────────────────────

def example_detect_and_crop(img_path: str, _seg_path: str) -> None:
    """Single-call convenience when only cropping one volume."""
    print("\n── Example 4: detect_and_crop() one-liner ───────────────────────")

    crop_nii, bbox = detect_and_crop(img_path)
    print(f"  cropped shape : {crop_nii.shape}")
    print(f"  bbox keys      : {[k for k in bbox if not k.startswith('_')]}")

    out = Path(img_path).with_name(Path(img_path).stem.replace(".nii", "") + "_ex4_crop.nii.gz")
    nib.save(crop_nii, out)
    print(f"  → {out}")


# ─── Example 5 — uncrop ────────────────────────────────────────

def example_restore(img_path: str, seg_path: str) -> None:
    """Crop the SC segmentation with the same bbox, restore it to the original space.

    Uses the real SC segmentation (t2_seg.nii.gz) to demonstrate that
    uncrop() correctly places it back at the right anatomical
    location. Open the output in FSLeyes alongside the original to verify.
    """
    print("\n── Example 5: crop label + uncrop() ──────────────")

    crop_nii, bbox = detect_and_crop(img_path)
    crop_seg       = crop(nib.load(seg_path), bbox)

    # Simulate a model output: use the cropped real segmentation as-is
    seg_full   = uncrop(crop_seg, bbox)
    orig_shape = nib.load(img_path).shape[:3]
    assert seg_full.shape == orig_shape, f"expected {orig_shape}, got {seg_full.shape}"

    n = int(np.asarray(seg_full.dataobj).sum())
    print(f"  crop shape          : {crop_nii.shape}")
    print(f"  restored seg shape  : {seg_full.shape}  ({n} non-zero voxels)")

    out = Path(img_path).with_name(Path(img_path).stem.replace(".nii", "") + "_ex5_seg.nii.gz")
    nib.save(seg_full, out)
    print(f"  → {out}")
    print(f"  verify: fsleyes {img_path} {out} -cm red")


# ─── Example 6 — GPU inference (PyTorch, no ONNX) ────────────────────────────

def example_gpu(img_path: str, _seg_path: str, device: str = "cuda") -> None:
    """Use PyTorch inference on GPU instead of the default ONNX Runtime."""
    print("\n── Example 6: GPU inference (--no-onnx) ─────────────────────────")
    print("  (requires: pip install 'sc-crop[yolo]'  and  CUDA/MPS device)")
    try:
        bbox = detect(img_path, use_onnx=False, device=device)
        print(f"  GPU ({device}) inference bbox: x[{bbox['xmin']}:{bbox['xmax']}]")
    except Exception as exc:
        print(f"  skipped — {exc}")


# ─── Runner ───────────────────────────────────────────────────────────────────

EXAMPLES = {
    1: ("Basic detect + crop",                    example_basic),
    2: ("Multi-volume: image + label",             example_multi_volume),
    3: ("Padding variants",                        example_padding),
    4: ("detect_and_crop() one-liner",             example_detect_and_crop),
    5: ("crop label + uncrop()",     example_restore),
    6: ("GPU inference (PyTorch, skip if no GPU)", example_gpu),
}


def main():
    p = argparse.ArgumentParser(
        description="sc-crop API cookbook — runnable examples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?", default=None,
                   help="Input NIfTI volume (.nii or .nii.gz) — downloads tutorial T2 if omitted")
    p.add_argument("--seg", default=None,
                   help="SC segmentation label — downloads tutorial t2_seg.nii.gz if omitted")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Run all examples (default)")
    g.add_argument("--ex",  type=int, choices=list(EXAMPLES), metavar="N",
                   help=f"Run only example N ({', '.join(str(k) for k in EXAMPLES)})")
    args = p.parse_args()

    if args.input is None:
        img_path, seg_path = _get_tutorial_data()
    else:
        img_path = args.input
        seg_path = args.seg or str(Path(img_path).with_name(
            Path(img_path).name.replace(".nii.gz", "").replace(".nii", "") + "_seg.nii.gz"
        ))

    print(f"\nsc-crop API examples — {img_path}")
    for idx, (label, fn) in EXAMPLES.items():
        if args.ex and args.ex != idx:
            continue
        fn(img_path, seg_path)

    print("\n✅  Done.")


if __name__ == "__main__":
    main()
