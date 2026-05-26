"""
sc-crop demo — spinal cord detection and cropping.

Detects the spinal cord bounding box, prints coordinates, and optionally saves
the cropped volume. No segmentation model required — only the SC detector.
detect() is pure and writes no files; cropping and saving are done here explicitly.

Usage:
    python examples/demo.py t2.nii.gz
    python examples/demo.py t2.nii.gz --crop
    python examples/demo.py t2.nii.gz --crop --pad-sup 50 --pad-inf 80
    python examples/demo.py t2.nii.gz --crop --pad-si 30
    python examples/demo.py t2.nii.gz --crop --time

Requirements:
    pip install git+https://github.com/ivadomed/sc-crop.git
"""

import argparse
from pathlib import Path

import nibabel as nib

from sc_crop import detect, crop


def main():
    p = argparse.ArgumentParser(
        description="sc-crop demo: detect SC bbox and optionally crop the volume.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",        help="Input NIfTI volume (.nii or .nii.gz)")
    p.add_argument("--crop",       action="store_true", help="Save the cropped volume")
    p.add_argument("--pad-sup",    type=float, default=None, dest="pad_superior",
                   metavar="MM",   help="Superior padding mm (default 40)")
    p.add_argument("--pad-inf",    type=float, default=None, dest="pad_inferior",
                   metavar="MM",   help="Inferior padding mm (default 60)")
    p.add_argument("--pad-si",     type=float, default=None, dest="pad_si",
                   metavar="MM",   help="Symmetric SI")
    p.add_argument("--pad-rl",     type=float, default=None, dest="pad_rl",
                   metavar="MM",   help="Symmetric RL padding mm (default 10)")
    p.add_argument("--pad-ap",     type=float, default=None, dest="pad_ap",
                   metavar="MM",   help="Symmetric AP padding mm (default 15)")
    p.add_argument("--time",       action="store_true", help="Print elapsed time per step")
    args = p.parse_args()

    print(f"\n{'─'*60}")
    print(f"  sc-crop demo")
    print(f"{'─'*60}")

    bbox = detect(
        args.input,
        pad_superior=args.pad_superior,
        pad_inferior=args.pad_inferior,
        pad_si=args.pad_si,
        pad_rl=args.pad_rl,
        pad_ap=args.pad_ap,
        time_steps=args.time,
    )

    print(f"\n{'─'*60}")
    print(f"  Bounding box (inclusive voxel indices, native space)")
    print(f"{'─'*60}")
    print(f"  x  [{bbox['xmin']:4d} : {bbox['xmax']:4d}]  ({bbox['xmax'] - bbox['xmin'] + 1} voxels)")
    print(f"  y  [{bbox['ymin']:4d} : {bbox['ymax']:4d}]  ({bbox['ymax'] - bbox['ymin'] + 1} voxels)")
    print(f"  z  [{bbox['zmin']:4d} : {bbox['zmax']:4d}]  ({bbox['zmax'] - bbox['zmin'] + 1} voxels)")

    if args.crop:
        orig     = nib.load(args.input)
        crop_nii = crop(orig, bbox)

        inp      = Path(args.input)
        stem     = inp.name.replace(".nii.gz", "").replace(".nii", "")
        out_path = inp.parent / f"{stem}_crop.nii.gz"
        nib.save(crop_nii, out_path)

        orig_mm = [round(float(v), 2) for v in orig.header.get_zooms()[:3]]
        print(f"\n{'─'*60}")
        print(f"  Crop")
        print(f"{'─'*60}")
        print(f"  Original : {orig.shape}  {orig_mm} mm")
        print(f"  Cropped  : {crop_nii.shape}")
        print(f"  → {out_path}")

    print()


if __name__ == "__main__":
    main()
