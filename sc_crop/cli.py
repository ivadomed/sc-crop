"""
Command-line interface for sc_crop.

Three modes, selected automatically from the arguments:

  Detect (default — no coordinates):
    sc_crop -i t2.nii.gz [-o bbox.txt]
    sc_crop -i t2.nii.gz --detect [-o bbox.txt]

  Crop (coordinates provided, or --crop flag):
    sc_crop -i t2.nii.gz -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311
    sc_crop -i t2.nii.gz --crop -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311 [-o out.nii.gz]

  Detect + crop (--detect-crop flag):
    sc_crop -i t2.nii.gz --detect-crop [-o out.nii.gz]

  Download models:
    sc_crop download
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib

from .crop import detect, crop, _write_bbox_txt, _read_bbox_txt, _stem, _warn_overwrite
from .download import download

GREEN, RESET = "\033[32m", "\033[0m"


def _crop_from_coords(input_path, xmin, xmax, ymin, ymax, zmin, zmax,
                      output, translate):
    """Crop a NIfTI using explicit voxel coordinates."""
    parent, stem = _stem(input_path)
    bbox = {
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
    }
    cropped   = crop(nib.load(input_path), bbox, translate=translate)
    crop_path = Path(output) if output else parent / f"{stem}_crop.nii.gz"
    _warn_overwrite(crop_path)
    nib.save(cropped, crop_path)
    print(f"Crop    : {crop_path}  shape={cropped.shape}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
        return

    parser = argparse.ArgumentParser(
        description="Spinal cord detection and cropping.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
modes:
  detect (default)     sc_crop -i t2.nii.gz
  crop from coords     sc_crop -i t2.nii.gz -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311
  crop from bbox file  sc_crop -i t2.nii.gz --bbox t2_bbox.txt
  detect + crop        sc_crop -i t2.nii.gz --detect-crop

examples:
  sc_crop -i t2.nii.gz                                            # detect → bbox.txt
  sc_crop -i t2.nii.gz -xmin 12 -xmax 54 -ymin 72 -ymax 164 -zmin 0 -zmax 311
  sc_crop -i t2.nii.gz --bbox t2_bbox.txt                        # crop from bbox file
  sc_crop -i label.nii.gz --bbox t2_bbox.txt                     # crop label with same bbox
  sc_crop -i t2.nii.gz --detect-crop                             # detect + crop in one step
  sc_crop -i t2.nii.gz --detect-crop --pad-sup 50 --pad-inf 80
""",
    )

    # ── Input / output ────────────────────────────────────────────────────────
    parser.add_argument("input", nargs="?",
                        help="Input NIfTI volume (.nii or .nii.gz)")
    parser.add_argument("-i", dest="input_flag", default=None,
                        help="Input NIfTI volume — SCT-style alias for positional input")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: bbox.txt (detect), cropped volume (crop/detect-crop)")

    # ── Mode flags ────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detect",      action="store_true",
                      help="Detect spinal cord and write bbox.txt (default mode)")
    mode.add_argument("--crop",        action="store_true",
                      help="Crop from coordinates (requires -xmin/-xmax/-ymin/-ymax/-zmin/-zmax)")
    mode.add_argument("--detect-crop", dest="detect_crop", action="store_true",
                      help="Detect spinal cord then crop in one step")

    # ── Crop coordinates ──────────────────────────────────────────────────────
    coords = parser.add_argument_group("crop coordinates (voxel indices, inclusive)")
    coords.add_argument("--bbox", default=None, metavar="FILE",
                        help="Bbox txt file produced by sc_crop detect (alternative to -xmin/-xmax/…)")
    coords.add_argument("-xmin", type=int, default=None)
    coords.add_argument("-xmax", type=int, default=None)
    coords.add_argument("-ymin", type=int, default=None)
    coords.add_argument("-ymax", type=int, default=None)
    coords.add_argument("-zmin", type=int, default=None)
    coords.add_argument("-zmax", type=int, default=None)

    # ── Crop options ──────────────────────────────────────────────────────────
    parser.add_argument("--las", action="store_true",
                        help="Output cropped volume in LAS orientation (detect-crop only)")
    parser.add_argument("--no-translate", dest="translate", action="store_false",
                        help="Do not update affine origin after cropping")
    parser.set_defaults(translate=True)

    # ── Detection options ─────────────────────────────────────────────────────
    parser.add_argument("--model", default=None,
                        help="Path to model.pt (override sc_crop/models/)")
    pad = parser.add_argument_group("padding (mm, detect / detect-crop) — individual > symmetric > default")
    pad.add_argument("--pad-sup",  type=float, default=None, dest="pad_superior",  metavar="MM")
    pad.add_argument("--pad-inf",  type=float, default=None, dest="pad_inferior",  metavar="MM")
    pad.add_argument("--pad-si",   type=float, default=None, dest="pad_si",        metavar="MM")
    pad.add_argument("--pad-left", type=float, default=None, dest="pad_left",      metavar="MM")
    pad.add_argument("--pad-right",type=float, default=None, dest="pad_right",     metavar="MM")
    pad.add_argument("--pad-rl",   type=float, default=None, dest="pad_rl",        metavar="MM")
    pad.add_argument("--pad-ant",  type=float, default=None, dest="pad_anterior",  metavar="MM")
    pad.add_argument("--pad-post", type=float, default=None, dest="pad_posterior", metavar="MM")
    pad.add_argument("--pad-ap",   type=float, default=None, dest="pad_ap",        metavar="MM")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--regularization", default=None, choices=["cls", "graphtrim", "none"])
    parser.add_argument("--cls-conf", type=float, default=None, dest="cls_conf")
    parser.add_argument("--no-onnx", dest="use_onnx", action="store_false")
    parser.set_defaults(use_onnx=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--norm-scope", dest="norm_scope", default="volume",
                        choices=["volume", "slice"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--time",  action="store_true")

    args = parser.parse_args()
    input_path = args.input_flag or args.input
    if not input_path:
        parser.error("an input file is required (positional or -i)")

    coords_provided = any(v is not None for v in
                          [args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax])

    # ── Mode: crop from bbox file ─────────────────────────────────────────────
    if args.bbox:
        xmin, xmax, ymin, ymax, zmin, zmax = _read_bbox_txt(args.bbox)
        _crop_from_coords(input_path, xmin, xmax, ymin, ymax, zmin, zmax,
                          args.output, args.translate)
        return

    # ── Mode: crop from coordinates ───────────────────────────────────────────
    if args.crop or coords_provided:
        missing = [name for name, val in [
            ("-xmin", args.xmin), ("-xmax", args.xmax),
            ("-ymin", args.ymin), ("-ymax", args.ymax),
            ("-zmin", args.zmin), ("-zmax", args.zmax),
        ] if val is None]
        if missing:
            parser.error(f"crop mode requires all 6 coordinates or --bbox file — missing: {', '.join(missing)}")
        _crop_from_coords(input_path,
                          args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax,
                          args.output, args.translate)
        return

    # ── Detect (shared by detect and detect-crop modes) ───────────────────────
    bbox = detect(
        input_path,
        model_path    = args.model,
        pad_superior  = args.pad_superior,
        pad_inferior  = args.pad_inferior,
        pad_si        = args.pad_si,
        pad_left      = args.pad_left,
        pad_right     = args.pad_right,
        pad_rl        = args.pad_rl,
        pad_anterior  = args.pad_anterior,
        pad_posterior = args.pad_posterior,
        pad_ap        = args.pad_ap,
        conf          = args.conf,
        regularization = args.regularization,
        cls_conf       = args.cls_conf,
        device         = args.device,
        use_onnx       = args.use_onnx,
        norm_scope     = args.norm_scope,
        debug          = args.debug,
        time_steps     = args.time,
    )

    parent, stem = _stem(input_path)
    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]

    # ── Mode: detect-crop ─────────────────────────────────────────────────────
    if args.detect_crop:
        if args.las:
            cropped   = bbox["_bbox_pad_las"].crop(bbox["_img_las"], translate=args.translate)
            crop_path = Path(args.output) if args.output else parent / f"{stem}_crop_las.nii.gz"
        else:
            cropped   = crop(nib.load(input_path), bbox, translate=args.translate)
            crop_path = Path(args.output) if args.output else parent / f"{stem}_crop.nii.gz"
        _warn_overwrite(crop_path)
        nib.save(cropped, crop_path)
        print(f"Crop    : {crop_path}  shape={cropped.shape}")
        return

    # ── Mode: detect only ─────────────────────────────────────────────────────
    from .crop import BBox3D
    bbox_orig = BBox3D(
        bbox["xmin"], bbox["xmax"] + 1,
        bbox["ymin"], bbox["ymax"] + 1,
        bbox["zmin"], bbox["zmax"] + 1,
    )
    bbox_txt = Path(args.output) if args.output else parent / f"{stem}_bbox.txt"
    _write_bbox_txt(bbox_txt, bbox_orig)
    print(f"          → {bbox_txt}")

    print(f"\nTo crop with sc_crop:")
    print(f"  {GREEN}sc_crop -i {input_path} --bbox {bbox_txt}{RESET}")
    print(f"\nTo crop with SCT (if installed):")
    print(f"  {GREEN}sct_crop_image -i {input_path} -xmin {xmin} -xmax {xmax} -ymin {ymin} -ymax {ymax} -zmin {zmin} -zmax {zmax}{RESET}")


if __name__ == "__main__":
    main()
