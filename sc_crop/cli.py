"""
Command-line interface for sc_crop.

Usage:
    sc_crop t2.nii.gz                        # default: writes <stem>_bbox.txt (native, inclusive)
    sc_crop t2.nii.gz --crop                 # also saves <stem>_crop.nii.gz (native orientation)
    sc_crop t2.nii.gz --crop --las           # save crop in LAS orientation
    sc_crop t2.nii.gz --crop --translate     # update affine for correct FSLeyes overlay
    sc_crop t2.nii.gz --crop -o output.nii.gz
    sc_crop t2.nii.gz --norm-scope slice     # per-slice normalisation (default: volume)
    sc_crop t2.nii.gz --debug                # also saves <stem>_debug.png
    sc_crop download                         # pre-download ONNX models
"""

import argparse
import sys
from pathlib import Path

import nibabel as nib

from .crop import detect, crop, _write_bbox_txt, _stem, _warn_overwrite
from .download import download


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
        return

    parser = argparse.ArgumentParser(
        description="Detect spinal cord and output crop indices. Optionally crop the volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
output:
  <stem>_bbox.txt  — inclusive voxel indices in native image space (xmin xmax ymin ymax zmin zmax)
  <stem>_crop.nii.gz  — cropped volume (only with --crop)

examples:
  sc_crop t2.nii.gz                                # bbox txt only
  sc_crop t2.nii.gz --crop                         # + cropped volume (native orientation)
  sc_crop t2.nii.gz --crop --las                   # + cropped volume in LAS orientation
  sc_crop t2.nii.gz --crop -o my_crop.nii.gz       # explicit output path
  sc_crop t2.nii.gz --crop --no-translate          # affine NOT updated (no FSLeyes overlay)
  sc_crop t2.nii.gz --crop --pad-sup 50 --pad-inf 80   # custom SI padding
  sc_crop t2.nii.gz --crop --pad-si 30            # symmetric SI padding
  sc_crop t2.nii.gz --crop --time                 # print elapsed time per step
""",
    )
    parser.add_argument("input", nargs="?",
                        help="Input NIfTI volume (.nii or .nii.gz)")
    parser.add_argument("-i", dest="input_flag", default=None,
                        help="Input NIfTI volume (.nii or .nii.gz) — SCT-style alias for positional input")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path: crop volume if --crop, else bbox txt")
    parser.add_argument("--model", default=None,
                        help="Path to model.pt (override sc_crop/models/)")
    parser.add_argument("--crop", action="store_true",
                        help="Save the cropped volume (default: bbox txt only)")
    parser.add_argument("--las", action="store_true",
                        help="Output cropped volume in LAS orientation (requires --crop)")
    parser.add_argument("--no-translate", dest="translate", action="store_false",
                        help="Do not update affine (by default affine is updated for correct FSLeyes overlay)")
    parser.set_defaults(translate=True)
    pad = parser.add_argument_group("padding (mm) — individual > symmetric > default")
    pad.add_argument("--pad-sup",  type=float, default=None, dest="pad_superior",
                     metavar="MM", help="Superior padding mm (default 40)")
    pad.add_argument("--pad-inf",  type=float, default=None, dest="pad_inferior",
                     metavar="MM", help="Inferior padding mm (default 60)")
    pad.add_argument("--pad-si",   type=float, default=None, dest="pad_si",
                     metavar="MM", help="Symmetric SI — overridden by --pad-sup/inf")
    pad.add_argument("--pad-left", type=float, default=None, dest="pad_left",
                     metavar="MM", help="Left padding mm (default 10)")
    pad.add_argument("--pad-right",type=float, default=None, dest="pad_right",
                     metavar="MM", help="Right padding mm (default 10)")
    pad.add_argument("--pad-rl",   type=float, default=None, dest="pad_rl",
                     metavar="MM", help="Symmetric RL — overridden by --pad-left/right")
    pad.add_argument("--pad-ant",  type=float, default=None, dest="pad_anterior",
                     metavar="MM", help="Anterior padding mm (default 15)")
    pad.add_argument("--pad-post", type=float, default=None, dest="pad_posterior",
                     metavar="MM", help="Posterior padding mm (default 15)")
    pad.add_argument("--pad-ap",   type=float, default=None, dest="pad_ap",
                     metavar="MM", help="Symmetric AP — overridden by --pad-ant/post")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default: from config.yaml)")
    parser.add_argument("--regularization", default=None, choices=["cls", "graphtrim", "none"],
                        help="Regularization method (default: from config.yaml, usually 'cls')")
    parser.add_argument("--cls-conf", type=float, default=None, dest="cls_conf",
                        help="Classification confidence threshold for --regularization cls (default: 0.5)")
    parser.add_argument("--no-onnx", dest="use_onnx", action="store_false",
                        help="Use PyTorch .pt inference instead of ONNX Runtime (supports --device)")
    parser.set_defaults(use_onnx=True)
    parser.add_argument("--device", default=None,
                        help="Inference device: cpu, cuda, mps — only with --no-onnx")
    parser.add_argument("--norm-scope", dest="norm_scope", default="volume",
                        choices=["volume", "slice"],
                        help="Normalisation scope: volume (default) computes percentiles once on "
                             "the full volume; slice computes per-slice independently")
    parser.add_argument("--debug", action="store_true",
                        help="Save <stem>_debug.png: all slices with max-confidence bbox")
    parser.add_argument("--time", action="store_true",
                        help="Print elapsed time for each pipeline step")
    args = parser.parse_args()
    input_path = args.input_flag or args.input
    if not input_path:
        parser.error("an input file is required (positional or -i)")

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

    # ── Write bbox.txt ────────────────────────────────────────────────────────
    from .crop import BBox3D
    bbox_orig = BBox3D(
        bbox["xmin"], bbox["xmax"] + 1,
        bbox["ymin"], bbox["ymax"] + 1,
        bbox["zmin"], bbox["zmax"] + 1,
    )
    bbox_txt = Path(args.output) if (args.output and not args.crop) else parent / f"{stem}_bbox.txt"
    _write_bbox_txt(bbox_txt, bbox_orig)
    print(f"          → {bbox_txt}")

    # ── Crop ──────────────────────────────────────────────────────────────────
    if args.crop:
        if args.las:
            cropped   = bbox["_bbox_pad_las"].crop(bbox["_img_las"], translate=args.translate)
            crop_path = Path(args.output) if args.output else parent / f"{stem}_crop_las.nii.gz"
        else:
            cropped   = crop(nib.load(input_path), bbox, translate=args.translate)
            crop_path = Path(args.output) if args.output else parent / f"{stem}_crop.nii.gz"
        _warn_overwrite(crop_path)
        nib.save(cropped, crop_path)
        print(f"Crop    : {crop_path}  shape={cropped.shape}")

    xmin, xmax = bbox["xmin"], bbox["xmax"]
    ymin, ymax = bbox["ymin"], bbox["ymax"]
    zmin, zmax = bbox["zmin"], bbox["zmax"]
    inp        = input_path

    GREEN, RESET = "\033[32m", "\033[0m"
    print(f"\nTo crop with SCT (if installed):")
    print(f"  {GREEN}sct_crop_image -i {inp} -xmin {xmin} -xmax {xmax} -ymin {ymin} -ymax {ymax} -zmin {zmin} -zmax {zmax}{RESET}")
    print(f"\nTo crop with sc_crop:")
    print(f"  {GREEN}sc_crop -i {inp} --crop{RESET}")


if __name__ == "__main__":
    main()
