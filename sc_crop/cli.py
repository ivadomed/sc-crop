"""
Command-line interface for sc_crop.

Usage:
    sc_crop t2.nii.gz                        # default: writes <stem>_bbox.txt (native, inclusive)
    sc_crop t2.nii.gz --crop                 # also saves <stem>_crop.nii.gz (native orientation)
    sc_crop t2.nii.gz --crop --las           # save crop in LAS orientation
    sc_crop t2.nii.gz --crop --translate     # update affine for correct FSLeyes overlay
    sc_crop t2.nii.gz --crop -o output.nii.gz
    sc_crop t2.nii.gz --debug                # also saves <stem>_debug.png
    sc_crop download                         # download the model
"""

import argparse
import sys

from .crop import run
from .download import download


def _parse_padding(value):
    """Parse padding argument: either single float or 'left right' tuple."""
    if isinstance(value, str) and " " in value:
        parts = value.split()
        if len(parts) == 2:
            return tuple(float(p) for p in parts)
    return float(value)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
        return

    parser = argparse.ArgumentParser(
        description="Detect spinal cord and output crop indices. Optionally crop the volume.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
output (default):
  <stem>_bbox.txt  — inclusive voxel indices in native image space:
                     xmin xmax ymin ymax zmin zmax
                     compatible with SCT's ImageCropper:
                       from spinalcordtoolbox.cropping import ImageCropper
                       from spinalcordtoolbox.image import Image
                       c = ImageCropper(Image("t2.nii.gz"))
                       c.get_bbox_from_minmax(xmin, xmax, ymin, ymax, zmin, zmax)
                       img_crop = c.crop()

examples:
  sc_crop t2.nii.gz                          # bbox txt only (native space)
  sc_crop t2.nii.gz --crop                   # + t2_crop.nii.gz (native)
  sc_crop t2.nii.gz --crop --las             # + t2_crop_las.nii.gz
  sc_crop t2.nii.gz --crop                   # + t2_crop.nii.gz (affine updated by default)
  sc_crop t2.nii.gz --crop --no-translate    # affine NOT updated
  sc_crop t2.nii.gz --crop --las             # LAS crop with correct affine
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
    parser.add_argument("--padding-rl", type=str, default="10.0",
                        help="Padding in Right-Left direction (mm). Single value or 'left right'")
    parser.add_argument("--padding-ap", type=str, default="15.0",
                        help="Padding in Anterior-Posterior direction (mm). Single value or 'ant post'")
    parser.add_argument("--padding-si", type=str, default="20.0",
                        help="Padding in Superior-Inferior direction (mm). Single value or 'sup inf'")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default: from config.yaml)")
    parser.add_argument("--debug", action="store_true",
                        help="Save <stem>_debug.png: all slices with max-confidence bbox")
    parser.add_argument("--time", action="store_true",
                        help="Print elapsed time for each pipeline step")
    args = parser.parse_args()
    input_path = args.input_flag or args.input
    if not input_path:
        parser.error("an input file is required (positional or -i)")

    result = run(
        input_path    = input_path,
        output_path   = args.output,
        model_path    = args.model,
        padding_rl_mm = _parse_padding(args.padding_rl),
        padding_ap_mm = _parse_padding(args.padding_ap),
        padding_si_mm = _parse_padding(args.padding_si),
        conf          = args.conf,
        debug         = args.debug,
        crop          = args.crop,
        las           = args.las,
        translate     = args.translate,
        time_steps    = args.time,
    )

    if "output" in result:
        print(f"Crop    : {result['output']}")

    xmin, xmax = result["xmin"], result["xmax"]
    ymin, ymax = result["ymin"], result["ymax"]
    zmin, zmax = result["zmin"], result["zmax"]
    inp        = input_path

    GREEN, RESET = "\033[32m", "\033[0m"
    print(f"\nTo crop with SCT (if installed):")
    print(f"  {GREEN}sct_crop_image -i {inp} -xmin {xmin} -xmax {xmax} -ymin {ymin} -ymax {ymax} -zmin {zmin} -zmax {zmax}{RESET}")
    print(f"\nTo crop with sc_crop:")
    print(f"  {GREEN}sc_crop -i {inp} --crop{RESET}")


if __name__ == "__main__":
    main()
