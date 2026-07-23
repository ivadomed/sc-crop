"""
Write a `crop_metadata.yaml` for a model release consumed by SpinalCordToolbox (SCT).

SCT's `sct_deepseg` reads this file from the root of a model's release .zip to know
whether/how to run the sc-crop pipeline for that model — see the padding contract
below. Include the same file (identical content) at the root of every mirror/fold zip
that make up the release; SCT merges them by path when installing.

Usage:
    python write_sct_crop_metadata.py --pad-superior 40 --pad-inferior 100 \
        --pad-left 15 --pad-right 15 --pad-anterior 15 --pad-posterior 22

Use the exact same padding values passed to detect() when building the training crops.
"""
import argparse
import yaml

import sc_crop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="crop_metadata.yaml")
    for face in ("superior", "inferior", "left", "right", "anterior", "posterior"):
        parser.add_argument(f"--pad-{face}", type=float, required=True,
                            help=f"Padding (mm) used at training time for the {face} face.")
    args = parser.parse_args()

    metadata = {
        "cropped_image": True,
        "sc_crop_version": sc_crop.__version__,
        "pad_superior": args.pad_superior,
        "pad_inferior": args.pad_inferior,
        "pad_left": args.pad_left,
        "pad_right": args.pad_right,
        "pad_anterior": args.pad_anterior,
        "pad_posterior": args.pad_posterior,
    }
    with open(args.output, "w") as fp:
        yaml.safe_dump(metadata, fp, sort_keys=False)
    print(f"Wrote {args.output} -- include it at the root of your release .zip.")


if __name__ == "__main__":
    main()
