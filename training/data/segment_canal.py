"""
Segment the spinal canal of every image of the MSD dataset (from build_msd_dataset.py) with
SCT (sct_deepseg canal, on GPU) and save the predictions in the derivatives of each dataset.

Outputs (in each dataset of the data folder):
    derivatives/pred-canal-seg/<same path as the image>/<image name>_label-canal_seg.nii.gz

Predictions that already exist are not recomputed, so the script can be stopped and run again.
Images that could not be segmented are listed in segment_canal.log in an output folder.
Images where canal segmentation is empty are listed in segment_canal_empty.log in an output folder.
A copy of the MSD json file is saved and for each image, the path of the predicted canal segmentation is added in the "canal" field.

With --part K N, only part K of the N parts of the images is segmented, so N jobs can run at the same time
on different machines/GPUs without overlap. Their logs and json are suffixed with _part<K>. Once all parts
are done, a run without --part only builds the json and logs of the whole dataset (predictions exist).

Author: Pierre-Louis Benveniste
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

OUTPUT_DIR = Path("derivatives") / "pred-canal-seg"
SUFFIX = "_label-canal_seg.nii.gz"


def get_parser():
    parser = argparse.ArgumentParser(description="Segment the spinal canal of all images of the MSD dataset.")
    parser.add_argument("--msd", required=True, type=Path, help="MSD json file built by build_msd_dataset.py.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Folder where the logs and the MSD json with the canal predictions are saved.")
    parser.add_argument("--part", type=int, nargs=2, default=(0, 1), metavar=("K", "N"),
                        help="Only segment part K of the N parts of the images (K = 0 ... N-1), to run N jobs "
                             "at the same time without overlap. Outputs are suffixed with _part<K>.")
    return parser


def get_output_path(image, dataset_root):
    """Return derivatives/pred-canal-seg/<path of the image in the dataset>/<image name>_label-canal_seg.nii.gz"""
    rel = image.relative_to(dataset_root)
    if rel.parts[0] == "derivatives":   # image from a derivatives folder: drop derivatives/<name>/
        rel = Path(*rel.parts[2:])
    return dataset_root / OUTPUT_DIR / rel.parent / rel.name.replace(".nii.gz", SUFFIX)


def main():
    args = get_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    msd = json.loads(args.msd.read_text())
    data_dir = Path(msd["data_dir"])
    splits = ("training", "validation", "test")

    # One segmentation per image (an image can have several entries)
    images = sorted({(e["dataset"], e["image"]) for split in splits for e in msd[split]})
    # Part K of N: images K, K + N, K + 2N, ... (each part has a mix of datasets)
    k, n = args.part
    images = images[k::n]
    suffix = f"_part{k}" if n > 1 else ""
    if n > 1:
        part_images = {image for _, image in images}
        for split in splits:
            msd[split] = [e for e in msd[split] if e["image"] in part_images]
    canal_of, failed, empty = {}, [], []   # {image: predicted canal segmentation}
    for i, (dataset, image) in enumerate(images, 1):
        output = get_output_path(Path(image), data_dir / dataset)
        if output.exists():
            print(f"[{i}/{len(images)}] already done: {output}")
        else:
            print(f"[{i}/{len(images)}] {image}")
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                assert os.system(f"SCT_USE_GPU=1 sct_deepseg canal -i {image} -o {output}") == 0
            except AssertionError:
                failed.append(image)
                continue
        canal_of[image] = str(output)
        if not np.asarray(nib.load(output).dataobj).any():
            empty.append(f"{image} → {output}")

    # Copy of the MSD json with the predicted canal segmentation of each image (None if it failed)
    for split in splits:
        for e in msd[split]:
            e["canal"] = canal_of.get(e["image"])
    json_path = args.output_dir / f"{args.msd.stem}_canal{suffix}.json"
    json_path.write_text(json.dumps(msd, indent=4))

    for name, images_list, title in (("segment_canal", failed, "Failed images"),
                                     ("segment_canal_empty", empty, "Images with an empty canal segmentation")):
        (args.output_dir / f"{name}{suffix}.log").write_text(
            f"MSD dataset: {args.msd.resolve()}\n"
            f"Part: {k} of {n} (images {k}, {k} + {n}, {k} + 2x{n}, ...): {len(images)} images\n"
            f"{title}: {len(images_list)}\n" + "".join(f"  {f}\n" for f in images_list))
    print(f"Done. {len(images)} images, {len(failed)} failed, {len(empty)} empty canal segmentations "
          f"(see {args.output_dir})\nSaved {json_path}")


if __name__ == "__main__":
    main()
