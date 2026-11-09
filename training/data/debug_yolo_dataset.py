"""
Debug build_yolo_dataset.py on one image: run its processing (process_entry) on an image and its SC and canal
masks, then plot each slice with its SC box (green) and canal box (red).

The slices and labels are made by the same code as the YOLO dataset, so what is plotted is what YOLO sees.

Outputs (in OUTPUT_FOLDER):
    images/debug/, labels/debug/ : the png slices and YOLO label files, as written by build_yolo_dataset.py
    boxes/slice_NNN.png          : each slice with its SC (green) and canal (red) boxes
    overview.png                 : all slices in one figure
    sagittal.png                 : sagittal view at the center (RL) of the canal, with the AP extent of the SC
                                   (green) and canal (red) boxes of each slice
    debug.log                    : masks with several connected components, SC outside the canal

Usage:
    python debug_yolo_dataset.py --image <image.nii.gz> --canal-seg <canal.nii.gz> [--sc-seg <sc.nii.gz>] \
        --output-dir <output_folder> [--rgb]

Author: Pierre-Louis Benveniste
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no display needed (servers)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from build_yolo_dataset import process_entry

# Color of the box of each class (0: SC, 1: canal)
COLORS = {0: "lime", 1: "red"}
NAMES = {0: "SC", 1: "canal"}


def get_parser():
    parser = argparse.ArgumentParser(description="Plot the slices and boxes made by build_yolo_dataset.py for one image.")
    parser.add_argument("--image", required=True, type=Path, help="Image (nii.gz).")
    parser.add_argument("--canal-seg", required=True, type=Path, help="Canal segmentation (nii.gz).")
    parser.add_argument("--sc-seg", type=Path, default=None, help="SC segmentation (nii.gz). None: canal only (e.g. CT).")
    parser.add_argument("--sc-label-value", type=int, default=1, help="Value of the SC in the SC segmentation.")
    parser.add_argument("--canal-label-value", type=int, default=None,
                        help="Value of the canal in the canal segmentation (default: all non-zero voxels).")
    parser.add_argument("--ignore-affine", action="store_true",
                        help="Use the image affine for the masks (like sc_ignore_affine and canal_ignore_affine).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder where the slices and plots are saved.")
    parser.add_argument("--rgb", action="store_true", help="Pseudo-RGB slices (like build_yolo_dataset.py --rgb).")
    return parser


def read_boxes(label_path, width, height):
    """Return [(class, (x1, y1, x2, y2))] in pixels from a YOLO label file."""
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            cls, cx, cy, w, h = line.split()[:5]
            cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
            boxes.append((int(cls), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return boxes


def plot_slice(ax, png_path, label_path):
    """Plot one slice with its boxes on `ax`."""
    image = np.array(Image.open(png_path))
    # extent: pixel i covers [i, i + 1], like the box coordinates
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None, extent=(0, image.shape[1], image.shape[0], 0))
    for cls, box in read_boxes(label_path, image.shape[1], image.shape[0]):
        ax.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, color=COLORS.get(cls, "yellow"),
                               linewidth=1.5, label=NAMES.get(cls, str(cls))))
    ax.set_title(png_path.stem.rsplit("__", 1)[-1], fontsize=8)
    ax.axis("off")


def plot_sagittal(pngs, labels, title, path):
    """Save the sagittal view at the center (RL) of the canal boxes, with the AP extent of the box of each class
    on each slice. Rows = slices (superior at the top, 10 mm), columns = AP (anterior on the left, 1 mm)."""
    # Axial slices stacked from superior to inferior; pseudo-RGB pngs: G = current slice
    axial = [np.array(Image.open(png)) for png in pngs]
    axial = np.stack([a[:, :, 1] if a.ndim == 3 else a for a in axial])   # (slice, AP, RL)
    boxes = [read_boxes(label, axial.shape[2], axial.shape[1]) for label in labels]
    # RL position: median center of the canal boxes, else of all boxes, else the middle of the image
    centers = ([(b[0] + b[2]) / 2 for slice_boxes in boxes for cls, b in slice_boxes if cls == 1]
               or [(b[0] + b[2]) / 2 for slice_boxes in boxes for _, b in slice_boxes] or [axial.shape[2] / 2])
    x = min(int(np.median(centers)), axial.shape[2] - 1)

    fig, ax = plt.subplots(figsize=(6, 8))
    # extent: slice z covers [z, z + 1], AP pixel i covers [i, i + 1], like the box coordinates
    ax.imshow(axial[:, :, x], cmap="gray", extent=(0, axial.shape[1], axial.shape[0], 0), aspect=10)
    for z, slice_boxes in enumerate(boxes):
        for cls, b in slice_boxes:
            ax.add_patch(Rectangle((b[1], z), b[3] - b[1], 1, fill=False, color=COLORS.get(cls, "yellow"), linewidth=1))
    ax.legend(handles=[Patch(fill=False, edgecolor=COLORS[c], label=NAMES[c]) for c in COLORS],
              loc="lower right", fontsize=8)
    ax.set_title(f"{title}\nsagittal at RL = {x} mm, superior at the top, anterior on the left", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    args = get_parser().parse_args()
    output_dir = args.output_dir.resolve()
    for folder in ("images/debug", "labels/debug", "boxes"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    # Same entry as in the MSD json, processed by build_yolo_dataset.py
    entry = {"image": str(args.image), "dataset": "debug", "split": "debug",
             "sc_seg": str(args.sc_seg) if args.sc_seg else None, "sc_label_value": args.sc_label_value,
             "canal_seg": str(args.canal_seg), "canal_label_value": args.canal_label_value, "canal_silver": None,
             "sc_ignore_affine": args.ignore_affine, "canal_ignore_affine": args.ignore_affine}
    warnings, sc_outside = [], []
    n_slices = process_entry(entry, output_dir, args.rgb, warnings, sc_outside)

    pngs = sorted((output_dir / "images" / "debug").glob("*.png"))
    labels = [output_dir / "labels" / "debug" / f"{png.stem}.txt" for png in pngs]

    # One figure per slice
    for png, label in zip(pngs, labels):
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_slice(ax, png, label)
        if ax.patches:
            ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "boxes" / f"{png.stem.rsplit('__', 1)[-1]}.png", dpi=100)
        plt.close(fig)

    # All slices in one figure
    n_cols = min(8, len(pngs))
    n_rows = int(np.ceil(len(pngs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, png, label in zip(axes.flat, pngs, labels):
        plot_slice(ax, png, label)
    fig.suptitle(f"{args.image.name}: {n_slices} slices (from superior to inferior), SC green, canal red")
    fig.tight_layout()
    fig.savefig(output_dir / "overview.png", dpi=100)
    plt.close(fig)

    plot_sagittal(pngs, labels, args.image.name, output_dir / "sagittal.png")

    def n_with(cls):
        """Number of slices with a box of class `cls`."""
        return sum(any(line.split()[:1] == [str(cls)] for line in label.read_text().splitlines()) for label in labels)

    log = (f"Image: {args.image}\nSC segmentation: {args.sc_seg}\nCanal segmentation: {args.canal_seg}\n"
           f"Slices: {n_slices}, with SC box: {n_with(0)}, with canal box: {n_with(1)}\n"
           f"SC outside the canal: {sc_outside or 'no'}\nWarnings: {warnings or 'none'}\n")
    (output_dir / "debug.log").write_text(log)
    print(log + f"Saved in {output_dir} (overview.png, sagittal.png, boxes/)")


if __name__ == "__main__":
    main()
