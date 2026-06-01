"""
Quality-control helpers for sc_crop detection-based cropping.

Public API:
    check_label_crop(label, bbox)  → dict  — check that no SC voxels are lost after crop
    CropReport                             — accumulate and save per-volume QC results
"""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from .crop import crop


def check_label_crop(label: nib.Nifti1Image, bbox: dict) -> dict:
    """Check that crop() preserves all non-zero label voxels.

    A crop is valid only if every SC voxel in the original label is still present
    after cropping. Any loss means the detection bbox cut through the spinal cord.

    Args:
        label: original (uncropped) label NIfTI, as returned by nib.load()
        bbox:  context dict returned by detect()

    Returns:
        dict with keys:
            voxels_before  — non-zero voxel count in the original label
            voxels_after   — non-zero voxel count in the cropped label
            ok             — True iff voxels_after == voxels_before

    Example::

        from sc_crop import detect, crop, check_label_crop
        import nibabel as nib

        bbox  = detect("t2.nii.gz")
        qc    = check_label_crop(nib.load("t2_seg.nii.gz"), bbox)
        if not qc["ok"]:
            print(f"Bad crop: lost {qc['voxels_before'] - qc['voxels_after']} voxels")
        crop_label = crop(nib.load("t2_seg.nii.gz"), bbox)
    """
    voxels_before = int(np.count_nonzero(np.asarray(label.dataobj)))
    voxels_after  = int(np.count_nonzero(np.asarray(crop(label, bbox).dataobj)))
    return {
        "voxels_before": voxels_before,
        "voxels_after":  voxels_after,
        "ok":            voxels_after == voxels_before,
    }


class CropReport:
    """Accumulate per-volume crop QC results during preprocessing and save to CSV.

    Initialise once at the start of the preprocessing pipeline, call add() for
    each processed volume, then save() at the end.

    Example::

        from sc_crop import detect, crop, check_label_crop, CropReport
        import nibabel as nib

        report = CropReport()

        for item in data:
            bbox = detect(item["image"])
            qc   = check_label_crop(nib.load(item["label"]), bbox)
            report.add(item["label"], qc)
            crop_label = crop(nib.load(item["label"]), bbox)

        report.save("crop_qc_report.csv")
        print(f"{report.n_failed()} bad crops / {len(report)}")
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def add(self, label_path: "str | Path", qc: dict) -> None:
        """Append one QC result. qc must be the dict returned by check_label_crop()."""
        self._entries.append({"label": str(label_path), **qc})

    def save(self, path: "str | Path") -> None:
        """Write all accumulated results to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "voxels_before", "voxels_after", "ok"])
            writer.writeheader()
            writer.writerows(self._entries)

    def save_summary(self, path: "str | Path") -> None:
        """Write a JSON summary: total/ok/failed counts and list of failed labels."""
        import json
        failed = [e for e in self._entries if not e["ok"]]
        summary = {
            "total":  len(self),
            "ok":     len(self) - len(failed),
            "failed": len(failed),
            "failed_labels": [
                {"label": e["label"], "voxels_before": e["voxels_before"], "voxels_after": e["voxels_after"]}
                for e in failed
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Crop QC : {summary['ok']}/{summary['total']} ok — {summary['failed']} failed — {path}")

    def n_failed(self) -> int:
        """Number of volumes where the crop lost at least one SC voxel."""
        return sum(1 for e in self._entries if not e["ok"])

    def __len__(self) -> int:
        return len(self._entries)
