"""
nnUNet dataset preprocessing using sc_crop detection (GPU cross-volume batch).

Replaces a manual GT-bbox crop step with automatic spinal cord detection.
Crops both image and label with the same bbox, saves in RPI orientation.

Batch inference strategy (cross-volume):
  Phase 1 (parallel CPU) : load + resample + extract 2D slices for every volume.
  Phase 2 (GPU)          : pool ALL slices across volumes → YOLO in fixed-size
                           batches → maximum GPU utilisation.
  Phase 3 (parallel CPU) : per-volume bbox aggregation, crop image + label, save.

Usage (Python API):
    from pathlib import Path
    from sc_crop.nnunet import preprocess_dataset

    preprocess_dataset(
        dataset_dir=Path("/data/nnUNet_raw/Dataset001_MyDataset"),
        output_dir=Path("/data/nnUNet_raw"),
        device="cuda",
    )

Usage (CLI):
    sc_crop preprocess-nnunet \\
        --nnunet-dir /data/nnUNet_raw/Dataset001_MyDataset \\
        --output     /data/nnUNet_raw \\
        --device cuda

    sc_crop preprocess-nnunet \\
        --input /path/to/msd_datalists/ \\
        --output /data/nnUNet_raw \\
        --taskname TempContrastAgnosticCropped --tasknumber 1000
"""

import argparse
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from tqdm import tqdm

from .crop import (
    BBox3D,
    _resolve_padding,
    aggregate_bbox_3d,
    build_slices,
    load_config,
    reorient_to_las,
    resample_for_inference,
)
from .download import ensure_cls_model, ensure_pt_model
from .infer_onnx import cls_comp_filter_onnx, load_session


# ── Volume metadata ───────────────────────────────────────────────────────────

@dataclass
class VolumeMeta:
    vol_id:    int
    image_path: Path
    label_path: Optional[Path]
    slices:    list = field(default_factory=list)
    las_idxs:  list = field(default_factory=list)
    si_zoom:   float = 0.0
    shape_las: tuple = ()
    zooms_las: tuple = ()
    las_ornt:  Optional[np.ndarray] = None
    orig_ornt: Optional[np.ndarray] = None


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def _load_and_slice(meta: VolumeMeta, config: dict) -> VolumeMeta:
    img = nib.load(meta.image_path)
    meta.orig_ornt = io_orientation(img.affine)
    img_las = reorient_to_las(img)
    meta.las_ornt  = io_orientation(img_las.affine)
    meta.zooms_las = tuple(float(v) for v in img_las.header.get_zooms()[:3])
    meta.shape_las = img_las.shape[:3]

    img_inf  = resample_for_inference(img_las, config["si_res"], config.get("inplane_res"))
    data_inf = img_inf.get_fdata(dtype=np.float32)

    meta.si_zoom = float(img_las.header.get_zooms()[2]) / config["si_res"]
    meta.slices, meta.las_idxs = build_slices(
        data_inf, config.get("channels", 3), norm_scope="volume"
    )
    return meta


def phase1_load_all(metas: list[VolumeMeta], config: dict, workers: int) -> list[VolumeMeta]:
    results: dict[int, VolumeMeta] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_load_and_slice, m, config): m.vol_id for m in metas}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Phase 1 — load+slice"):
            meta = fut.result()
            results[meta.vol_id] = meta
    return [results[m.vol_id] for m in metas]


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2_batch_infer(
    metas: list[VolumeMeta],
    det_model,
    cls_model,
    batch_size: int,
    conf: float,
    cls_conf: float,
) -> dict[int, dict]:
    """Pool all slices across volumes, send to YOLO in fixed-size batches.
    Returns raw_preds[vol_id][las_idx] = (cx, cy, w, h)."""
    tagged: list[tuple[int, int, np.ndarray]] = []
    for meta in metas:
        for las_idx, sl in zip(meta.las_idxs, meta.slices):
            tagged.append((meta.vol_id, las_idx, sl))

    raw_preds: dict[int, dict] = {m.vol_id: {} for m in metas}
    n_batches = (len(tagged) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Phase 2 — GPU inference"):
        batch   = tagged[i * batch_size : (i + 1) * batch_size]
        arrays  = [item[2] for item in batch]
        results = det_model.predict(arrays, conf=conf, verbose=False)
        for (vol_id, las_idx, _), r in zip(batch, results):
            if r.boxes is not None and len(r.boxes) > 0:
                best = int(r.boxes.conf.argmax())
                raw_preds[vol_id][las_idx] = tuple(r.boxes.xywhn[best].tolist())

    for meta in tqdm(metas, desc="Phase 2 — cls filter"):
        preds = raw_preds[meta.vol_id]
        if preds:
            raw_preds[meta.vol_id] = cls_comp_filter_onnx(
                preds, meta.slices, meta.las_idxs, cls_model, cls_conf,
            )

    return raw_preds


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def _crop_and_save(
    meta: VolumeMeta,
    preds: dict,
    out_image: Path,
    out_label: Optional[Path],
    pad_left: float,
    pad_right: float,
    pad_anterior: float,
    pad_posterior: float,
    pad_superior: float,
    pad_inferior: float,
    skip_failed: bool,
) -> bool:
    if not preds:
        msg = f"No SC detected in {meta.image_path.name}"
        if skip_failed:
            print(f"  [SKIP] {msg}")
            return False
        raise RuntimeError(msg)

    RL, AP, Z   = meta.shape_las
    bbox_las    = aggregate_bbox_3d(preds, RL, AP, Z, meta.si_zoom)
    bbox_padded = bbox_las.pad(
        pad_left, pad_right, pad_anterior, pad_posterior,
        pad_superior, pad_inferior,
        zooms=meta.zooms_las, shape=meta.shape_las,
    )

    rpi_ornt          = axcodes2ornt(("R", "P", "I"))
    bbox_rpi, _       = bbox_padded.reorient(meta.shape_las, meta.las_ornt, rpi_ornt)

    def _process(src_path, dst_path, is_label):
        img     = nib.load(src_path)
        img_rpi = img.as_reoriented(ornt_transform(meta.orig_ornt, rpi_ornt))
        cropped = bbox_rpi.crop(img_rpi)

        affine  = cropped.affine.copy()
        R, s    = affine[:3, :3], np.linalg.norm(affine[:3, :3], axis=0)
        U, _, Vt = np.linalg.svd(R / s)
        affine[:3, :3] = (U @ Vt) * s

        data = cropped.get_fdata(dtype=np.float32)
        if is_label:
            data = (data > 0.5).astype(np.uint8)
        nib.save(nib.Nifti1Image(data, affine), dst_path)

    _process(meta.image_path, out_image, is_label=False)
    if meta.label_path and out_label:
        _process(meta.label_path, out_label, is_label=True)
    return True


def phase3_crop_all(
    metas: list[VolumeMeta],
    raw_preds: dict[int, dict],
    out_images: list[Path],
    out_labels: list[Optional[Path]],
    pad_left: float,
    pad_right: float,
    pad_anterior: float,
    pad_posterior: float,
    pad_superior: float,
    pad_inferior: float,
    workers: int,
    skip_failed: bool,
) -> list[bool]:
    ok = [False] * len(metas)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                _crop_and_save,
                meta, raw_preds[meta.vol_id],
                out_images[i], out_labels[i],
                pad_left, pad_right, pad_anterior, pad_posterior,
                pad_superior, pad_inferior, skip_failed,
            ): i
            for i, meta in enumerate(metas)
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Phase 3 — crop+save"):
            ok[futs[fut]] = fut.result()
    return ok


# ── Dataset I/O helpers ───────────────────────────────────────────────────────

def collect_pairs_from_nnunet_dir(dataset_dir: Path) -> tuple[list, list]:
    """Collect (image, label) pairs from an existing nnUNet raw dataset directory."""
    def _pairs(img_dir, lbl_dir):
        if not img_dir.exists():
            return []
        pairs = []
        for img in sorted(img_dir.glob("*.nii.gz")):
            stem = img.name.replace("_0000.nii.gz", "")
            lbl  = lbl_dir / f"{stem}.nii.gz"
            pairs.append({"image": str(img), "label": str(lbl) if lbl.exists() else None})
        return pairs
    return (
        _pairs(dataset_dir / "imagesTr", dataset_dir / "labelsTr"),
        _pairs(dataset_dir / "imagesTs", dataset_dir / "labelsTs"),
    )


def collect_pairs_from_datalists(datalist_dir: Path) -> tuple[list, list]:
    """Load MSD JSON datalists (*_seed50.json) → (train+val pairs, test pairs)."""
    train_val, test = [], []
    for f in sorted(datalist_dir.glob("*_seed50.json")):
        d = json.loads(f.read_text())
        train_val += d.get("train", []) + d.get("validation", [])
        test      += d.get("test", [])
    return train_val, test


def _metas(pairs: list, start_id: int) -> list[VolumeMeta]:
    return [
        VolumeMeta(
            vol_id=start_id + i,
            image_path=Path(p["image"]),
            label_path=Path(p["label"]) if p.get("label") else None,
        )
        for i, p in enumerate(pairs)
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_dataset(
    output_dir: Path,
    *,
    dataset_dir: Optional[Path] = None,
    datalist_dir: Optional[Path] = None,
    taskname: str = "SCCropped",
    tasknumber: int = 999,
    device: str = "cuda",
    batch_size: int = 64,
    workers: Optional[int] = None,
    pad_superior: Optional[float] = None,
    pad_inferior: Optional[float] = None,
    pad_left: Optional[float] = None,
    pad_right: Optional[float] = None,
    pad_anterior: Optional[float] = None,
    pad_posterior: Optional[float] = None,
    pad_si: Optional[float] = None,
    pad_rl: Optional[float] = None,
    pad_ap: Optional[float] = None,
    conf: float = 0.1,
    cls_conf: float = 0.5,
    skip_failed: bool = False,
) -> Path:
    """Crop a nnUNet dataset using sc_crop detection. Returns the output dataset path.

    Padding (priority: individual > symmetric > default):
      Individual: pad_superior(40), pad_inferior(60), pad_left(10), pad_right(10),
                  pad_anterior(15), pad_posterior(15).
      Symmetric:  pad_si (= superior + inferior), pad_rl (= left + right),
                  pad_ap (= anterior + posterior).

    Args:
        output_dir:   Parent directory where the output dataset folder is created.
        dataset_dir:  Existing nnUNet raw dataset (imagesTr/labelsTr/imagesTs/labelsTs).
        datalist_dir: Folder of MSD JSON datalists (*_seed50.json). Mutually exclusive
                      with dataset_dir.
        taskname:     nnUNet dataset name suffix.
        tasknumber:   nnUNet dataset number.
        device:       "cuda" or "cpu" for YOLO inference.
        batch_size:   Number of 2D slices per GPU batch (cross-volume).
        workers:      CPU threads for parallel load/save (default: min(8, cpu_count)).
        conf:         YOLO detection confidence threshold.
        cls_conf:     Classifier confidence threshold.
        skip_failed:  If True, skip volumes where detection fails; else raise.

    Returns:
        Path to the created dataset directory.
    """
    if (dataset_dir is None) == (datalist_dir is None):
        raise ValueError("Provide exactly one of dataset_dir or datalist_dir.")

    pad_left_r, pad_right_r, pad_ant_r, pad_post_r, pad_sup_r, pad_inf_r = _resolve_padding(
        pad_si=pad_si, pad_superior=pad_superior, pad_inferior=pad_inferior,
        pad_rl=pad_rl, pad_left=pad_left,         pad_right=pad_right,
        pad_ap=pad_ap, pad_anterior=pad_anterior, pad_posterior=pad_posterior,
    )

    if workers is None:
        workers = min(8, os.cpu_count() or 1)

    output_dir = Path(output_dir)
    task_dir   = output_dir / f"Dataset{tasknumber:03d}_{taskname}"
    imgs_tr    = task_dir / "imagesTr"; imgs_tr.mkdir(parents=True, exist_ok=True)
    imgs_ts    = task_dir / "imagesTs"; imgs_ts.mkdir(parents=True, exist_ok=True)
    lbls_tr    = task_dir / "labelsTr"; lbls_tr.mkdir(parents=True, exist_ok=True)
    lbls_ts    = task_dir / "labelsTs"; lbls_ts.mkdir(parents=True, exist_ok=True)

    if datalist_dir is not None:
        train_val_pairs, test_pairs = collect_pairs_from_datalists(Path(datalist_dir))
    else:
        train_val_pairs, test_pairs = collect_pairs_from_nnunet_dir(Path(dataset_dir))

    print(f"Training+val : {len(train_val_pairs)} volumes")
    print(f"Test         : {len(test_pairs)} volumes")

    config    = load_config()
    from ultralytics import YOLO
    det_model = YOLO(str(ensure_pt_model()))
    det_model.to(device)
    cls_sess  = load_session(ensure_cls_model())

    print(f"Device: {device}  |  batch size: {batch_size}")

    def _run_split(pairs, img_out_dir, lbl_out_dir, split_name, id_offset):
        if not pairs:
            return []
        metas    = _metas(pairs, id_offset)
        out_imgs = [img_out_dir / f"{taskname}_{m.vol_id:03d}_0000.nii.gz" for m in metas]
        out_lbls = [lbl_out_dir / f"{taskname}_{m.vol_id:03d}.nii.gz"
                    if m.label_path else None for m in metas]

        todo = [m for m in metas if not out_imgs[m.vol_id - id_offset].exists()]
        print(f"\n{split_name}: {len(todo)}/{len(metas)} volumes to process")

        if todo:
            todo      = phase1_load_all(todo, config, workers)
            raw_preds = phase2_batch_infer(todo, det_model, cls_sess,
                                           batch_size, conf, cls_conf)
            phase3_crop_all(
                todo, raw_preds,
                [out_imgs[m.vol_id - id_offset] for m in todo],
                [out_lbls[m.vol_id - id_offset] for m in todo],
                pad_left_r, pad_right_r, pad_ant_r, pad_post_r,
                pad_sup_r, pad_inf_r, workers, skip_failed,
            )

        return [
            {"image": str(out_imgs[i]), "label": str(out_lbls[i]) if out_lbls[i] else None}
            for i in range(len(metas))
            if out_imgs[i].exists()
        ]

    tr_results = _run_split(train_val_pairs, imgs_tr, lbls_tr, "Train+val", id_offset=1)
    ts_results = _run_split(test_pairs,      imgs_ts, lbls_ts, "Test",      id_offset=1)

    dataset_json = OrderedDict({
        "name":              taskname,
        "description":       f"{taskname} — sc_crop preprocessed",
        "tensorImageSize":   "3D",
        "channel_names":     {"0": "MRI"},
        "labels":            {"background": 0, "sc": 1},
        "numTraining":       len(tr_results),
        "numTest":           len(ts_results),
        "file_ending":       ".nii.gz",
        "image_orientation": "RPI",
        "training":          tr_results,
        "test":              ts_results,
        "sc_crop_params": {
            "pad_left_mm":      pad_left_r,
            "pad_right_mm":     pad_right_r,
            "pad_anterior_mm":  pad_ant_r,
            "pad_posterior_mm": pad_post_r,
            "pad_superior_mm":  pad_sup_r,
            "pad_inferior_mm":  pad_inf_r,
            "conf":             conf,
            "device":           device,
        },
    })
    (task_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    print(f"\nDataset ready → {task_dir}")

    failed = len(train_val_pairs) + len(test_pairs) - len(tr_results) - len(ts_results)
    if failed:
        print(f"WARNING: {failed} volumes skipped (detection failed)")

    return task_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="sc_crop preprocessing for nnUNet datasets (GPU cross-volume batching)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--nnunet-dir", help="Existing nnUNet raw dataset directory")
    src.add_argument("--input",      help="Folder of MSD JSON datalists (*_seed50.json)")

    p.add_argument("--output",      required=True, help="Output parent directory (nnUNet_raw)")
    p.add_argument("--taskname",    default="SCCropped")
    p.add_argument("--tasknumber",  type=int, default=999)
    p.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size",  type=int, default=64,
                   help="Number of 2D slices per GPU batch (cross-volume)")
    p.add_argument("--workers",       type=int,   default=None,
                   help="CPU workers for parallel load/save (default: min(8, cpu_count))")
    p.add_argument("--pad-sup",  type=float, default=None, dest="pad_superior",  metavar="MM", help="Superior padding mm (default 40)")
    p.add_argument("--pad-inf",  type=float, default=None, dest="pad_inferior",  metavar="MM", help="Inferior padding mm (default 60)")
    p.add_argument("--pad-si",   type=float, default=None, dest="pad_si",        metavar="MM", help="Symmetric SI")
    p.add_argument("--pad-left", type=float, default=None, dest="pad_left",      metavar="MM", help="Left padding mm (default 10)")
    p.add_argument("--pad-right",type=float, default=None, dest="pad_right",     metavar="MM", help="Right padding mm (default 10)")
    p.add_argument("--pad-rl",   type=float, default=None, dest="pad_rl",        metavar="MM", help="Symmetric RL")
    p.add_argument("--pad-ant",  type=float, default=None, dest="pad_anterior",  metavar="MM", help="Anterior padding mm (default 15)")
    p.add_argument("--pad-post", type=float, default=None, dest="pad_posterior", metavar="MM", help="Posterior padding mm (default 15)")
    p.add_argument("--pad-ap",   type=float, default=None, dest="pad_ap",        metavar="MM", help="Symmetric AP")
    p.add_argument("--conf",          type=float, default=0.1,  help="YOLO detection confidence")
    p.add_argument("--cls-conf",      type=float, default=0.5,  help="Classifier confidence")
    p.add_argument("--skip-failed",   action="store_true",
                   help="Skip volumes where detection fails (default: raise)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    preprocess_dataset(
        output_dir    = Path(args.output),
        dataset_dir   = Path(args.nnunet_dir) if args.nnunet_dir else None,
        datalist_dir  = Path(args.input)      if args.input      else None,
        taskname      = args.taskname,
        tasknumber    = args.tasknumber,
        device        = args.device,
        batch_size    = args.batch_size,
        workers       = args.workers,
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
        cls_conf      = args.cls_conf,
        skip_failed   = args.skip_failed,
    )


if __name__ == "__main__":
    main()
