"""
Core logic for spinal cord detection and bounding box computation.

Default output: <stem>_bbox.txt with inclusive voxel indices in native image space,
compatible with SCT's ImageCropper.get_bbox_from_minmax(xmin, xmax, ymin, ymax, zmin, zmax).

Usage:
    from sc_crop.crop import run
    result = run("t2.nii.gz")                          # bbox txt only
    result = run("t2.nii.gz", crop=True)               # + cropped volume (native)
    result = run("t2.nii.gz", crop=True, las=True)     # + cropped volume (LAS)
    result = run("t2.nii.gz", crop=True, translate=True)  # + correct FSLeyes affine
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import axcodes2ornt, ornt_transform
from nibabel.processing import resample_to_output
from PIL import Image as PILImage
from PIL import ImageDraw


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(model_dir: str | Path) -> dict:
    """Load config.yaml from the directory containing model.pt."""
    import yaml
    return yaml.safe_load((Path(model_dir) / "config.yaml").read_text())


# ─── BBox3D: single source of truth for voxel bboxes in LAS index space ──────

Pair = tuple[float, float]


def _as_pair(p: float | tuple) -> Pair:
    """Normalize float or (a, b) → (a, b)."""
    return (float(p), float(p)) if isinstance(p, (int, float)) else (float(p[0]), float(p[1]))


@dataclass(frozen=True)
class BBox3D:
    """Voxel bbox in LAS index space: (rl1, rl2, ap1, ap2, z1, z2)."""
    rl1: int
    rl2: int
    ap1: int
    ap2: int
    z1: int
    z2: int

    def as_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.rl1, self.rl2, self.ap1, self.ap2, self.z1, self.z2)

    def pad(self,
            pad_rl: Pair, pad_ap: Pair, pad_si: Pair,
            zooms: tuple[float, float, float],
            shape: tuple[int, int, int]) -> "BBox3D":
        """Return a new BBox3D padded in mm (per face), clamped to image bounds.

        LAS convention:
          - RL: Left (0) to Right (max)     → pad_rl = (left_mm, right_mm)
          - AP: Anterior (0) to Posterior   → pad_ap = (anterior_mm, posterior_mm)
          - SI: Superior (0) to Inferior    → pad_si = (superior_mm, inferior_mm)
        """
        rl_mm, ap_mm, si_mm = zooms
        RL, AP, Z = shape
        return BBox3D(
            rl1=max(0,  self.rl1 - int(np.ceil(pad_rl[0] / rl_mm))),
            rl2=min(RL, self.rl2 + int(np.ceil(pad_rl[1] / rl_mm))),
            ap1=max(0,  self.ap1 - int(np.ceil(pad_ap[0] / ap_mm))),
            ap2=min(AP, self.ap2 + int(np.ceil(pad_ap[1] / ap_mm))),
            z1=max(0,   self.z1  - int(np.ceil(pad_si[0] / si_mm))),
            z2=min(Z,   self.z2  + int(np.ceil(pad_si[1] / si_mm))),
        )

    def to_mm(self, img: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
        """Convert to mm space → (corner_mm, sizes_mm).

        Generic over orientation: uses img.affine and zooms[:3]. The bbox indices
        must be expressed in img's voxel orientation.
        """
        a_mm, b_mm, c_mm = [float(v) for v in img.header.get_zooms()[:3]]
        corner_mm = img.affine[:3, :3] @ np.array([self.rl1, self.ap1, self.z1]) \
                    + img.affine[:3, 3]
        sizes_mm = np.array([(self.rl2 - self.rl1) * a_mm,
                             (self.ap2 - self.ap1) * b_mm,
                             (self.z2  - self.z1)  * c_mm])
        return corner_mm, sizes_mm

    def crop(self, img: nib.Nifti1Image, translate: bool = True) -> nib.Nifti1Image:
        """Crop a NIfTI. With translate=True, updates affine so the crop sits at the
        correct world position (required for FSLeyes overlay).

        Generic over orientation: bbox indices must match img's voxel orientation.
        """
        data    = img.get_fdata(dtype=np.float32)
        cropped = data[self.rl1:self.rl2, self.ap1:self.ap2, self.z1:self.z2]
        affine  = img.affine.copy()
        if translate:
            affine[:3, 3] = img.affine[:3, :3] @ np.array([self.rl1, self.ap1, self.z1]) \
                           + img.affine[:3, 3]
        return nib.Nifti1Image(cropped, affine)

    def reorient(self,
                 src_shape: tuple[int, int, int],
                 src_ornt: np.ndarray,
                 dst_ornt: np.ndarray) -> tuple["BBox3D", tuple[int, int, int]]:
        """Reorient bbox voxel indices from src to dst orientation.

        Returns (new_bbox, new_shape). nibabel convention:
        ornt_transform(src, dst)[src_ax] = [dst_ax, flip].
        """
        src_ranges = [(self.rl1, self.rl2), (self.ap1, self.ap2), (self.z1, self.z2)]
        T = ornt_transform(src_ornt, dst_ornt)

        dst_ranges: list[tuple[int, int] | None] = [None, None, None]
        dst_shape:  list[int | None]             = [None, None, None]

        for src_ax, (dst_ax, flip) in enumerate(T):
            dst_ax = int(dst_ax)
            n      = int(src_shape[src_ax])
            lo, hi = src_ranges[src_ax]
            dst_ranges[dst_ax] = (lo, hi) if flip == 1 else (n - hi, n - lo)
            dst_shape[dst_ax]  = n

        (a1, a2), (b1, b2), (c1, c2) = dst_ranges  # type: ignore[misc]
        return BBox3D(a1, a2, b1, b2, c1, c2), tuple(dst_shape)  # type: ignore[return-value]


# ─── Orientation helpers ──────────────────────────────────────────────────────

def reorient_to_las(img: nib.Nifti1Image) -> nib.Nifti1Image:
    current = nib.io_orientation(img.affine)
    target  = axcodes2ornt(("L", "A", "S"))
    return img.as_reoriented(ornt_transform(current, target))


def reorient_to_original(img_las: nib.Nifti1Image,
                          original_ornt: np.ndarray) -> nib.Nifti1Image:
    las_ornt  = axcodes2ornt(("L", "A", "S"))
    transform = ornt_transform(las_ornt, original_ornt)
    return img_las.as_reoriented(transform)


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_for_inference(img_las: nib.Nifti1Image,
                            si_res: float,
                            inplane_res: float | None) -> nib.Nifti1Image:
    """Resample LAS image to match training preprocessing resolution (order=1)."""
    rl_mm, ap_mm, si_mm = [float(v) for v in img_las.header.get_zooms()[:3]]
    target_rl = inplane_res if inplane_res is not None else rl_mm
    target_ap = inplane_res if inplane_res is not None else ap_mm
    if (abs(target_rl - rl_mm) < 0.01
            and abs(target_ap - ap_mm) < 0.01
            and abs(si_res - si_mm) < 0.01):
        return img_las
    return resample_to_output(img_las, voxel_sizes=(target_rl, target_ap, si_res), order=1)


# ─── Slice extraction ─────────────────────────────────────────────────────────

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    flat = arr.ravel()
    nz   = flat[flat > 0]
    if not len(nz):
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(nz, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def _get_slice(data: np.ndarray, las_idx: int, black: np.ndarray) -> np.ndarray:
    """Extract one axial slice in (AP, RL) uint8, identical to preprocess.py.

    Convention: data[:, :, las_idx].T[::-1, ::-1]
      rows = AP (row 0 = Anterior), cols = RL (col 0 = Left).
    Out-of-bounds las_idx returns a black frame.
    """
    Z = data.shape[2]
    if las_idx < 0 or las_idx >= Z:
        return black
    return normalize_to_uint8(data[:, :, las_idx]).T[::-1, ::-1]


def build_slices(data: np.ndarray, channels: int) -> tuple[list, list]:
    """Build all axial slices Superior→Inferior, matching preprocess.py convention.

    3ch: R=Superior neighbour (las_idx+1), G=current, B=Inferior neighbour (las_idx-1).
    Border channels are black (zeros).

    Returns (slices, las_idxs); las_idx 0 = Inferior, Z-1 = Superior.
    """
    RL, AP, Z = data.shape
    black     = np.zeros((AP, RL), dtype=np.uint8)
    slices, las_idxs = [], []

    for las_idx in range(Z - 1, -1, -1):   # Superior → Inferior
        cur = _get_slice(data, las_idx, black)
        if channels == 3:
            sup = _get_slice(data, las_idx + 1, black)
            inf = _get_slice(data, las_idx - 1, black)
            slices.append(np.stack([sup, cur, inf], axis=2))
        else:
            slices.append(cur)
        las_idxs.append(las_idx)

    return slices, las_idxs


# ─── YOLO inference ───────────────────────────────────────────────────────────

def infer_slices(model, slices: list, las_idxs: list, conf_thresh: float) -> dict:
    """Run YOLO inference on pre-built slices.

    Returns {las_idx: (cx, cy, w, h)} in slice-image normalised coords [0,1].
    """
    results = model.predict(slices, conf=conf_thresh, verbose=False)
    preds   = {}
    for las_idx, res in zip(las_idxs, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        best         = int(res.boxes.conf.argmax())
        cx, cy, w, h = res.boxes.xywhn[best].tolist()
        preds[las_idx] = (cx, cy, w, h)
    return preds


# ─── Per-slice → LAS bbox aggregation ─────────────────────────────────────────

def aggregate_bbox_3d(preds: dict,
                      RL_nat: int, AP_nat: int, Z_nat: int,
                      si_zoom: float) -> BBox3D:
    """Aggregate per-slice detections → BBox3D in native LAS voxels.

    Slices were presented as (AP, RL) with T[::-1, ::-1], so to map back to native LAS:
      rl_c = cx * RL_nat  (no flip — kept as in original logic)
      ap_c = (1 - cy) * AP_nat
    Normalised in-plane coords are FOV-preserving → valid in native space directly.

    si_zoom = si_mm_nat / si_res; z_nat = round(las_idx_inf / si_zoom).
    """
    rl1s, rl2s, ap1s, ap2s, zs = [], [], [], [], []
    for las_idx_inf, (cx, cy, w, h) in preds.items():
        z_nat   = min(Z_nat - 1, round(las_idx_inf / si_zoom))
        rl_c    = cx * RL_nat
        ap_c    = (1.0 - cy) * AP_nat
        rl_half = w / 2 * RL_nat
        ap_half = h / 2 * AP_nat
        rl1s.append(max(0,      int(rl_c - rl_half)))
        rl2s.append(min(RL_nat, int(rl_c + rl_half)))
        ap1s.append(max(0,      int(ap_c - ap_half)))
        ap2s.append(min(AP_nat, int(ap_c + ap_half)))
        zs.append(z_nat)
    return BBox3D(min(rl1s), max(rl2s), min(ap1s), max(ap2s), min(zs), max(zs))


# ─── Debug panel ──────────────────────────────────────────────────────────────

def save_debug_panel(model, slices: list, las_idxs: list,
                     conf_thresh: float, out_path: str,
                     padded_bbox: BBox3D | None = None,
                     H: int | None = None, W: int | None = None) -> None:
    """Save a near-square panel of all axial slices with max-confidence bbox.

    Runs inference at conf=0.001 so every slice shows its best prediction.
    bbox colors:
        - Green/Orange: YOLO detection (green if conf ≥ conf_thresh, else orange)
        - Red: 3D crop region boundaries (if padded_bbox provided)

    Slice convention: data[:, :, z].T[::-1,::-1] → row 0 = Anterior, col 0 = Left.
    """
    CELL    = 128
    results = model.predict(slices, conf=0.001, verbose=False)
    has_pad = padded_bbox is not None and H is not None and W is not None
    cells   = []

    for las_idx, res, sl in zip(las_idxs, results, slices):
        rgb  = sl if sl.ndim == 3 else np.stack([sl] * 3, axis=-1)
        cell = PILImage.fromarray(rgb).resize((CELL, CELL), PILImage.BILINEAR)
        draw = ImageDraw.Draw(cell)

        # YOLO detection (green/orange)
        if res.boxes is not None and len(res.boxes) > 0:
            best         = int(res.boxes.conf.argmax())
            conf         = float(res.boxes.conf[best])
            cx, cy, w, h = res.boxes.xywhn[best].tolist()
            x1, y1 = (cx - w / 2) * CELL, (cy - h / 2) * CELL
            x2, y2 = (cx + w / 2) * CELL, (cy + h / 2) * CELL
            color = (0, 220, 0) if conf >= conf_thresh else (255, 140, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            draw.text((2, CELL - 11), f"{conf:.2f}", fill=color)

        # 3D crop region (red), drawn on slices within the padded z-range.
        # T[::-1,::-1]: col 0 = Left (LAS RL max), row 0 = Anterior (LAS AP max).
        # LAS rl_vox → image_x = (rl_vox / W) * CELL  (no flip — matches original)
        # LAS ap_vox → image_y = (1 - ap_vox / H) * CELL
        if has_pad and padded_bbox.z1 <= las_idx <= padded_bbox.z2:
            x0 = (padded_bbox.rl1 / W) * CELL
            x1 = (padded_bbox.rl2 / W) * CELL
            y0 = (1 - padded_bbox.ap2 / H) * CELL
            y1 = (1 - padded_bbox.ap1 / H) * CELL
            draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                           outline=(255, 0, 0), width=2)

        draw.text((2, 1), f"z{las_idx}", fill=(200, 200, 200))
        cells.append(cell)

    n    = len(cells)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    canvas = PILImage.new("RGB", (cols * CELL, rows * CELL), (20, 20, 20))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        canvas.paste(cell, (c * CELL, r * CELL))

    canvas.save(out_path)
    print(f"Debug   : {out_path}")


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def _stem(input_path: str) -> tuple[Path, str]:
    inp  = Path(input_path)
    stem = inp.name.replace(".nii.gz", "").replace(".nii", "")
    return inp.parent, stem


_YELLOW = "\033[33m"
_RESET  = "\033[0m"


def _warn_overwrite(path: Path) -> None:
    if path.exists():
        print(f"{_YELLOW}Warning : {path} already exists and will be overwritten{_RESET}")


def _write_bbox_txt(path: Path, bbox: BBox3D) -> None:
    """Write inclusive voxel indices compatible with SCT's ImageCropper.get_bbox_from_minmax()."""
    _warn_overwrite(path)
    with open(path, "w") as f:
        f.write("# Bounding box in native voxel space (inclusive indices)\n")
        f.write("# xmin xmax ymin ymax zmin zmax\n")
        f.write(f"{bbox.rl1} {bbox.rl2 - 1} {bbox.ap1} {bbox.ap2 - 1} {bbox.z1} {bbox.z2 - 1}\n")


# ─── Main entry point ─────────────────────────────────────────────────────────

def run(input_path: str,
        config: dict | None = None,
        output_path: str | None = None,
        model_path: str | None = None,
        padding_rl_mm: float | tuple = 10.0,
        padding_ap_mm: float | tuple = 15.0,
        padding_si_mm: float | tuple = 20.0,
        conf: float | None = None,
        debug: bool = False,
        crop: bool = False,
        las: bool = False,
        translate: bool = True,
        time_steps: bool = False) -> dict:
    """Full pipeline: load → LAS → resample → infer → bbox 3D → save.

    Default output: <stem>_bbox.txt with inclusive voxel indices in native space,
    compatible with SCT's ImageCropper.get_bbox_from_minmax().

    --crop:         also save the cropped volume (-o sets its path)
    --las:          output cropped volume in LAS orientation (requires --crop)
    --no-translate: do not update affine (by default affine is updated for FSLeyes overlay)
    """
    import time as _time

    def _tick(label: str, t0: float) -> float:
        t1 = _time.perf_counter()
        if time_steps:
            print(f"  [{label}] {t1 - t0:.2f}s")
        return t1

    from .download import ensure_model
    from ultralytics import YOLO

    t0 = _time.perf_counter()

    model_path = Path(model_path) if model_path else ensure_model()
    config     = config if config is not None else load_config(model_path.parent)
    si_res      = config["si_res"]
    inplane_res = config.get("inplane_res")
    channels    = config.get("channels", 3)
    conf        = conf if conf is not None else config.get("conf", 0.1)

    pad_rl = _as_pair(padding_rl_mm)
    pad_ap = _as_pair(padding_ap_mm)
    pad_si = _as_pair(padding_si_mm)

    img              = nib.load(input_path)
    original_ornt    = nib.io_orientation(img.affine)
    original_axcodes = "".join(str(a) for a in nib.aff2axcodes(img.affine))
    img_las          = reorient_to_las(img)
    las_ornt         = axcodes2ornt(("L", "A", "S"))
    zooms            = tuple(float(v) for v in img_las.header.get_zooms()[:3])
    shape            = img_las.shape

    print(f"Input   : {Path(input_path).name}  shape={img.shape}  ornt={original_axcodes}")
    t0 = _tick("load + reorient", t0)

    model    = YOLO(str(model_path))
    t0 = _tick("load model", t0)

    si_zoom  = zooms[2] / si_res
    img_inf  = resample_for_inference(img_las, si_res, inplane_res)
    data_inf = img_inf.get_fdata(dtype=np.float32)
    t0 = _tick("resample", t0)

    slices, las_idxs = build_slices(data_inf, channels)
    t0 = _tick("build slices", t0)

    preds = infer_slices(model, slices, las_idxs, conf)
    print(f"Detected: {len(preds)}/{data_inf.shape[2]} slices")
    t0 = _tick("inference", t0)

    if not preds:
        raise RuntimeError("No spinal cord detected — check the volume or lower --conf")

    bbox          = aggregate_bbox_3d(preds, shape[0], shape[1], shape[2], si_zoom)
    bbox_pad      = bbox.pad(pad_rl, pad_ap, pad_si, zooms, shape)
    bbox_pad_orig, _ = bbox_pad.reorient(shape, las_ornt, original_ornt)
    t0 = _tick("bbox aggregation", t0)

    if debug:
        parent, stem = _stem(input_path)
        save_debug_panel(model, slices, las_idxs, conf,
                         str(parent / f"{stem}_debug.png"),
                         padded_bbox=bbox_pad, H=shape[1], W=shape[0])
        t0 = _tick("debug panel", t0)

    parent, stem = _stem(input_path)
    bbox_txt = Path(output_path) if (output_path and not crop) else parent / f"{stem}_bbox.txt"
    _write_bbox_txt(bbox_txt, bbox_pad_orig)
    xmin, xmax = bbox_pad_orig.rl1, bbox_pad_orig.rl2 - 1
    ymin, ymax = bbox_pad_orig.ap1, bbox_pad_orig.ap2 - 1
    zmin, zmax = bbox_pad_orig.z1,  bbox_pad_orig.z2  - 1
    print(f"BBox    : xmin={xmin} xmax={xmax}  ymin={ymin} ymax={ymax}  zmin={zmin} zmax={zmax}")
    print(f"          → {bbox_txt}")

    result = {
        "bbox_file":        str(bbox_txt),
        "original_axcodes": original_axcodes,
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "zmin": zmin, "zmax": zmax,
    }

    if crop:
        if las:
            cropped   = bbox_pad.crop(img_las, translate=translate)
            crop_path = Path(output_path) if output_path else parent / f"{stem}_crop_las.nii.gz"
        else:
            img_orig  = reorient_to_original(img_las, original_ornt)
            cropped   = bbox_pad_orig.crop(img_orig, translate=translate)
            crop_path = Path(output_path) if output_path else parent / f"{stem}_crop.nii.gz"
        t0 = _tick("crop", t0)
        _warn_overwrite(crop_path)
        nib.save(cropped, crop_path)
        print(f"Crop    : {crop_path}  shape={cropped.shape}")
        t0 = _tick("save", t0)
        result["output"] = str(crop_path)

    return result