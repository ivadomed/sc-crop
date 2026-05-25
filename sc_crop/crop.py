"""
Core logic for spinal cord detection and bounding box computation.

Default output: <stem>_bbox.txt with inclusive voxel indices in native image space,
compatible with SCT's ImageCropper.get_bbox_from_minmax(xmin, xmax, ymin, ymax, zmin, zmax).

High-level API for inference pipelines:
    detect_and_crop(img_path)           → (crop_nii, ctx)  — crop in memory + context
    restore_segmentation(seg_nii, ctx)  → full-space NIfTI — restore after model inference

Regularization (--regularization):
  cls       (default): runs a classifier on detection components from most superior
                       to most inferior; stops at first component with ≥1 positive slice;
                       keeps all detections from that component's min_z downward.
  graphtrim:           removes superior outlier detections by checking the 2 topmost
                       SI edges; discards slices above the first broken edge.
  none:                no regularization, raw detection output.

Normalisation (--norm-scope):
  volume (default): percentiles 0.5/99.5 computed once on all non-zero voxels of the
                    resampled volume, then applied to every slice. Preserves relative
                    intensity across slices — matches the training pipeline default.
  slice:            percentiles computed independently per slice (legacy behaviour).

Usage:
    from sc_crop.crop import run
    result = run("t2.nii.gz")                              # ONNX + cls regularization (default)
    result = run("t2.nii.gz", use_onnx=False)             # PyTorch .pt inference
    result = run("t2.nii.gz", regularization="graphtrim")  # graphtrim regularization
    result = run("t2.nii.gz", regularization="none")       # no regularization
    result = run("t2.nii.gz", norm_scope="slice")          # per-slice normalisation
    result = run("t2.nii.gz", crop=True)                   # + cropped volume (native)
    result = run("t2.nii.gz", crop=True, las=True)         # + cropped volume (LAS)
    result = run("t2.nii.gz", use_onnx=False, device="cuda")  # GPU inference (.pt only)
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

def load_config() -> dict:
    """Load config.yaml bundled with the package."""
    import importlib.resources
    import yaml
    config_path = importlib.resources.files("sc_crop").joinpath("models/config.yaml")
    return yaml.safe_load(Path(config_path).read_text())


# ─── BBox3D: single source of truth for voxel bboxes in LAS index space ──────

Pair = tuple[float, float]


def _as_pair(p, default: float = 0.0) -> Pair:
    """Normalize to (a, b). 'default' sentinel keeps that face's axis default value.

    Accepted forms:
        10.0                      → (10.0, 10.0)
        (30, 20)                  → (30.0, 20.0)
        ('default', 50)           → (default, 50.0)
        (30, 'default')           → (30.0, default)
        '30 20'                   → (30.0, 20.0)          # CLI string
        'default 50'              → (default, 50.0)       # CLI string
    """
    def _resolve(v: object) -> float:
        return default if (isinstance(v, str) and v.strip().lower() == "default") else float(v)

    if isinstance(p, (int, float)):
        return (float(p), float(p))
    if isinstance(p, str):
        parts = p.split()
        if len(parts) == 2:
            return (_resolve(parts[0]), _resolve(parts[1]))
        return (_resolve(p), _resolve(p))
    return (_resolve(p[0]), _resolve(p[1]))


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

def normalize_to_uint8(arr: np.ndarray,
                       lo: float | None = None,
                       hi: float | None = None) -> np.ndarray:
    """Normalize arr to uint8.

    If lo/hi are None, compute percentiles 0.5/99.5 from non-zero pixels (slice-level).
    Pass pre-computed lo/hi for volume-level normalisation.
    """
    if lo is None or hi is None:
        nz = arr.ravel()
        nz = nz[nz > 0]
        if not len(nz):
            return np.zeros_like(arr, dtype=np.uint8)
        lo, hi = np.percentile(nz, [0.5, 99.5])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def _volume_percentiles(data: np.ndarray) -> tuple[float, float]:
    """Compute percentiles 0.5/99.5 on all non-zero voxels of the volume."""
    nz = data.ravel()
    nz = nz[nz > 0]
    if not len(nz):
        return 0.0, 0.0
    lo, hi = np.percentile(nz, [0.5, 99.5])
    return float(lo), float(hi)


def _get_slice(data: np.ndarray, las_idx: int, black: np.ndarray,
               lo: float | None = None, hi: float | None = None) -> np.ndarray:
    """Extract one axial slice in (AP, RL) uint8, identical to preprocess.py.

    Convention: data[:, :, las_idx].T[::-1, ::-1]
      rows = AP (row 0 = Anterior), cols = RL (col 0 = Left).
    Out-of-bounds las_idx returns a black frame.
    lo/hi: pre-computed volume percentiles; None = compute per-slice.
    """
    Z = data.shape[2]
    if las_idx < 0 or las_idx >= Z:
        return black
    return normalize_to_uint8(data[:, :, las_idx], lo, hi).T[::-1, ::-1]


def _normalize_volume(data: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalize entire volume to uint8 in one vectorized pass.

    Equivalent to calling normalize_to_uint8 per-slice but ~2× faster because
    clip/scale operates on the full 3D array in one numpy call.
    """
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)
    return ((np.clip(data, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def build_slices(data: np.ndarray, channels: int,
                 norm_scope: str = "volume") -> tuple[list, list]:
    """Build all axial slices Superior→Inferior, matching preprocess.py convention.

    norm_scope: "volume" (default) — percentiles computed once, volume normalized
                in a single vectorized pass before slice extraction (~2× faster).
                "slice"            — percentiles computed independently per slice.
    3ch: R=Superior neighbour (las_idx+1), G=current, B=Inferior neighbour (las_idx-1).
    Border channels are black (zeros).

    Returns (slices, las_idxs); las_idx 0 = Inferior, Z-1 = Superior.
    """
    RL, AP, Z = data.shape
    black = np.zeros((AP, RL), dtype=np.uint8)

    if norm_scope == "volume":
        lo, hi   = _volume_percentiles(data)
        data_u8  = _normalize_volume(data, lo, hi)

        def _get(idx):
            if idx < 0 or idx >= Z:
                return black
            return data_u8[:, :, idx].T[::-1, ::-1]
    else:
        def _get(idx):
            return _get_slice(data, idx, black)  # per-slice percentiles

    slices, las_idxs = [], []
    for las_idx in range(Z - 1, -1, -1):   # Superior → Inferior
        cur = _get(las_idx)
        if channels == 3:
            slices.append(np.stack([_get(las_idx + 1), cur, _get(las_idx - 1)], axis=2))
        else:
            slices.append(cur)
        las_idxs.append(las_idx)

    return slices, las_idxs


# ─── YOLO inference ───────────────────────────────────────────────────────────

def infer_slices(model, slices: list, las_idxs: list, conf_thresh: float,
                 device: str | None = None) -> dict:
    """Run YOLO detection inference on pre-built slices.

    Returns {las_idx: (cx, cy, w, h)} in slice-image normalised coords [0,1].
    """
    kw = {"conf": conf_thresh, "verbose": False}
    if device:
        kw["device"] = device
    results = model.predict(slices, **kw)
    preds   = {}
    for las_idx, res in zip(las_idxs, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        best         = int(res.boxes.conf.argmax())
        cx, cy, w, h = res.boxes.xywhn[best].tolist()
        preds[las_idx] = (cx, cy, w, h)
    return preds


# ─── Regularization helpers ───────────────────────────────────────────────────

def _si_connected_components(preds: dict) -> list:
    """Group consecutive z-indices of preds into components (ascending z = superior first)."""
    if not preds:
        return []
    zs = sorted(preds)
    comp = [zs[0]]
    comps: list = []
    for z in zs[1:]:
        if z == comp[-1] + 1:
            comp.append(z)
        else:
            comps.append(comp)
            comp = [z]
    comps.append(comp)
    return comps


def _sc_class_idx(cls_model) -> int:
    for idx, name in cls_model.names.items():
        if name == "sc":
            return int(idx)
    raise ValueError(f"Class 'sc' not found in cls model names: {cls_model.names}")


def cls_comp_filter(preds: dict, slices: list, las_idxs: list,
                    cls_model, cls_conf: float, device: str | None) -> dict:
    """Keep first cls-validated SI component + all preds below it.

    Iterates components from most superior, runs cls in batch per component,
    stops as soon as one component has ≥1 positive slice (conf ≥ cls_conf).
    Returns all preds with z ≥ min_z of the validated component.
    Fallback: returns all preds if no component is validated.
    """
    comps = _si_connected_components(preds)
    sc_idx = _sc_class_idx(cls_model)
    slice_map = {idx: sl for idx, sl in zip(las_idxs, slices)}
    predict_kw: dict = {"verbose": False}
    if device:
        predict_kw["device"] = device

    for comp in comps:
        batch = [slice_map[z] for z in comp if z in slice_map]
        if not batch:
            continue
        results = cls_model.predict(batch, **predict_kw)
        if any(float(r.probs.data[sc_idx]) >= cls_conf for r in results):
            return {z: b for z, b in preds.items() if z >= min(comp)}
    return preds


def _graphreg_edge_broken(preds: dict, z_i: int, z_j: int,
                           H: int, W: int,
                           ap_mm: float, rl_mm: float, si_mm: float) -> bool:
    hop = z_j - z_i
    if hop * si_mm >= 40.0:
        return True
    cx_i, cy_i, w_i, h_i = preds[z_i]
    cx_j, cy_j, w_j, h_j = preds[z_j]
    return (abs((cx_j + w_j / 2) - (cx_i + w_i / 2)) * W * rl_mm > 15.0 * hop or
            abs((cx_j - w_j / 2) - (cx_i - w_i / 2)) * W * rl_mm > 15.0 * hop or
            abs((cy_j - h_j / 2) - (cy_i - h_i / 2)) * H * ap_mm > 25.0 * hop or
            abs((cy_j + h_j / 2) - (cy_i + h_i / 2)) * H * ap_mm > 25.0 * hop)


def graphtrim_superior_filter(preds: dict, H: int, W: int,
                               ap_mm: float, rl_mm: float, si_mm: float) -> dict:
    """Remove superior outlier detections by checking only the 2 topmost SI edges.

    If the edge between the 1st↔2nd or 2nd↔3rd most superior slice is broken
    (SI gap ≥ 40mm or face shift exceeds per-hop threshold), slices above the
    first broken edge are discarded.
    """
    if len(preds) <= 2:
        return preds
    zs = sorted(preds)
    cut = 0
    for i in range(min(2, len(zs) - 1)):
        if _graphreg_edge_broken(preds, zs[i], zs[i + 1], H, W, ap_mm, rl_mm, si_mm):
            cut = i + 1
    return preds if cut == 0 else {z: preds[z] for z in zs[cut:]}


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
        regularization: str | None = None,
        cls_conf: float | None = None,
        device: str | None = None,
        use_onnx: bool = True,
        norm_scope: str = "volume",
        debug: bool = False,
        crop: bool = False,
        las: bool = False,
        translate: bool = True,
        time_steps: bool = False) -> dict:
    """Full pipeline: load → LAS → resample → infer → regularize → bbox 3D → save.

    use_onnx:       True (default) — ONNX Runtime inference (CPU, no ultralytics overhead).
                    False — PyTorch .pt inference via ultralytics (supports --device cuda/mps).
    regularization: "cls" (default), "graphtrim", or "none".
    norm_scope:     "volume" (default) — percentiles from full volume (matches training pipeline).
                    "slice" — percentiles per slice independently.
    device:         "cpu", "cuda", "mps" — only used when use_onnx=False.
    """
    import time as _time

    def _tick(label: str, t0: float) -> float:
        t1 = _time.perf_counter()
        if time_steps:
            print(f"  [{label}] {t1 - t0:.2f}s")
        return t1

    from .download import ensure_model

    t0 = _time.perf_counter()

    model_path = Path(model_path) if model_path else ensure_model()
    config     = config if config is not None else load_config()
    si_res        = config["si_res"]
    inplane_res   = config.get("inplane_res")
    channels      = config.get("channels", 3)
    conf          = conf          if conf          is not None else config.get("conf", 0.1)
    regularization = regularization if regularization is not None else config.get("regularization", "cls")
    cls_conf      = cls_conf      if cls_conf      is not None else config.get("cls_conf", 0.5)

    pad_rl = _as_pair(padding_rl_mm, default=10.0)
    pad_ap = _as_pair(padding_ap_mm, default=15.0)
    pad_si = _as_pair(padding_si_mm, default=20.0)

    img              = nib.load(input_path)
    original_ornt    = nib.io_orientation(img.affine)
    original_axcodes = "".join(str(a) for a in nib.aff2axcodes(img.affine))
    img_las          = reorient_to_las(img)
    las_ornt         = axcodes2ornt(("L", "A", "S"))
    zooms            = tuple(float(v) for v in img_las.header.get_zooms()[:3])
    shape            = img_las.shape

    print(f"Input   : {Path(input_path).name}  shape={img.shape}  ornt={original_axcodes}")
    t0 = _tick("load + reorient", t0)

    if use_onnx:
        from .infer_onnx import load_session, infer_slices_onnx, cls_comp_filter_onnx
        from .download import ensure_cls_model
        det_sess = load_session(model_path)
        cls_sess = load_session(ensure_cls_model()) if regularization == "cls" else None
        if debug:
            from ultralytics import YOLO
            det_pt = YOLO(str(model_path))
    else:
        from .download import ensure_cls_model
        from ultralytics import YOLO
        predict_kw: dict = {"verbose": False}
        if device:
            predict_kw["device"] = device
        det_pt   = YOLO(str(model_path))
        cls_model = YOLO(str(ensure_cls_model())) if regularization == "cls" else None
    t0 = _tick("load model", t0)

    si_zoom  = zooms[2] / si_res
    img_inf  = resample_for_inference(img_las, si_res, inplane_res)
    data_inf = img_inf.get_fdata(dtype=np.float32)
    t0 = _tick("resample", t0)

    slices, las_idxs = build_slices(data_inf, channels, norm_scope)
    t0 = _tick("build slices", t0)

    if use_onnx:
        preds = infer_slices_onnx(det_sess, slices, las_idxs, conf)
    else:
        preds = infer_slices(det_pt, slices, las_idxs, conf, device)
    print(f"Detected: {len(preds)}/{data_inf.shape[2]} slices")
    t0 = _tick("inference", t0)

    if regularization == "cls":
        if use_onnx and cls_sess is not None:
            preds = cls_comp_filter_onnx(preds, slices, las_idxs, cls_sess, cls_conf)
        elif not use_onnx and cls_model is not None:
            preds = cls_comp_filter(preds, slices, las_idxs, cls_model, cls_conf, device)
        print(f"Cls reg : {len(preds)} slices kept (conf≥{cls_conf})")
        t0 = _tick("cls regularization", t0)
    elif regularization == "graphtrim":
        inf_zooms = tuple(float(v) for v in img_inf.header.get_zooms()[:3])
        H_inf, W_inf = data_inf.shape[1], data_inf.shape[0]
        preds = graphtrim_superior_filter(preds, H_inf, W_inf,
                                          inf_zooms[1], inf_zooms[0], inf_zooms[2])
        print(f"Graphtrim: {len(preds)} slices kept")
        t0 = _tick("graphtrim regularization", t0)

    if debug:
        parent, stem = _stem(input_path)
        bbox_pad_for_debug = None
        if preds:
            bbox_for_debug     = aggregate_bbox_3d(preds, shape[0], shape[1], shape[2], si_zoom)
            bbox_pad_for_debug = bbox_for_debug.pad(pad_rl, pad_ap, pad_si, zooms, shape)
        save_debug_panel(det_pt, slices, las_idxs, conf,
                         str(parent / f"{stem}_debug.png"),
                         padded_bbox=bbox_pad_for_debug, H=shape[1], W=shape[0])
        t0 = _tick("debug panel", t0)

    if not preds:
        raise RuntimeError("No spinal cord detected — check the volume or lower --conf")

    bbox          = aggregate_bbox_3d(preds, shape[0], shape[1], shape[2], si_zoom)
    bbox_pad      = bbox.pad(pad_rl, pad_ap, pad_si, zooms, shape)
    bbox_pad_orig, _ = bbox_pad.reorient(shape, las_ornt, original_ornt)
    t0 = _tick("bbox aggregation", t0)

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


# ─── High-level inference helpers ─────────────────────────────────────────────

def detect(img_path, **kwargs) -> dict:
    """Detect the spinal cord bounding box and return a context dict.

    This is the entry point for pipelines that need to crop multiple volumes
    (e.g. image + label) with the same bbox. Call detect() once, then pass the
    returned context to crop() for each volume.

    Args:
        img_path: Path to the input NIfTI image (any orientation, any contrast).
        **kwargs: Forwarded to run() — padding_rl_mm, padding_ap_mm, padding_si_mm,
                  conf, cls_conf, regularization, device, use_onnx, norm_scope.

    Returns:
        ctx dict containing bbox coordinates and original image, to be passed to
        crop() and restore_segmentation().

    Example::

        from sc_crop import detect, crop, restore_segmentation
        import nibabel as nib

        ctx        = detect("t2.nii.gz", padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))
        crop_img   = crop(nib.load("t2.nii.gz"),       ctx)
        crop_label = crop(nib.load("t2_label.nii.gz"), ctx)

        seg_full = restore_segmentation(my_model(crop_img), ctx)
    """
    kwargs.pop("crop", None)
    result = run(img_path, crop=False, **kwargs)
    return {**result, "_original_img": nib.load(str(img_path))}


def crop(img: "nib.Nifti1Image", ctx: dict) -> "nib.Nifti1Image":
    """Crop a NIfTI image to the bbox detected by detect().

    Works for any volume in the same space as the image passed to detect() —
    use it for both the image and its label(s).

    Args:
        img: NIfTI image to crop.
        ctx: Context dict returned by detect() or detect_and_crop().

    Returns:
        nib.Nifti1Image cropped to the detected bbox, with updated affine.

    Example::

        from sc_crop import detect, crop
        import nibabel as nib

        ctx        = detect("t2.nii.gz", padding_rl_mm=10, padding_ap_mm=15, padding_si_mm=(30, 30))
        crop_img   = crop(nib.load("t2.nii.gz"),       ctx)
        crop_label = crop(nib.load("t2_label.nii.gz"), ctx)
    """
    xmin, xmax = ctx["xmin"], ctx["xmax"]
    ymin, ymax = ctx["ymin"], ctx["ymax"]
    zmin, zmax = ctx["zmin"], ctx["zmax"]

    data   = np.asarray(img.dataobj)
    affine = img.affine.copy()
    affine[:3, 3] = (img.affine @ np.array([xmin, ymin, zmin, 1.0]))[:3]
    return nib.Nifti1Image(data[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1], affine, img.header)


# backward-compatible alias
crop_nifti = crop


def detect_and_crop(img_path, **kwargs) -> tuple:
    """Convenience wrapper: detect + crop the image in one call.

    Equivalent to::

        ctx = detect(img_path, **kwargs)
        return crop(nib.load(img_path), ctx), ctx

    Use detect() + crop() directly when you need to crop multiple volumes
    (image + label) with the same bbox.

    Returns:
        (crop_nii, ctx) where crop_nii is the cropped image and ctx is passed
        to crop() or restore_segmentation().
    """
    ctx = detect(img_path, **kwargs)
    return crop(nib.load(str(img_path)), ctx), ctx


def restore_segmentation(seg_nii, ctx) -> "nib.Nifti1Image":
    """Place a segmentation (cropped space) back into the full original image space.

    The segmentation must be in the same orientation as the original image.
    If your model reoriented the crop (e.g., to RPI), reorient the segmentation
    back before calling this function (see detect_and_crop() example).

    Args:
        seg_nii: Binary segmentation NIfTI in cropped space (original orientation).
        ctx:     Context dict returned by detect_and_crop().

    Returns:
        nib.Nifti1Image with segmentation padded to the full original image space,
        using the original affine and header.
    """
    original_img = ctx["_original_img"]
    xmin, xmax   = ctx["xmin"], ctx["xmax"]
    ymin, ymax   = ctx["ymin"], ctx["ymax"]
    zmin, zmax   = ctx["zmin"], ctx["zmax"]

    full    = np.zeros(original_img.shape[:3], dtype=np.uint8)
    seg_arr = np.asarray(seg_nii.dataobj).astype(np.uint8)
    full[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = seg_arr

    return nib.Nifti1Image(full, original_img.affine, original_img.header)