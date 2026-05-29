"""
Model download for sc_crop.

Models are cached in ~/.cache/sc_crop/<model_version>/ so multiple versions of
sc-crop installed in different envs can coexist without interfering. On every load,
the SHA256 of the cached file is verified — if it mismatches, the file is re-downloaded.

The model version (_MODEL_TAG) is intentionally decoupled from the package version:
upgrading sc_crop code does not force a model re-download unless _ASSETS changes.

To reclaim disk space from old model versions:
    rm -rf ~/.cache/sc_crop/v0.0.5   # remove a specific old version

Usage:
    sc_crop download           # optional pre-download; happens automatically on first use
    from sc_crop.download import ensure_model, ensure_cls_model
"""

import hashlib
import urllib.request
from pathlib import Path

_MODEL_TAG = "v0.0.7"
_BASE_URL = f"https://github.com/ivadomed/sc-crop/releases/download/{_MODEL_TAG}"

_ASSETS = {
    "model.onnx": {
        "url": f"{_BASE_URL}/model.onnx",
        "sha256": "96d761eb5f047d5f31c281a8a73b2b4b7dc7f84f065406b9514f1acad9fdaf3f",
    },
    "cls_model.onnx": {
        "url": f"{_BASE_URL}/cls_model.onnx",
        "sha256": "af77f37973fa4304dbdf409c94a41605ae111bffd5242c14f03af552ca2879fc",
    },
    "model.pt": {
        "url": f"{_BASE_URL}/model.pt",
        "sha256": "ec667d683bf7766305c32346eac14c09079d522bb43d1dd925d7c613a8461417",
    },
    "cls_model.pt": {
        "url": f"{_BASE_URL}/cls_model.pt",
        "sha256": "6f2c66153904e644835fec5e71342904f11cd527ad9310a0dde54c122b1af482",
    },
}

_CACHE_DIR = Path.home() / ".cache" / "sc_crop" / _MODEL_TAG


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_file(name: str) -> Path:
    info = _ASSETS[name]
    dest = _CACHE_DIR / name
    if dest.exists() and _sha256(dest) == info["sha256"]:
        return dest
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sc_crop {name} …", flush=True)
    urllib.request.urlretrieve(info["url"], dest)
    actual = _sha256(dest)
    if actual != info["sha256"]:
        dest.unlink()
        raise RuntimeError(
            f"SHA256 mismatch for {name}.\n"
            f"  expected : {info['sha256']}\n"
            f"  got      : {actual}\n"
            "The download may be corrupted — please retry."
        )
    return dest


def ensure_model() -> Path:
    """Return path to model.onnx, downloading and verifying if needed."""
    return _ensure_file("model.onnx")


def ensure_cls_model() -> Path:
    """Return path to cls_model.onnx, downloading and verifying if needed."""
    return _ensure_file("cls_model.onnx")


def ensure_pt_model() -> Path:
    """Return path to model.pt (for GPU batch inference), downloading if needed."""
    return _ensure_file("model.pt")


def ensure_pt_cls_model() -> Path:
    """Return path to cls_model.pt, downloading if needed."""
    return _ensure_file("cls_model.pt")


def download() -> None:
    """Pre-download ONNX model files (CPU inference). Optional — auto on first use."""
    for name in ("model.onnx", "cls_model.onnx"):
        _ensure_file(name)
    print(f"sc_crop ONNX models ready in {_CACHE_DIR}")


def download_pt() -> None:
    """Pre-download PyTorch model files (GPU batch inference)."""
    for name in ("model.pt", "cls_model.pt"):
        _ensure_file(name)
    print(f"sc_crop PT models ready in {_CACHE_DIR}")
