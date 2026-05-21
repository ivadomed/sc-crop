"""
Model download for sc_crop.

Models are cached in ~/.cache/sc_crop/ (shared across all conda envs and venvs on
the machine). On every load, the SHA256 of the cached file is checked against the
value pinned in this module — if it mismatches (corruption or stale version), the
file is re-downloaded automatically.

The model version (_MODEL_TAG) is intentionally decoupled from the package version:
upgrading sc_crop code does not force a model re-download unless _ASSETS changes.

Usage:
    sc_crop download           # optional pre-download; happens automatically on first use
    from sc_crop.download import ensure_model, ensure_cls_model
"""

import hashlib
import urllib.request
from pathlib import Path

_MODEL_TAG = "v0.0.3"
_BASE_URL = f"https://github.com/ivadomed/sc-crop/releases/download/{_MODEL_TAG}"

_ASSETS = {
    "model.onnx": {
        "url": f"{_BASE_URL}/model.onnx",
        "sha256": "56961ddabba0a208a97ef2cb065001726b354d6adb5c1d979010e61f12070901",
    },
    "cls_model.onnx": {
        "url": f"{_BASE_URL}/cls_model.onnx",
        "sha256": "1153922fc4c39d19d3d475f0246252d10782aef428b11e1378fc040e4f71b61e",
    },
}

_CACHE_DIR = Path.home() / ".cache" / "sc_crop"


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


def download() -> None:
    """Pre-download all model files. Optional — happens automatically on first use."""
    for name in _ASSETS:
        _ensure_file(name)
    print(f"sc_crop models ready in {_CACHE_DIR}")
