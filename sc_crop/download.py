"""
Model download for sc_crop.

Models are cached in ~/.cache/sc_crop/<model_version>/. SHA256 is verified on every load.
The model version (_MODEL_TAG) is decoupled from the package version.

To reclaim disk space from old versions:
    rm -rf ~/.cache/sc_crop/v0.0.9
"""

import hashlib
import urllib.request
from pathlib import Path

_MODEL_TAG = "v0.0.10"
_BASE_URL = f"https://github.com/ivadomed/sc-crop/releases/download/{_MODEL_TAG}"

_ASSETS = {
    "model.onnx": {
        "url": f"{_BASE_URL}/model.onnx",
        "sha256": "fb19dc819806b9ba3e0ce248d743397bf7a57914eb67057aa2cb38eff371ad7a",
    },
    "cls_model.onnx": {
        "url": f"{_BASE_URL}/cls_model.onnx",
        "sha256": "7562251b3d1ee7b6088e7549bca4e901ca52018723b732a96f9cead127559fd4",
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


def download() -> None:
    """Pre-download model files. Optional — auto on first use."""
    for name in ("model.onnx", "cls_model.onnx"):
        _ensure_file(name)
    print(f"sc_crop models ready in {_CACHE_DIR}")
