"""
Model download for sc_crop.

Models are stored in sc_crop/models/ inside the installation.
Both model.pt (detection) and cls_model.pt (classification) are downloaded
from the same release zip.

Usage:
    sc_crop download
    from sc_crop.download import ensure_model, ensure_cls_model
"""

import importlib.resources
import urllib.request
import zipfile
from pathlib import Path

# Updated on each release.
_RELEASE_URL = (
    "https://github.com/ivadomed/sc-crop"
    "/releases/download/v0.0.3/sc_crop_models_v0.0.3.zip"
)


def _models_dir() -> Path:
    return Path(importlib.resources.files("sc_crop").joinpath("models"))


def ensure_model() -> Path:
    """Return path to model.pt, downloading from release if absent."""
    model_path = _models_dir() / "model.pt"
    if model_path.exists():
        return model_path
    download()
    return model_path


def ensure_cls_model(models_dir=None) -> Path:
    """Return path to cls_model.pt, downloading from release if absent."""
    d = models_dir if models_dir is not None else _models_dir()
    cls_path = d / "cls_model.pt"
    if cls_path.exists():
        return cls_path
    download()
    return cls_path


def download() -> None:
    """Download and extract the release zip into sc_crop/models/."""
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / "_download.zip"

    print("Downloading sc_crop models from release …")
    urllib.request.urlretrieve(_RELEASE_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(models_dir)
    zip_path.unlink()

    missing = [m for m in ("model.pt", "cls_model.pt") if not (models_dir / m).exists()]
    if missing:
        raise RuntimeError(
            f"{missing} absent from {models_dir} after download — "
            "check the release zip structure."
        )
    print(f"Models saved to {models_dir}")
