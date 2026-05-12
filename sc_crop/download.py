"""
Model download for sc_crop.

Le modèle est stocké dans le package lui-même (sc_crop/models/), à l'intérieur
du venv d'installation. Supprimer le venv supprime aussi le modèle.

Lors de l'intégration SCT, remplacer ensure_model() par :
  from spinalcordtoolbox.utils.sys import sct_dir_local_path
  model_path = Path(sct_dir_local_path('data', 'sc_crop_models', 'model.pt'))

Usage:
    sc_crop download
    from sc_crop.download import ensure_model; model_path = ensure_model()
"""

import importlib.resources
import urllib.request
import zipfile
from pathlib import Path

# Mis à jour à chaque release.
_RELEASE_URL = (
    "https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo"
    "/releases/download/v0.2.0/sc_crop_models_v0.2.0.zip"
)


def _models_dir() -> Path:
    return Path(importlib.resources.files("sc_crop").joinpath("models"))


def ensure_model() -> Path:
    """Retourne le chemin vers model.pt, télécharge depuis la release si absent."""
    model_path = _models_dir() / "model.pt"
    if model_path.exists():
        return model_path
    download()
    return model_path


def download() -> None:
    """Télécharge et décompresse le zip de release dans sc_crop/models/."""
    models_dir = _models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    zip_path = models_dir / "_download.zip"

    print("Downloading sc_crop model from release …")
    urllib.request.urlretrieve(_RELEASE_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(models_dir)
    zip_path.unlink()

    if not (models_dir / "model.pt").exists():
        raise RuntimeError(
            f"model.pt absent de {models_dir} après téléchargement — "
            "vérifier la structure du zip de release."
        )
    print(f"Model saved to {models_dir}")
