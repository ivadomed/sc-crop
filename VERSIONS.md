# sc-crop — liens de version

Ce fichier documente le lien entre la version du **package sc-crop**, la version du **modèle de détection** et le **run d'entraînement** qui l'a produit.

## Tableau des versions

| Package | Modèle  | Run d'entraînement                                   | Commit training repo                     |
|---------|---------|------------------------------------------------------|------------------------------------------|
| v0.0.5  | v0.0.3  | `pipeline_yolo26n_axial_200ep_20260430_2319582`      | `9c984315700344391d35817579478d6e7905222d` |
| v0.0.4  | v0.0.3  | `pipeline_yolo26n_axial_200ep_20260430_2319582`      | `9c984315700344391d35817579478d6e7905222d` |
| v0.0.3  | v0.0.3  | `pipeline_yolo26n_axial_200ep_20260430_2319582`      | `9c984315700344391d35817579478d6e7905222d` |

## Lecture du tableau

| Colonne | Description |
|---------|-------------|
| **Package** | Tag GitHub sur `ivadomed/sc-crop` — version installée via `pip install git+…@vX.Y.Z` |
| **Modèle** | Tag du release hébergeant `model.onnx`, `model.pt`, `cls_model.onnx`, `cls_model.pt` sur `ivadomed/sc-crop`. Correspond à `_MODEL_TAG` dans `sc_crop/download.py`. |
| **Run** | Dossier `checkpoints/<run_id>/weights/best.pt` dans [model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo) |
| **Commit** | SHA du commit de `model_cropping_sc_contrast-agnostic_yolo` au moment de l'export |

## Interroger la version depuis Python

```python
import sc_crop
print(sc_crop.__version__)        # version du package, ex. "0.0.5"
print(sc_crop.__model_version__)  # version du modèle,   ex. "v0.0.3"
```

---

## Procédure de release (résumé)

### 1. Entraîner et exporter le modèle
```bash
cd model_cropping_sc_contrast-agnostic_yolo
python scripts/export_model.py --run-dir checkpoints/<run_id> --version X.Y
# → model.onnx, model.pt, cls_model.onnx, cls_model.pt, config.yaml
```

### 2. Créer la release GitHub du modèle sur ivadomed/sc-crop
```bash
gh release create vX.Y model.onnx model.pt cls_model.onnx cls_model.pt \
  --title "sc-crop model vX.Y" \
  --notes "YOLO26n axial 3ch si_res=10mm — run <run_id>"
```

### 3. Mettre à jour download.py
Dans `sc_crop/download.py`, mettre à jour `_MODEL_TAG` et les SHA256 correspondants :
```python
_MODEL_TAG = "vX.Y"
_ASSETS = {
    "model.onnx":     {"url": "…", "sha256": "…"},
    "cls_model.onnx": {"url": "…", "sha256": "…"},
    "model.pt":       {"url": "…", "sha256": "…"},
    "cls_model.pt":   {"url": "…", "sha256": "…"},
}
```

### 4. Bumper la version du package
Dans `pyproject.toml` : `version = "0.0.N"`

### 5. Mettre à jour VERSIONS.md
Ajouter une ligne dans le tableau ci-dessus.

### 6. Committer et tagger
```bash
git add sc_crop/download.py pyproject.toml VERSIONS.md
git commit -m "release: v0.0.N — model vX.Y (run <run_id>)"
git tag v0.0.N
git push && git push --tags
```
