# sc-crop — liens de version

Ce fichier documente le lien entre la version du **package sc-crop**, la version du **modèle de détection** et le **run d'entraînement** qui l'a produit.

## Tableau des versions

| Package | Modèle  | Run d'entraînement                                   | Commit training repo                     |
|---------|---------|------------------------------------------------------|------------------------------------------|
| v0.1.9 | v0.0.9 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
| v0.1.8 | v0.0.8 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
| v0.1.7 | v0.0.7 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
| v0.1.6 | v0.0.6 | *(même modèle que v0.1.5 — refactor package uniquement)* | — |
| v0.1.5 | v0.0.6 | det=20260524_224406  cls=20260525_150625 | `bbdf79afaa1d` |
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

## Procédure de release

Modifier les 4 variables en haut de `scripts/release.sh` dans [model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo), puis lancer :

```bash
bash scripts/release.sh
```

Le script orchestre automatiquement : export ONNX, calcul des SHA256, création de la release GitHub, tag des deux dépôts, mise à jour de `download.py` et de ce fichier, bump de version du package, commit + push.
