# sc-crop — liens de version

Ce fichier documente le lien entre la version du **package sc-crop**, la version du **modèle de détection** et le **run d'entraînement** qui l'a produit.

## Tableau des versions

| Package | Modèle  | Run d'entraînement                                   | Commit training repo                     |
|---------|---------|------------------------------------------------------|------------------------------------------|
| v0.8.0 | v0.0.10 | *(même modèle que v0.4.1 — breaking change package : retrait `segment_onnx`/`segment_pt`/`sc-crop[segment]`, ajout licence MIT, voir [CHANGELOG.md](CHANGELOG.md))* | — |
| v0.7.2 | v0.0.10 | *(fix packaging uniquement — même modèle que v0.4.1 : `project.urls`, `classifiers`, script `publish_release.sh`)* | — |
| v0.7.1 | v0.0.10 | *(fix packaging uniquement — même modèle que v0.4.1 : ajout `readme` PyPI, suppression des poids `.pt`/`.onnx` embarqués par erreur dans le sdist v0.7.0)* | — |
| v0.4.1 | v0.0.10 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
| v0.4.0 | v0.0.10 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
| v0.3.0 | v0.0.10 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
| v0.2.x | v0.0.9 | det=20260528_192341  cls=20260529_090157 | `57a25ca33cc5` |
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
| **Commit** | SHA du commit du repo **d'entraînement** (`model_cropping_sc_contrast-agnostic_yolo`) au moment de l'export du modèle |

## Deux commits différents — ne pas confondre

Chaque ligne fait intervenir **deux** dépôts, donc **deux** commits :

1. **Commit du package sc-crop** — le tag `vX.Y.Z` (colonne *Package*) **est** un commit sur `ivadomed/sc-crop`. Installer `@v0.4.1` installe exactement ce commit du code. Ce commit **n'apparaît pas** dans le tableau (c'est le tag lui-même).
2. **Commit du repo d'entraînement** (colonne *Commit*) — un commit sur l'**autre** dépôt, `model_cropping_…`, qui dit avec quel état du code d'entraînement le modèle `.onnx`/`.pt` a été exporté.

Autrement dit :

- **Oui, chaque version de package est associée à un commit** : c'est son propre tag git sur `ivadomed/sc-crop`.
- La colonne *Commit* du tableau est un commit **différent** (celui du repo d'entraînement), qui sert à reproduire le **modèle**, pas le code du package.

| Ce que tu veux retrouver | Où regarder |
|---|---|
| Le code exact d'une version de package | `git checkout vX.Y.Z` sur `ivadomed/sc-crop` |
| Le modèle (poids) d'une version | release *Modèle* (`_MODEL_TAG`) sur `ivadomed/sc-crop/releases` |
| Comment ce modèle a été entraîné/exporté | colonne *Run* + *Commit* sur `model_cropping_…` |

Note : plusieurs versions de package peuvent pointer vers le **même** modèle (ex. v0.3.0→v0.4.1 partagent le modèle v0.0.10) — seul le code du package change (preprocessing, padding par défaut), pas les poids.

## Interroger la version depuis Python

```python
import sc_crop
print(sc_crop.__version__)        # version du package, ex. "0.0.5"
print(sc_crop.__model_version__)  # version du modèle,   ex. "v0.0.3"
```

---

## Procédure de release

Deux étapes, deux dépôts — voir [MIGRATION.md](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo/blob/main/MIGRATION.md) sur le repo d'entraînement pour le détail complet.

1. Dans [model_cropping_sc_contrast-agnostic_yolo](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo) : `python scripts/export_model.py --run-dir <det> --cls-run-dir <cls> --version <MODEL_VERSION>` → produit `release_export/`, tag ce dépôt automatiquement.
2. Ici : `bash scripts/publish_release.sh --export-dir <chemin/vers/release_export> --package-version <PACKAGE_VERSION>`

`publish_release.sh` orchestre automatiquement : création de la release GitHub (poids), déploiement de `config.yaml`, mise à jour de `download.py` et de ce fichier, bump de version du package, commit + tag + push, **et publication PyPI** (`build` + `twine upload`, avec vérification qu'aucun poids n'est embarqué par erreur).
