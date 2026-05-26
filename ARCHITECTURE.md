# sc_crop — Architecture & Design Decisions

## Les 3 usages fondamentaux

### Usage 1 : Inférence standalone
`from sc_crop import run` → résultat immédiat. L'approche actuelle (ONNX + auto-download) est adaptée.

### Usage 2 : Entraînement (preprocessing offline)
Le cropping par détection est **toujours offline** dans les systèmes comparables (nnDetection, MONAI, SCT). Le modèle YOLO ne tourne jamais dans la boucle d'entraînement — trop lent. Le pattern universel :

```
Phase 1 (une fois) : sc_crop détecte → sauve les bboxes et/ou images croppées
Phase 2 (chaque epoch) : charge les crops pré-calculés + perturbation aléatoire du bbox GT
```

sc_crop n'a **pas** de rôle direct dans la boucle d'entraînement. Son rôle dans l'entraînement est de **CLI batch processor**, pas de librairie importée dans le dataloader.

### Usage 3 : Intégration SCT (Spinal Cord Toolbox)
SCT a sa propre infrastructure de download (`sct_download_data`, modèles dans `$SCT_DIR/data/`). SCT voudra :
- Contrôler l'emplacement des modèles (pas `~/.cache/sc_crop/`)
- Gérer les versions indépendamment du package sc_crop
- Éviter les conflits de dépendances numpy/scipy

---

## État de l'art — outils comparables

| Outil | Interface | Modèles | Training | Inference |
|---|---|---|---|---|
| **MONAI** | `Transform` composable | Bundle auto-téléchargé | Online random crops sur GT labels | Déterministe |
| **TorchIO** | `Transform` inversible | Dans le package | Augmentation légère | Preprocessing |
| **SCT** | CLI + API Python | Dans `$SCT_DIR/`, géré par SCT | Offline (ivadomed) | sct_deepseg |
| **nnDetection** | Preprocessing offline | Résultats sauvegardés en NPZ | Crops précompués | Batch |
| **HuggingFace** | Python API | `~/.cache/huggingface/hub/{repo}/{commit_hash}/` | — | Commit hash immuable |
| **Ultralytics YOLO** | Python + CLI | `~/.config/Ultralytics/`, taille seule vérifiée | Release tag GitHub | Existence-based cache |

---

## Problèmes identifiés dans l'architecture actuelle

### Bug — Cache thrashing entre versions
`~/.cache/sc_crop/model.onnx` est partagé à plat. Deux envs avec des versions différentes de sc_crop (donc des SHA256 différents) s'écrasent mutuellement.

**Fix** : versionner le sous-dossier.
```python
_CACHE_DIR = Path.home() / ".cache" / "sc_crop" / _MODEL_TAG
# → ~/.cache/sc_crop/v0.0.3/model.onnx
```

### Intégration SCT — emplacement non configurable
Sans variable d'environnement, SCT ne peut pas rediriger les modèles vers `$SCT_DIR/data/`.

**Fix** :
```python
_CACHE_DIR = Path(os.environ.get("SC_CROP_CACHE_DIR",
    Path.home() / ".cache" / "sc_crop" / _MODEL_TAG))
```

### API — manque de fonctions publiques atomiques
Pour SCT et d'autres intégrateurs, `run()` est monolithique. Il faudrait exposer :
```python
sc_crop.detect(image) → bbox          # détection seule, sans crop
sc_crop.crop(image, bbox) → image     # crop seul, sans détection
sc_crop.run(image) → {bbox, image}    # composition des deux (actuel)
```

### Breaking change non versionnée
La suppression de l'argument `models_dir` dans `ensure_cls_model()` en v0.0.4 est un changement cassant qui aurait nécessité un bump majeur (`0.1.0`).

### Modèle .pt non distribué
ONNX uniquement dans le cache bloque le fine-tuning / ré-entraînement sur de nouvelles données. Les chercheurs qui veulent adapter le modèle ont besoin du `.pt`. À documenter explicitement.

---

## Architecture cible recommandée

Inspirée du pattern **MONAI Bundle** adapté à GitHub ivadomed :

```
sc_crop/
├── detect(image, model_path=None) → bbox     # fonction pure, modèle chargé paresseusement
├── crop(image, bbox) → image                 # fonction pure, pas de modèle
├── run(image, model_path=None) → result      # composition detect + crop
└── ModelManager
    ├── cache_dir()    # SC_CROP_CACHE_DIR ?? ~/.cache/sc_crop/{MODEL_TAG}/
    ├── ensure()       # download + SHA256 verification
    └── model_path()   # retourne le chemin résolu
```

Le `ModelManager` est instancié une fois, configurable par env var ou argument direct. `detect()` ne télécharge rien si `model_path` est fourni explicitement — c'est ce qui permet l'intégration SCT sans friction.

---

## Priorités d'implémentation

| Priorité | Changement | Justification |
|---|---|---|
| 1 (bug) | Cache versionné `~/.cache/sc_crop/v0.0.3/` | Cache thrashing entre envs |
| 2 (intégration) | `SC_CROP_CACHE_DIR` env var | Nécessaire pour SCT et HPC air-gapped |
| 3 (API) | Exposer `detect()` et `crop()` séparément | Nécessaire pour intégrateurs |
| 4 (doc) | Documenter `.pt` pour fine-tuning | Cas d'usage recherche |

---

## Décisions actées

- **ONNX pour l'inférence** : élimine les conflits torch/numpy entre pipelines
- **GitHub ivadomed releases** : tous les modèles du lab restent sur ivadomed, pas HuggingFace
- **SHA256 pincé dans le code** : équivalent du commit hash HF, sans dépendre de HF
- **_MODEL_TAG découplé de la version package** : upgrader le code ne force pas un re-download des modèles si ceux-ci n'ont pas changé
- **Auto-download sur premier appel** : pas d'étape explicite nécessaire pour les utilisateurs
