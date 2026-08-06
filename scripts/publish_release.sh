#!/bin/bash
# ============================================================
# sc-crop release — GitHub release + PyPI publish
#
# Input: a release_export/ directory produced by export_model.py in the
# training repo (model_cropping_sc_contrast-agnostic_yolo) — model.pt,
# model.onnx, cls_model.pt, cls_model.onnx, config.yaml, sha256.yaml.
# This script only touches this repo (sc-crop) — it never reads anything
# else from the training repo (no runs/, no checkpoints).
#
# Usage:
#   bash scripts/publish_release.sh --export-dir <path> --package-version X.Y.Z
#   bash scripts/publish_release.sh --export-dir <path> --package-version X.Y.Z --skip-pypi
# ============================================================
set -euo pipefail

SCCROP_REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(command -v python)"

EXPORT_DIR=""
PACKAGE_VERSION=""
SKIP_PYPI=false
while [ $# -gt 0 ]; do
    case "$1" in
        --export-dir)      EXPORT_DIR="$2"; shift 2 ;;
        --package-version) PACKAGE_VERSION="$2"; shift 2 ;;
        --skip-pypi)        SKIP_PYPI=true; shift ;;
        *) echo "ERREUR : argument inconnu '$1'"; exit 1 ;;
    esac
done

[ -n "$EXPORT_DIR" ]      || { echo "ERREUR : --export-dir requis (dossier release_export/ produit par export_model.py)."; exit 1; }
[ -n "$PACKAGE_VERSION" ] || { echo "ERREUR : --package-version requis, ex. 0.1.10"; exit 1; }
EXPORT_DIR="$(cd "$EXPORT_DIR" && pwd)"

SHA256_YAML="${EXPORT_DIR}/sha256.yaml"
CONFIG_SRC="${EXPORT_DIR}/config.yaml"
for f in "$SHA256_YAML" "$CONFIG_SRC" "${EXPORT_DIR}/model.pt" "${EXPORT_DIR}/model.onnx" "${EXPORT_DIR}/cls_model.pt" "${EXPORT_DIR}/cls_model.onnx"; do
    [ -f "$f" ] || { echo "ERREUR : $f introuvable — relance export_model.py côté training repo."; exit 1; }
done

command -v gh >/dev/null || { echo "ERREUR : gh (GitHub CLI) introuvable dans le PATH."; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "ERREUR : gh non authentifié — lance 'gh auth login'."; exit 1; }

if [ "$SKIP_PYPI" = false ]; then
    [ -f "$HOME/.pypirc" ] || { echo "ERREUR : ~/.pypirc introuvable — nécessaire pour 'twine upload'. Crée-le (section [pypi], username=__token__, password=<ton token PyPI>) ou relance avec --skip-pypi."; exit 1; }
    grep -q '^\[pypi\]' "$HOME/.pypirc" || { echo "ERREUR : ~/.pypirc n'a pas de section [pypi]."; exit 1; }
    "$PYTHON" -c "import build" 2>/dev/null || { echo "ERREUR : module Python 'build' introuvable — pip install build"; exit 1; }
    command -v twine >/dev/null || { echo "ERREUR : twine introuvable dans le PATH — pip install twine"; exit 1; }
fi

MODEL_VERSION=$(     "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['version'])")
DET_RUN=$(            "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['det_run'])")
CLS_RUN=$(            "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['cls_run'])")
EXPORT_GIT_HASH=$(    "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['export_git_hash'])")
SHA_MODEL_ONNX=$(     "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['assets']['model.onnx'])")
SHA_MODEL_PT=$(       "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['assets']['model.pt'])")
SHA_CLS_ONNX=$(       "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['assets']['cls_model.onnx'])")
SHA_CLS_PT=$(         "$PYTHON" -c "import yaml; print(yaml.safe_load(open('${SHA256_YAML}'))['assets']['cls_model.pt'])")

# Garde-fou anti-dérive : la version doit être strictement supérieure à la dernière publiée
# dans VERSIONS.md (déjà pris en défaut une fois — brouillon local en retard sur PyPI).
LAST_PUBLISHED=$("$PYTHON" -c "
import re
text = open('${SCCROP_REPO}/VERSIONS.md').read()
rows = re.findall(r'^\| v([\d.]+)\s*\| v([\d.]+)\s*\|', text, re.MULTILINE)
pkg, model = rows[0]
print(f'{pkg} {model}')
")
LAST_PACKAGE_VERSION="${LAST_PUBLISHED%% *}"
LAST_MODEL_VERSION="${LAST_PUBLISHED##* }"

version_gt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]; }

if ! version_gt "${PACKAGE_VERSION}" "${LAST_PACKAGE_VERSION}"; then
    echo "ERREUR : --package-version=${PACKAGE_VERSION} n'est pas > dernière version publiée v${LAST_PACKAGE_VERSION} (VERSIONS.md)."
    exit 1
fi
if ! version_gt "${MODEL_VERSION}" "${LAST_MODEL_VERSION}"; then
    echo "ERREUR : version modèle (sha256.yaml) =${MODEL_VERSION} n'est pas > dernière version publiée v${LAST_MODEL_VERSION} (VERSIONS.md). Modèle déjà publié ?"
    exit 1
fi

GRN="\033[32m"; YLW="\033[33m"; BLD="\033[1m"; RST="\033[0m"
step() { echo -e "\n${BLD}${GRN}▶ $*${RST}"; }
info() { echo -e "  ${YLW}$*${RST}"; }


# ════════════════════════════════════════════════════════════
# PHASE 1 — RELEASE GITHUB (poids du modèle)
# ════════════════════════════════════════════════════════════
step "Phase 1 — Création de la release GitHub (modèle v${MODEL_VERSION})"

gh release create "v${MODEL_VERSION}" \
    "${EXPORT_DIR}/model.pt"       \
    "${EXPORT_DIR}/model.onnx"     \
    "${EXPORT_DIR}/cls_model.pt"   \
    "${EXPORT_DIR}/cls_model.onnx" \
    --repo ivadomed/sc-crop \
    --title "sc-crop model v${MODEL_VERSION}" \
    --notes "det=${DET_RUN}  cls=${CLS_RUN}  export_commit=${EXPORT_GIT_HASH:0:12}"

info "Release v${MODEL_VERSION} créée sur ivadomed/sc-crop"


# ════════════════════════════════════════════════════════════
# PHASE 2 — DÉPLOIEMENT DE config.yaml
# ════════════════════════════════════════════════════════════
step "Phase 2 — Déploiement de config.yaml dans sc_crop/"

CONFIG_DST="${SCCROP_REPO}/sc_crop/config.yaml"

# Keep only inference-relevant keys (strip training-side traceability fields)
"$PYTHON" - <<PYEOF
import yaml
from pathlib import Path

src = yaml.safe_load(Path("${CONFIG_SRC}").read_text())

inference_keys = ["si_res", "inplane_res", "channels", "norm_scope",
                  "imgsz", "conf", "regularization", "cls_conf"]
comments = {
    "si_res":        "SI resampling resolution (mm) — must match training",
    "inplane_res":   "In-plane resampling resolution (mm) — must match training",
    "channels":      "Input channels (3 = superior/current/inferior)",
    "norm_scope":    "Normalisation scope — must match training: volume | slice_all | slice",
    "imgsz":         "YOLO inference imgsz — must match training; required for ONNX (no metadata)",
    "conf":          "Detection confidence threshold",
    "regularization":"cls | graphtrim | none",
    "cls_conf":      "Classification confidence threshold (used with regularization=cls)",
}

lines = []
for k in inference_keys:
    if k not in src:
        continue
    lines.append(f"{k}: {src[k]!r:<14}  # {comments.get(k, '')}")

Path("${CONFIG_DST}").write_text("\n".join(lines) + "\n")
print("  config.yaml déployé :", "${CONFIG_DST}")
PYEOF

info "config.yaml → sc_crop/config.yaml"


# ════════════════════════════════════════════════════════════
# PHASE 3 — MISE À JOUR DE sc_crop/download.py
# ════════════════════════════════════════════════════════════
step "Phase 3 — Mise à jour de sc_crop/download.py"

"$PYTHON" - <<PYEOF
import re, textwrap
from pathlib import Path

path = Path("${SCCROP_REPO}/sc_crop/download.py")
src  = path.read_text()

new_assets = textwrap.dedent("""
    _MODEL_TAG = "v${MODEL_VERSION}"
    _BASE_URL = f"https://github.com/ivadomed/sc-crop/releases/download/{_MODEL_TAG}"

    _ASSETS = {
        "model.onnx": {
            "url": f"{_BASE_URL}/model.onnx",
            "sha256": "${SHA_MODEL_ONNX}",
        },
        "cls_model.onnx": {
            "url": f"{_BASE_URL}/cls_model.onnx",
            "sha256": "${SHA_CLS_ONNX}",
        },
        "model.pt": {
            "url": f"{_BASE_URL}/model.pt",
            "sha256": "${SHA_MODEL_PT}",
        },
        "cls_model.pt": {
            "url": f"{_BASE_URL}/cls_model.pt",
            "sha256": "${SHA_CLS_PT}",
        },
    }
""").strip()

src = re.sub(
    r'_MODEL_TAG\s*=.*?(?=\n_CACHE_DIR)',
    new_assets + "\n",
    src,
    flags=re.DOTALL,
)
path.write_text(src)
print("  download.py mis à jour")
PYEOF


# ════════════════════════════════════════════════════════════
# PHASE 4 — MISE À JOUR DE VERSIONS.md
# ════════════════════════════════════════════════════════════
step "Phase 4 — Mise à jour de VERSIONS.md"

NEW_ROW="| v${PACKAGE_VERSION} | v${MODEL_VERSION} | det=${DET_RUN}  cls=${CLS_RUN} | \`${EXPORT_GIT_HASH:0:12}\` |"

"$PYTHON" - <<PYEOF
from pathlib import Path

path = Path("${SCCROP_REPO}/VERSIONS.md")
lines = path.read_text().splitlines()
insert_after = next(i for i, l in enumerate(lines) if l.startswith("|---"))
lines.insert(insert_after + 1, "${NEW_ROW}")
path.write_text("\n".join(lines) + "\n")
print("  VERSIONS.md mis à jour")
PYEOF


# ════════════════════════════════════════════════════════════
# PHASE 5 — BUMP VERSION DU PACKAGE
# ════════════════════════════════════════════════════════════
step "Phase 5 — Bump version package → v${PACKAGE_VERSION}"

sed -i "s/^version\s*=\s*\".*\"/version     = \"${PACKAGE_VERSION}\"/" "${SCCROP_REPO}/pyproject.toml"
sed -i "s/__version__\s*=\s*\".*\"/__version__       = \"${PACKAGE_VERSION}\"/" "${SCCROP_REPO}/sc_crop/__init__.py"

info "pyproject.toml : version = \"${PACKAGE_VERSION}\""
info "__init__.py    : __version__ = \"${PACKAGE_VERSION}\""


# ════════════════════════════════════════════════════════════
# PHASE 6 — COMMIT + TAG + PUSH
# ════════════════════════════════════════════════════════════
step "Phase 6 — Commit + tag v${PACKAGE_VERSION} + push"

git -C "${SCCROP_REPO}" add \
    sc_crop/download.py \
    sc_crop/__init__.py \
    sc_crop/config.yaml \
    pyproject.toml \
    VERSIONS.md

git -C "${SCCROP_REPO}" commit \
    -m "release: v${PACKAGE_VERSION} — model v${MODEL_VERSION} (det=${DET_RUN} cls=${CLS_RUN})"

git -C "${SCCROP_REPO}" tag "v${PACKAGE_VERSION}"
git -C "${SCCROP_REPO}" push
git -C "${SCCROP_REPO}" push --tags

info "Tag v${PACKAGE_VERSION} poussé sur $(git -C "${SCCROP_REPO}" remote get-url origin)"


# ════════════════════════════════════════════════════════════
# PHASE 7 — PUBLICATION PyPI (délègue à publish.sh)
# ════════════════════════════════════════════════════════════
# publish.sh build depuis un `git archive HEAD` propre (pas le working directory) —
# ça évite structurellement d'embarquer des fichiers gitignorés (ex. poids .pt/.onnx
# de test), contrairement à un `python -m build` direct sur le dossier du repo.
# Une seule implémentation du build+upload, ici, pas deux qui peuvent diverger.
if [ "$SKIP_PYPI" = true ]; then
    echo -e "\n${YLW}Phase 7 sautée (--skip-pypi) — publier plus tard :${RST}"
    echo "  bash ${SCCROP_REPO}/publish.sh"
else
    step "Phase 7 — Publication PyPI v${PACKAGE_VERSION}"
    bash "${SCCROP_REPO}/publish.sh"
    info "Publié : https://pypi.org/project/sc-crop/${PACKAGE_VERSION}/"
fi


# ════════════════════════════════════════════════════════════
# RÉSUMÉ
# ════════════════════════════════════════════════════════════
echo ""
echo -e "${BLD}${GRN}════ Release terminée ════${RST}"
echo -e "  Modèle  : ${GRN}v${MODEL_VERSION}${RST}  → https://github.com/ivadomed/sc-crop/releases/tag/v${MODEL_VERSION}"
echo -e "  Package : ${GRN}v${PACKAGE_VERSION}${RST} → pip install git+https://github.com/ivadomed/sc-crop.git@v${PACKAGE_VERSION}"
echo -e "  VERSIONS.md : entrée ajoutée"
if [ "$SKIP_PYPI" = false ]; then
    echo -e "  PyPI    : ${GRN}pip install sc-crop==${PACKAGE_VERSION}${RST} → https://pypi.org/project/sc-crop/${PACKAGE_VERSION}/"
fi
