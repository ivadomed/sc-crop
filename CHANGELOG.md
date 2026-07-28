# Changelog

Notable changes to the `sc-crop` **package** (API, CLI, behavior). For which detection model
ships with which package version, see [VERSIONS.md](VERSIONS.md) — this file does not repeat
that mapping. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [0.10.0] — 2026-07-27

### Added
- `write_crop_metadata()` and the `sc_crop write-metadata` CLI subcommand, to generate a
  `crop_metadata.yaml` for a model release (e.g. for SCT's `sct_deepseg`) without cloning the
  repo. Replaces `examples/write_sct_crop_metadata.py`, which is removed.

### Changed
- `crop_metadata.yaml` no longer includes a `cropped_image` key. A release only ever bundles
  this file for a model that actually uses sc-crop, so the file's presence is already the
  signal — the key was redundant and consumers (SCT) never enforced it as authoritative.

## [0.9.0] — 2026-07-23

### Changed (breaking)
- `check_seg_truncation()` now returns `bbox` key strings (e.g. `["zmax", "zmin"]`) instead of
  `pad_*` anatomical face names (e.g. `["pad_superior", "pad_inferior"]`). The old return values
  weren't valid keys into the `bbox` dict returned by `detect()` (only `xmin`/`xmax`/etc. are),
  so `bbox[face]` on the old output always raised `KeyError` — this was never usable as
  documented. The new keys are directly usable as `bbox[key]`, and map 1:1 to the `-box-*` CLI
  arguments in SCT's `sct_deepseg`.

## [0.8.1] — 2026-07-21
### Fixed
- `LICENSE` copyright holder corrected to NeuroPoly Lab, Polytechnique Montréal.

## [0.8.0] — 2026-07-21

### Removed (breaking)
- `segment_onnx()`, `segment_pt()`, and the CLI commands `sc-segment-onnx`/`sc-segment-pt` —
  the end-to-end nnUNet segmentation convenience wrappers, and the `sc-crop[segment]` /
  `nnunet-onnx` optional dependency they required. `detect()`/`crop()`/`uncrop()` are unaffected;
  the same 4-step pattern (detect → crop → run your own model → uncrop) is documented in the
  README's "nnUNet integration" section and `examples/infer_with_sc_crop.py`.

### Added
- `LICENSE` — MIT.
- `[project.urls]` (Homepage/Repository/Issues) and `classifiers` in `pyproject.toml` — was
  entirely absent before, so PyPI showed no project links and no license.
- `scripts/publish_release.sh` — the PyPI/GitHub-release publish step is now scripted (build,
  anti-regression check for accidentally-bundled model weights, `twine check`, `twine upload`)
  instead of a manual, untracked step. See [MIGRATION.md](https://github.com/ivadomed/model_cropping_sc_contrast-agnostic_yolo/blob/main/MIGRATION.md)
  on the training repo for the full release flow (training repo exports → this repo publishes).
- `config.yaml`'s `imgsz` field is now propagated automatically from the training run at export
  time (previously had to be added by hand — see `v0.6.0` note below).

### Fixed
- `pyproject.toml` was missing `readme = "README.md"`, so the PyPI project description was
  blank. (`v0.7.1`)
- `v0.7.0` accidentally shipped `.pt`/`.onnx` model weights inside the sdist (22 MB instead of
  ~40 KB) — gitignored files picked up by `python -m build` from an unclean local checkout, with
  no corresponding git tag. `publish_release.sh` now refuses to publish if any `.pt`/`.onnx` file
  is found in the built sdist. (`v0.7.1`)

## [0.7.2] — 2026-07-21
Packaging metadata only (`project.urls`, `classifiers`) — see above, folded into the `0.8.0` notes since it shipped one day earlier in the same cleanup.

## [0.7.1] — 2026-07-20
PyPI description + bundled-weights fix — see above.

## [0.6.0] — 2026-06-23
### Changed
- Detector inference migrated from a hand-rolled ONNX Runtime path to `ultralytics`
  `YOLO.predict()`; classifier inference uses `onnxruntime` directly with an explicit letterbox
  matching `LetterboxClsTransform` from training (avoids `ultralytics` forcing `CenterCrop` on
  ONNX classify models). `ultralytics` promoted from an optional extra to a core dependency.
  Verified bit-exact against the previous path (`scripts/compare_inference.py`,
  `scripts/compare_cls_inference.py`).
- `config.yaml` gained an `imgsz` field (added by hand in this commit, not by any automated
  step — the gap closed in `0.8.0`, see above). Required because ONNX models don't embed `imgsz`
  metadata the way `.pt` checkpoints do.

## [0.5.0] — earlier 2026-06
### Changed
- Output changed from a bbox `.txt` file to a `_cropbox.nii.gz` NIfTI mask; `detect()`+`crop()`
  (crop by default) replaces the old detect-only default CLI mode.
### Added
- Crop QC helpers: `check_label_crop()`, `CropReport`, per-face extra-padding reporting.

## [0.3.0]–[0.4.1] — 2026-05/06
### Changed
- ONNX preprocessing made to match YOLO's own `predict()` letterbox exactly (fixed an SI padding
  inversion bug in the process).
- Padding defaults tuned from real overshoot measurements: inferior 60→100mm, left/right
  10→15mm (`0.4.0`); posterior 15→22mm, anterior/posterior split from a single symmetric AP
  value (`0.4.1`).

## [0.2.0] — earlier 2026-05
### Added
- `segment_onnx()`/`segment_pt()` end-to-end nnUNet segmentation pipeline (removed in `0.8.0`,
  see above).
- `norm_scope` read from `config.yaml` instead of a hardcoded default; `slice_all` mode added.
- Model cache moved to `~/.cache/sc_crop/<model_tag>/` with SHA256 verification, decoupled from
  package version.

## Earlier (v0.0.x–v0.1.x)
Initial CLI/API development: first release, ONNX Runtime inference path (later replaced in
`0.6.0`), GPU support, `cls`/`graphtrim` regularization modes, `detect_and_crop()` convenience
function, volume-level normalisation, vectorized slice-building (5× speedup). See `git log`
for individual commits if needed — not reconstructed in detail here as none of it reflects
current behavior.
