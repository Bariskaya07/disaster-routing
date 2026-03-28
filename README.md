---
title: TUA Disaster Routing MVP
emoji: 🛰️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
---

# TUA Disaster Routing MVP

This repository implements a hackathon-ready MVP for disaster response routing from pre/post satellite imagery.

## What It Does

- Runs a dual-stage xView2-compatible DPN pipeline:
  - `dpn_unet` for building localization
  - `dpn_seamese_unet_shared` for damage classification
- Converts building damage output into a georeferenced damage raster
- Pulls a real street network with OSMnx, projects it to a metric CRS, snaps operator GPS points to routable nodes, and solves:
  - shortest-length route
  - damage-aware safest route
- Serves the full workflow through a Streamlit + Folium operator console

## Local Weight Strategy

The repository is designed with `local-first, HF-fallback` loading:

1. explicit environment override paths
2. project-root local checkpoint files
3. Hugging Face model repository cache

Expected checkpoint names:

- `localization_dpn_unet_dpn92_0_best_dice`
- `pseudo_dpn_seamese_unet_shared_dpn92_0_best_xview`

These files are ignored by git on purpose.

## Metadata Contract

Routing requires a metadata JSON payload with a geographic bounding box:

```json
{
  "scene_id": "ankara-demo-01",
  "bbox": [32.8000, 39.8500, 32.8100, 39.8600],
  "start": [39.8510, 32.8010],
  "goal": [39.8590, 32.8090]
}
```

- `bbox` order: `[min_lon, min_lat, max_lon, max_lat]`
- `start` and `goal` order: `[lat, lon]`

If bbox is missing, the application will still run inference but will intentionally disable routing.

## Demo-First Behavior

The Streamlit app now starts in demo mode by default so it can be opened directly by judges without uploading files first.

- select a disaster type
- select a scene
- click `Demo Yükle`
- click `Analizi Başlat`

## Demo Library Workflow

The xView2 archive labels are not the same as the application's lightweight metadata contract.

- xView2 label files contain building polygons and damage annotations
- the app expects a compact `metadata.json` with `bbox`, optional `start`, and optional `goal`

Recommended local workflow:

1. Build a candidate library from `hold_images_labels_targets.tar`
2. Inspect `data/demo_library/manifest.json`
3. Activate one scene into `data/demo/`

```bash
/home/bariskaya/Projelerim/UAV/venv/bin/python scripts/prepare_demo_library.py --activate-scene santa-rosa-wildfire_00000157
```

If you want to switch the active scene later:

```bash
/home/bariskaya/Projelerim/UAV/venv/bin/python scripts/activate_demo_scene.py data/demo_library/palu-tsunami/palu-tsunami_00000180
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## CLI Inference

```bash
python inference.py \
  --pre path/to/pre.png \
  --post path/to/post.png \
  --metadata path/to/metadata.json \
  --output-dir outputs/inference
```

## Hugging Face Spaces Deployment

This repository is configured for a Docker-based Hugging Face Space because the built-in Streamlit SDK is deprecated by Hugging Face documentation.

Recommended environment variables for Space deployment:

- `HF_MODEL_REPO_ID`
- `HF_MODEL_REVISION` (optional)
- `HF_TOKEN` (optional for private model repos)
- `LOCALIZATION_WEIGHTS_PATH` (optional override)
- `DAMAGE_WEIGHTS_PATH` (optional override)

If the model weights are not available in the Space environment, the app still boots and falls back to a lightweight difference-based demo mode.

## Tests

```bash
python -m unittest discover -s tests -v
```
