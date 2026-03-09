# PI-CAI XAI Pipeline

Explainability (XAI) pipeline for prostate cancer detection models on the [PI-CAI dataset](https://pi-cai.grand-challenge.org/).
Supports three models: **U-MambaMTL**, **SwinUNETR**, and **nnUNet**.

---

## Workflow Overview

```
generate_xai_data.py   →   results/xai/          (NPZ files + progress.json)
                       →   results/metrics/       (sample_data.json + charts)
                            ↓
analyze_xai.py         →   results/analysis/      (summary.json + PNG charts)
                            ↓
web/app.py             →   http://localhost:5000   (interactive browser UI)
```

---

## 1. Generate XAI Data

`generate_xai_data.py` runs inference on validation cases for one or more models/folds, computes Saliency and Occlusion attribution maps for positive predictions, and saves per-case `.npz` files alongside per-fold `progress.json` files.

### Basic usage

```bash
# Single model, single fold, single method
python generate_xai_data.py --models umamba_mtl --fold 0 --methods saliency

# Multiple models, all folds, all methods
python generate_xai_data.py --models umamba_mtl swin_unetr nnunet \
    --fold 0,1,2,3,4 --methods saliency occlusion

# Shorthand: all models, all methods
python generate_xai_data.py --models all --fold 0 --methods all

# Skip inference — only recompute metrics from existing NPZ files
python generate_xai_data.py --models swin_unetr --compute-metrics-only
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--models` | — | Models to run: `umamba_mtl`, `swin_unetr`, `nnunet`, or `all` |
| `--fold` | — | Comma-separated fold indices, e.g. `0,1,2,3,4` |
| `--methods` | — | Attribution methods: `saliency`, `occlusion`, or `all` |
| `--compute-metrics-only` | off | Skip inference; recompute metrics from saved NPZ files |

### Outputs

| Path | Contents |
|---|---|
| `results/xai/{model}/fold_{n}/{case_id}.npz` | Image, prediction, label, zones, saliency/occlusion arrays |
| `results/xai/{model}/fold_{n}/progress.json` | Per-case status, classification (TP/FP/TN/FN), zone stats, channel fractions |
| `results/metrics/{model}/sample_data.json` | Aggregated case records used by the web app |
| `results/metrics/{model}/summary/` | Confusion matrix, zone distribution, channel activation PNGs |

---

## 2. Analyze XAI Results

`analyze_xai.py` reads the `progress.json` files produced above, aggregates statistics across all folds, and saves summary charts.

### Basic usage

```bash
# Analyze one model
python analyze_xai.py --model umamba_mtl

# Analyze all models
python analyze_xai.py --model all

# Analyze specific models
python analyze_xai.py --model umamba_mtl swin_unetr
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `all` | Model(s) to analyze: `umamba_mtl`, `swin_unetr`, `nnunet`, or `all` |
| `--output-dir` | `results/analysis/` | Directory for output files |

### Outputs (per model, under `results/analysis/{model}/`)

| File | Description |
|---|---|
| `summary.json` | Aggregated counts (TP/FP/TN/FN), precision, sensitivity, specificity, F1, zone distribution, mean channel fractions |
| `confusion_matrix.png` | Bar chart of classification counts + metrics table |
| `zone_distribution.png` | Case counts split by prostate zone category |
| `channel_activation/{method}/{filter}/pie.png` | Mean channel importance pie charts (T2W / ADC / HBV) for filters: `overall`, `tp`, `fp`, `pz`, `tz` |

---

## 3. Web App

`web/app.py` is a Flask server for interactive browsing of XAI results. It reads the `results/metrics/` data (from `generate_xai_data.py`) and streams rendered MRI slices from `results/xai/`.

### Start the server

```bash
# Default: http://127.0.0.1:5000
uv run python web/app.py

# Custom host/port
uv run python web/app.py --host 0.0.0.0 --port 8080

# Enable Flask debug mode
uv run python web/app.py --debug
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `5000` | Bind port |
| `--debug` | off | Enable Flask debug/auto-reload |

### Pages

| URL | Description |
|---|---|
| `/` | Overview dashboard — model stats (TP/FP/TN/FN, F1, precision, etc.) for all three models |
| `/model/{model}` | Case list + pre-generated charts for a specific model |
| `/model/{model}/case/{case_id}` | Per-case viewer: MRI slices (T2W/ADC/HBV), cancer probability, ground-truth label, saliency map, occlusion map, prostate zones |

The case viewer renders all depth slices as images; use the slice slider to navigate through the 3-D volume.

### API endpoints

| Endpoint | Returns |
|---|---|
| `GET /api/cases/{model}.json` | All case records for a model |
| `GET /api/case/{model}/{case_id}.json` | All rendered slices + stats for one case (cached after first load) |
| `GET /api/chart/{model}/{relpath}` | Serve a pre-generated PNG chart |

---

## Dependencies

Install with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Key packages: `torch`, `captum`, `flask`, `matplotlib`, `numpy`, `SimpleITK`.

---

## Directory Structure

```
.
├── generate_xai_data.py   # XAI data generation pipeline
├── analyze_xai.py         # Offline analysis of XAI results
├── web/
│   ├── app.py             # Flask web server
│   └── templates/         # Jinja2 HTML templates
├── results/
│   ├── xai/               # Per-case NPZ files + progress.json (generated)
│   ├── metrics/           # sample_data.json + charts (generated)
│   └── analysis/          # Summary JSON + charts from analyze_xai.py (generated)
├── src/                   # Shared utilities
├── U_MambaMTL_XAI/        # U-MambaMTL / SwinUNETR subproject
└── picai_nnunet/          # nnUNet subproject
```
