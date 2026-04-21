# PI-CAI XAI Pipeline

XAI attribution pipeline for prostate cancer detection on PI-CAI, with a Flask visualizer.

## Models

| Model | Activation |
|-------|------------|
| nnUNet | Softmax — `torch.softmax(out, dim=1)[:, 1]` |
| U-MambaMTL | Sigmoid — `torch.sigmoid(out[:, 1])` |
| SwinUNETR | Sigmoid — `torch.sigmoid(out[:, 1])` |

## Usage

```bash
# Run XAI (skips cases where all requested methods already exist in .npz)
uv run main.py --models umamba_mtl --fold 0 --methods saliency

# Add a new method to existing .npz files without touching other keys
uv run main.py --models all --fold 0,1,2,3,4 --methods integrated_gradients

# Force recompute and overwrite existing results
uv run main.py --models umamba_mtl --fold 0 --methods all --overwrite

# Recompute metrics/charts only (no GPU needed)
uv run main.py --models umamba_mtl --fold 0 --methods saliency --compute-metrics-only

# Aggregate analysis across all folds
uv run analyze_xai.py --model umamba_mtl

# Web viewer
uv run python web/app.py

# SLURM array jobs (folds 0–4)
sbatch jobs/umamba.slurm
sbatch jobs/swin.slurm
sbatch jobs/nnunet.slurm
```

## Source layout

```
main.py                          CLI + process_fold dispatcher
analyze_xai.py                   aggregate metrics and charts from .npz files
web/app.py                       Flask case browser

src/
├── utils.py                     shared helpers (_pad, _sentinel, methods_already_computed, …)
├── models/
│   ├── loader.py                load_model / load_nnunet / load_mamba
│   └── preprocessing.py        nnUNet preprocessing, splits, label loading
├── zones/
│   └── zones.py                 zone extraction + _ensure_umamba_zones
├── xai/
│   ├── forward_wrappers.py      _make_forward_func_sigmoid / _softmax
│   ├── occlusion.py             zone-median baseline sampling + baseline tensor builder
│   └── ablation_cam_3d.py       3D-compatible AblationCAM
├── metrics/
│   ├── compute.py               compute_metrics, _channel_stats, _zone_category
│   ├── progress.py              _build_progress_record, _save_progress
│   └── charts.py                generate_charts and all plot_* functions
└── pipeline/
    ├── monai_processor.py       process_fold_monai  (umamba_mtl, swin_unetr)
    └── nnunet_processor.py      process_fold_nnunet
```

## Outputs

```
results/xai/{model}/fold_{n}/{case_id}.npz      image, pred, label, zones, attributions
results/xai/zones/fold_{n}/{case_id}.npz        umamba zone predictions (shared by all models)
results/metrics/{model}/sample_data.json        per-case records (used by web app)
results/metrics/{model}/summary/*.png           confusion matrix, zone dist, channel charts
results/analysis/{model}/                       summary.json + charts from analyze_xai.py
```

## Key conventions

- `has_pca = lbl_crop.sum() > 1` — ground-truth positive (threshold 1 filters noise)
- XAI maps only computed and saved when `predicted_pos = True`
- Channel order: `[T2W, ADC, HBV]`
- Zone encoding in `.npz`: `0=background, 1=PZ, 2=TZ`
- Default run **merges** new method keys into existing `.npz`; `--overwrite` rewrites from scratch
