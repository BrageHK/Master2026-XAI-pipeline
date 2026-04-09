# PI-CAI XAI Pipeline

XAI attribution pipeline for prostate cancer detection on PI-CAI, with a Flask visualizer.

## Models

| Model | Script | Activation | Output layout |
|-------|--------|------------|---------------|
| nnUNet | `process_fold_nnunet` | **Softmax applied manually** — `torch.softmax(out, dim=1)[:, 1]` (model returns raw logits) | `(B, 2, D, H, W)` |
| U-MambaMTL | `process_fold_monai` | Sigmoid — `torch.sigmoid(out[:, 1])` | `(B, C, H, W, D)` |
| SwinUNETR | `process_fold_monai` | Sigmoid — `torch.sigmoid(out[:, 1])` | `(B, C, H, W, D)` |

Checkpoints: `U_MambaMTL_XAI/gc_algorithms/base_container/models/{model}/weights/f{fold}.ckpt`
nnUNet: `picai_nnunet/results/nnUNet/fold_{fold}/model_best.model`

## Key scripts

- `generate_xai_data.py` — inference + Saliency/Occlusion/AblationCAM/InputAblation per fold
- `analyze_xai.py` — aggregate metrics and charts from saved `.npz` files
- `web/app.py` — Flask case browser
- `src/ablation_cam_3d.py` — 3D AblationCAM implementation

## Running

```bash
uv run generate_xai_data.py --models umamba_mtl --fold 0 --methods all
uv run generate_xai_data.py --models all --fold 0,1,2,3,4 --methods all --compute-metrics-only
uv run python web/app.py
sbatch jobs/job_{umamba,swin,nnunet}.slurm   # SLURM array jobs (folds 0-4)
```

## Outputs

```
results/xai/{model}/fold_{n}/{case_id}.npz      # image, pred, label, zones, attributions
results/xai/{model}/fold_{n}/progress.json      # per-case classification + channel stats
results/metrics/{model}/sample_data.json        # all cases (used by web app)
results/metrics/{model}/summary/*.png           # confusion matrix, zone dist, channel charts
```

## Conventions

- `has_pca = lbl_crop.sum() > 1` — ground-truth positive (threshold 1 filters noise voxels)
- XAI maps are only computed and saved when `predicted_pos = True`
- Channel order: `[T2W, ADC, HBV]`
- Zone encoding in `.npz`: `0=background, 1=PZ, 2=TZ`
- `_make_forward_func_sigmoid` → MONAI models; `_make_forward_func_softmax` → nnUNet
