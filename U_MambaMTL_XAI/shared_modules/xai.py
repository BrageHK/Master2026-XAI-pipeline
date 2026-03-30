from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path
from monai.transforms.intensity.array import ScaleIntensityRangePercentiles

def create_confusion_matrix(OUTPUT_DIR: Path, results: list, model):

    # Extract ground truth and predictions
    y_true = [r["has_pca"] for r in results]
    y_pred = [r["predicted_positive"] for r in results]

    # Build confusion matrix
    labels = [False, True]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"{model} — PCa Detection Confusion Matrix (all folds)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)

    return cm.ravel()

def channel_wise_normalize_and_clamp(x):
    for i in range(3):
        #x[i] = torch.clamp(x[i] + 0.5 - x[i].mean(), min=-0.999, max=0.999)
        x[i] = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(x[i])
    return x

def normalize_and_clamp(x):
    print("Max val:")
    print(x.max())
    #x = torch.clamp(x + 0.5 - x.mean(), min=-0.999, max=0.999)
    x = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(x)
    return x