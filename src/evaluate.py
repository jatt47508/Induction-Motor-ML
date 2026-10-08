import click
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

FEATURES = ["voltage", "current", "temperature", "vibration"]
TARGET = "label"


@click.command()
@click.option("--model", "model_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_dir", type=click.Path(path_type=Path), default=Path("results"))
def main(model_path: Path, data_path: Path, output_dir: Path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    report = classification_report(y, y_pred, target_names=["normal", "fault"])
    cm = confusion_matrix(y, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if y_proba is not None:
        auc = roc_auc_score(y, y_proba)
        print(f"ROC-AUC: {auc:.4f}")
    print(f"\n{report}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics text
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1: {f1:.4f}\n\n{report}\nConfusion Matrix:\n{cm}\n")
    print(f"Saved metrics to {metrics_path}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["normal", "fault"])
    ax.set_yticks([0, 1], ["normal", "fault"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="white" if cm[i][j] > cm.max() / 2 else "black", fontsize=16)
    fig.colorbar(im)
    fig.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
