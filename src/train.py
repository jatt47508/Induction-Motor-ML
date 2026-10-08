import click
import pandas as pd
from pathlib import Path
import joblib

FEATURES = ["voltage", "current", "temperature", "vibration"]
TARGET = "label"


def get_models():
    models = {}
    try:
        from sklearn.ensemble import RandomForestClassifier
        models["random_forest"] = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    except (ImportError, OSError):
        pass
    try:
        from sklearn.svm import SVC
        models["svm"] = SVC(kernel="rbf", probability=True, random_state=42)
    except (ImportError, OSError):
        pass
    try:
        from sklearn.neural_network import MLPClassifier
        models["mlp"] = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    except (ImportError, OSError):
        pass
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss")
    except (ImportError, OSError):
        pass
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    except (ImportError, OSError):
        pass
    return models


@click.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model", "model_name", default="random_forest", show_default=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=Path("models/model.pkl"))
def main(data_path: Path, model_name: str, output_path: Path):
    models = get_models()
    if model_name not in models:
        raise SystemExit(f"Unknown model '{model_name}'. Available: {list(models.keys())}")

    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]

    model = models[model_name]
    print(f"Training {model_name} on {X.shape[0]} samples...")
    model.fit(X, y)

    train_acc = model.score(X, y)
    print(f"Train accuracy: {train_acc:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
