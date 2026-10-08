import click
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

FEATURES = ["voltage", "current", "temperature", "vibration"]
TARGET = "label"

MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, eval_metric="logloss"),
    "lightgbm": LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}


@click.command()
@click.option("--data", "data_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model", "model_name", type=click.Choice(list(MODELS.keys())), default="random_forest", show_default=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path), default=Path("models/model.pkl"))
def main(data_path: Path, model_name: str, output_path: Path):
    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]

    model = MODELS[model_name]
    print(f"Training {model_name} on {X.shape[0]} samples...")
    model.fit(X, y)

    train_acc = model.score(X, y)
    print(f"Train accuracy: {train_acc:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
