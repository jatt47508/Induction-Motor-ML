import click
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_FEATURES = ["Voltage (V)", "Current (A)", "Temperature (\u00b0C)", "Vibration (mm/s)"]
FEATURES = ["voltage", "current", "temperature", "vibration"]
TARGET = "label"

LABEL_MAP = {"normal": 0, "low": 1, "moderate": 1, "high": 1}


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", "output_dir", type=click.Path(path_type=Path), default=Path("data/processed"))
@click.option("--test-size", default=0.2, show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(input_path: Path, output_dir: Path, test_size: float, seed: int):
    df = pd.read_csv(input_path)
    print(f"Loaded {df.shape}")

    rename_map = dict(zip(RAW_FEATURES, FEATURES))
    rename_map["Label"] = "label"
    df = df.rename(columns=rename_map)

    if "label" in df.columns:
        df["label"] = df["label"].map(LABEL_MAP)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)
    print(f"Label distribution:\n{df[TARGET].value_counts().sort_index()}")

    X, y = df[FEATURES], df[TARGET]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    joblib.dump(scaler, output_dir / "scaler.pkl")

    print(f"train: {train.shape}  test: {test.shape}")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
