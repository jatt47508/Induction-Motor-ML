import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

FEATURES = ["voltage", "current", "temperature", "vibration"]


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "voltage": np.random.normal(400, 5, n),
        "current": np.random.normal(50, 3, n),
        "temperature": np.random.normal(65, 5, n),
        "vibration": np.random.exponential(1.5, n),
        "label": np.random.randint(0, 2, n),
    })


def test_sample_df_columns(sample_df):
    for col in FEATURES + ["label"]:
        assert col in sample_df.columns


def test_no_missing_values(sample_df):
    assert sample_df.isnull().sum().sum() == 0


def test_train_model(sample_df):
    X, y = sample_df[FEATURES], sample_df["label"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})
