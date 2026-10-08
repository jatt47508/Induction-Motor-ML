# AGENTS.md — Instructions for OpenCode

## Project
Induction Motor Fault Detection — binary classification (normal vs fault) using voltage, current, temperature, and vibration measurements from an induction motor.

## Objective
Use **existing/pre-trained models** (no custom model creation). Classify motor state as `0 = normal` or `1 = fault`.

## Directory Map
- `data/raw/` — huge raw CSVs (gitignored, upload locally)
- `data/processed/` — small ML-ready CSVs (committed)
- `src/inspect_data.py` — load & summarize raw data
- `src/preprocess.py` — clean, scale, split → `data/processed/`
- `src/train.py` — train a model using sklearn/xgboost/lightgbm
- `src/evaluate.py` — evaluate on test set, save metrics to `results/`
- `models/` — saved model `.pkl` files (gitignored)
- `results/` — confusion matrix PNG, metrics txt (committed)
- `notebooks/` — exploratory analysis
- `report/` — final report documents
- `presentation/` — slides

## Conventions
- Python 3.11+, scikit-learn for all models
- Feature columns: `voltage`, `current`, `temperature`, `vibration`
- Target column: `label` (0 = normal, 1 = fault)
- Use `StandardScaler` before fitting
- Save everything under `results/` with descriptive filenames
- Do NOT commit large CSVs or `.pkl` model files (see `.gitignore`)

## Commands
```bash
pip install -r requirements.txt
python src/inspect_data.py --input data/raw/motor_data.csv
python src/preprocess.py --input data/raw/motor_data.csv --output data/processed/
python src/train.py --data data/processed/train.csv --model random_forest --output models/random_forest.pkl
python src/evaluate.py --model models/random_forest.pkl --data data/processed/test.csv --output results/
pytest tests/ -v
```
