# Induction Motor Fault Detection

Classify induction motor state (**normal** vs **fault**) using voltage, current, temperature, and vibration measurements with existing ML models (Random Forest, XGBoost, LightGBM, SVM, MLP).

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

```bash
# 1. Inspect your raw data
python src/inspect_data.py --input data/raw/your_data.csv

# 2. Preprocess → train/test split + scaling
python src/preprocess.py --input data/raw/your_data.csv --output data/processed/

# 3. Train
python src/train.py --data data/processed/train.csv --model random_forest --output models/random_forest.pkl

# 4. Evaluate → saves results/metrics.txt + results/confusion_matrix.png
python src/evaluate.py --model models/random_forest.pkl --data data/processed/test.csv --output results/
```

## Data Format

CSV with columns: `voltage, current, temperature, vibration, label` (label: 0 = normal, 1 = fault)

## Structure

```
Induction-Motor-ML/
├── AGENTS.md              # AI agent instructions
├── data/
│   ├── raw/               # Huge CSVs (gitignored)
│   └── processed/         # ML-ready train/test CSVs
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── notebooks/
├── models/                # .pkl files (gitignored)
├── results/               # metrics.txt, confusion_matrix.png
├── report/
├── presentation/
└── tests/
```
