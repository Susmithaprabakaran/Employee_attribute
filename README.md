# Hybrid ML Employee Attrition Prediction & Workforce Analytics

A Python 3.12 machine learning pipeline that predicts employee attrition risk using a **hybrid stacking ensemble** and generates **workforce analytics** dashboards to help HR teams understand *why* employees leave — not just *who* might leave.

---

## Overview

Traditional attrition analysis is reactive — HR finds out someone is leaving only when they resign. This project builds a **predictive, explainable system** that:

- Flags at-risk employees *before* they resign
- Explains *why* the model thinks so (feature importance)
- Surfaces workforce-level patterns (department, overtime, income, tenure) so HR can act on root causes, not just symptoms

## Key Features

- **Hybrid Stacking Ensemble** — combines Random Forest, Gradient Boosting, SVM, and XGBoost as base learners, with a Logistic Regression meta-learner that learns how to best combine their predictions.
- **Handles class imbalance** — class-weighting + optional SMOTE oversampling (attrition datasets are naturally imbalanced).
- **Works with your data or none at all** — pass in your own HR CSV, or the script auto-generates a realistic synthetic dataset so it runs out of the box.
- **Full evaluation suite** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, 5-fold Cross-Validation, and a model leaderboard comparing all algorithms.
- **Workforce Analytics Dashboard** — attrition rate broken down by department, job role, overtime, marital status, business travel, and income distribution.
- **Explainability** — feature importance chart + correlation heatmap.
- **Exportable outputs** — trained model (`.joblib`), all charts (`.png`), leaderboard (`.csv`), classification report (`.txt`), and test-set predictions (`.csv`).

## Tech Stack

`Python 3.12` · `scikit-learn` · `XGBoost` · `imbalanced-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `joblib`

## Project Structure

```
├── employee_attrition_analytics.py   # main pipeline script
├── outputs/                          # generated on run
│   ├── employee_dataset.csv
│   ├── hybrid_attrition_model.joblib
│   ├── model_leaderboard.csv
│   ├── classification_report.txt
│   ├── test_set_predictions.csv
│   ├── workforce_analytics_dashboard.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── confusion_matrix.png
└── README.md
```

## Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn xgboost joblib
```
> `xgboost` and `imbalanced-learn` are optional — the script auto-detects and gracefully skips them if not installed.

### Run with synthetic data (no dataset needed)
```bash
python employee_attrition_analytics.py
```

### Run with your own dataset
```bash
python employee_attrition_analytics.py --data your_hr_data.csv
```
Your CSV just needs a binary `Attrition` column (`Yes`/`No` or `1`/`0`) — all other typical HR columns (Department, JobRole, OverTime, MonthlyIncome, etc.) are auto-detected.

### Custom output folder
```bash
python employee_attrition_analytics.py --data your_hr_data.csv --outdir results/
```

## How It Works

1. **Load / Generate Data** — real CSV or synthetic HR dataset (1,500 employees by default).
2. **Preprocess** — numeric scaling + one-hot encoding via `ColumnTransformer`.
3. **Train** — individual base models + the hybrid `StackingClassifier`.
4. **Evaluate** — full metric suite + cross-validation, ranked in a leaderboard.
5. **Analyze** — workforce dashboards, correlation heatmap, feature importances.
6. **Export** — model + charts + reports saved to `outputs/`.

## Sample Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Hybrid (Stacking Ensemble) | ~0.68 | — | — | — | ~0.68 |
| XGBoost | — | — | — | — | — |
| Random Forest | — | — | — | — | — |

*(Results vary by dataset size and quality — the table above is regenerated automatically as `model_leaderboard.csv` on every run.)*

## Roadmap / Future Scope

- [ ] Integrate with live HRMS/payroll data sources
- [ ] Add SHAP-based explainability for individual predictions
- [ ] Build a simple web dashboard (Streamlit/Flask) for HR self-service
- [ ] Add time-series attrition trend forecasting

## License

MIT License — free to use, modify, and distribute.

## Contributing

Issues and pull requests are welcome. If you use this on a real dataset, consider sharing anonymized performance benchmarks.
