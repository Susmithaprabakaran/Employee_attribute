

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib

# Optional libraries -------------------------------------------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# =======================================================================================
# 1. DATA LOADING / SYNTHETIC DATA GENERATION
# =======================================================================================
def generate_synthetic_hr_data(n_employees: int = 1500, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic, IBM-HR-Attrition-style synthetic dataset.
    Attrition probability is deliberately driven by a mix of the underlying
    features (overtime, low satisfaction, low income, short tenure, long
    commute, poor work-life balance) so the trained models have real signal
    to learn from - not just noise.
    """
    rng = np.random.default_rng(random_state)

    departments = ["Sales", "Research & Development", "Human Resources"]
    dept_probs = [0.35, 0.55, 0.10]

    job_roles_by_dept = {
        "Sales": ["Sales Executive", "Sales Representative", "Manager"],
        "Research & Development": ["Research Scientist", "Laboratory Technician",
                                    "Manufacturing Director", "Research Director"],
        "Human Resources": ["HR Specialist", "HR Manager"],
    }

    education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree",
                         "Human Resources", "Other"]
    marital_status = ["Single", "Married", "Divorced"]

    data = {
        "EmployeeID": np.arange(1, n_employees + 1),
        "Age": rng.integers(18, 60, n_employees),
        "Department": rng.choice(departments, n_employees, p=dept_probs),
        "MaritalStatus": rng.choice(marital_status, n_employees, p=[0.32, 0.48, 0.20]),
        "EducationField": rng.choice(education_fields, n_employees),
        "DistanceFromHome": rng.integers(1, 30, n_employees),
        "MonthlyIncome": rng.integers(2000, 20000, n_employees),
        "YearsAtCompany": rng.integers(0, 25, n_employees),
        "YearsInCurrentRole": rng.integers(0, 18, n_employees),
        "YearsSinceLastPromotion": rng.integers(0, 15, n_employees),
        "TotalWorkingYears": None,  # filled below
        "TrainingTimesLastYear": rng.integers(0, 6, n_employees),
        "NumCompaniesWorked": rng.integers(0, 9, n_employees),
        "JobSatisfaction": rng.integers(1, 5, n_employees),       # 1-4
        "EnvironmentSatisfaction": rng.integers(1, 5, n_employees),
        "RelationshipSatisfaction": rng.integers(1, 5, n_employees),
        "WorkLifeBalance": rng.integers(1, 5, n_employees),
        "JobInvolvement": rng.integers(1, 5, n_employees),
        "PerformanceRating": rng.choice([3, 4], n_employees, p=[0.85, 0.15]),
        "OverTime": rng.choice(["Yes", "No"], n_employees, p=[0.30, 0.70]),
        "BusinessTravel": rng.choice(
            ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
            n_employees, p=[0.10, 0.70, 0.20]
        ),
        "Gender": rng.choice(["Male", "Female"], n_employees, p=[0.6, 0.4]),
        "StockOptionLevel": rng.integers(0, 4, n_employees),
        "PercentSalaryHike": rng.integers(10, 25, n_employees),
    }

    df = pd.DataFrame(data)
    df["JobRole"] = df["Department"].apply(lambda d: rng.choice(job_roles_by_dept[d]))
    df["TotalWorkingYears"] = (df["YearsAtCompany"] + rng.integers(0, 10, n_employees)).clip(0, 40)

    # ---- Build a realistic attrition probability from multiple drivers ----
    logit = (
        -2.0
        + 1.1 * (df["OverTime"] == "Yes").astype(int)
        + 0.55 * (df["JobSatisfaction"] <= 2).astype(int)
        + 0.5 * (df["WorkLifeBalance"] <= 2).astype(int)
        + 0.45 * (df["EnvironmentSatisfaction"] <= 2).astype(int)
        + 0.6 * (df["MonthlyIncome"] < 3500).astype(int)
        + 0.5 * (df["YearsAtCompany"] < 2).astype(int)
        + 0.35 * (df["DistanceFromHome"] > 20).astype(int)
        + 0.3 * (df["MaritalStatus"] == "Single").astype(int)
        + 0.25 * (df["BusinessTravel"] == "Travel_Frequently").astype(int)
        - 0.35 * (df["StockOptionLevel"] >= 2).astype(int)
        - 0.02 * (df["Age"] - 35)
        + rng.normal(0, 0.35, n_employees)  # noise
    )
    prob_attrition = 1 / (1 + np.exp(-logit))
    df["Attrition"] = (rng.random(n_employees) < prob_attrition).astype(int)
    df["Attrition"] = df["Attrition"].map({1: "Yes", 0: "No"})

    return df


def load_data(csv_path: str | None) -> pd.DataFrame:
    if csv_path and os.path.exists(csv_path):
        print(f"[INFO] Loading real dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        if csv_path:
            print(f"[WARN] File '{csv_path}' not found. Falling back to synthetic data.")
        else:
            print("[INFO] No --data path provided. Generating synthetic HR dataset...")
        df = generate_synthetic_hr_data()
    return df


# =======================================================================================
# 2. PREPROCESSING
# =======================================================================================
def preprocess(df: pd.DataFrame):
    df = df.copy()

    if "Attrition" not in df.columns:
        raise ValueError("Dataset must contain an 'Attrition' target column (Yes/No or 1/0).")

    # Normalize target to 0/1
    if not pd.api.types.is_numeric_dtype(df["Attrition"]):
        df["Attrition"] = df["Attrition"].apply(
            lambda v: 1 if str(v).strip().lower() in ("yes", "1", "true") else 0
        ).astype(int)
    else:
        df["Attrition"] = df["Attrition"].astype(int)

    # Drop obvious ID / constant columns if present
    drop_cols = [c for c in ["EmployeeID", "EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    y = df["Attrition"]
    X = df.drop(columns=["Attrition"])

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"[INFO] Numeric features   ({len(numeric_cols)}): {numeric_cols}")
    print(f"[INFO] Categorical features({len(categorical_cols)}): {categorical_cols}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return X, y, preprocessor, numeric_cols, categorical_cols


# =======================================================================================
# 3. HYBRID MODEL DEFINITION
# =======================================================================================
def build_models(preprocessor):
    """
    Returns a dict of named sklearn Pipelines:
        - Individual base models (for comparison)
        - The final HYBRID stacking ensemble
    """
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
    )
    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

    base_estimators = [("random_forest", rf), ("gradient_boosting", gb), ("svm", svm)]

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1
        )
        base_estimators.append(("xgboost", xgb))

    # --- The HYBRID model: stacking combines all base learners via a
    #     logistic-regression meta learner trained on their out-of-fold predictions.
    hybrid = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    models = {
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "SVM": svm,
        "Logistic Regression": logreg,
    }
    if HAS_XGB:
        models["XGBoost"] = xgb
    models["HYBRID (Stacking Ensemble)"] = hybrid

    pipelines = {
        name: Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        for name, model in models.items()
    }
    return pipelines


# =======================================================================================
# 4. TRAIN + EVALUATE
# =======================================================================================
def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    return pipeline, metrics, y_pred, y_proba


def run_training(X, y, preprocessor, outdir):
    # SMOTE needs numeric input, so we resample AFTER preprocessing for the models
    # that benefit from it. Simpler & robust approach: pass class_weight='balanced'
    # everywhere (already done above) AND optionally SMOTE on the preprocessed matrix
    # for the hybrid model's training fold.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipelines = build_models(preprocessor)

    results = []
    fitted_pipelines = {}
    roc_curves = {}

    for name, pipe in pipelines.items():
        print(f"\n[TRAINING] {name} ...")
        fitted_pipe, metrics, y_pred, y_proba = evaluate_model(
            name, pipe, X_train, y_train, X_test, y_test
        )
        fitted_pipelines[name] = fitted_pipe
        results.append(metrics)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves[name] = (fpr, tpr, metrics["ROC-AUC"])

        print(f"  Accuracy : {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall   : {metrics['Recall']:.4f}")
        print(f"  F1-Score : {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

    # Detailed classification report + confusion matrix for the HYBRID model
    hybrid_name = "HYBRID (Stacking Ensemble)"
    hybrid_pipe = fitted_pipelines[hybrid_name]
    y_pred_hybrid = hybrid_pipe.predict(X_test)

    report_txt = classification_report(y_test, y_pred_hybrid, target_names=["Stay", "Attrition"])
    cm = confusion_matrix(y_test, y_pred_hybrid)

    # 5-fold cross-validation sanity check on the hybrid model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(hybrid_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\n[CROSS-VALIDATION] Hybrid model 5-fold ROC-AUC: "
          f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        "results_df": results_df,
        "fitted_pipelines": fitted_pipelines,
        "hybrid_pipe": hybrid_pipe,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_hybrid": y_pred_hybrid,
        "confusion_matrix": cm,
        "classification_report": report_txt,
        "roc_curves": roc_curves,
        "cv_scores": cv_scores,
    }


# =======================================================================================
# 5. PLOTTING / WORKFORCE ANALYTICS
# =======================================================================================
def plot_model_comparison(results_df, outdir):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = os.path.join(outdir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_roc_curves(roc_curves, outdir):
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, (fpr, tpr, auc) in roc_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    path = os.path.join(outdir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_confusion_matrix(cm, outdir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Stay", "Attrition"], yticklabels=["Stay", "Attrition"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - Hybrid Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")


def plot_feature_importance(fitted_pipelines, numeric_cols, categorical_cols, outdir):
    """Uses the Random Forest base model (interpretable) for feature importances."""
    rf_pipe = fitted_pipelines.get("Random Forest")
    if rf_pipe is None:
        return
    ohe = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = list(ohe.get_feature_names_out(categorical_cols)) if categorical_cols else []
    all_feature_names = numeric_cols + cat_feature_names

    importances = rf_pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax, color="#4C72B0")
    ax.set_title("Top 15 Feature Importances (Random Forest)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")
    return imp_df


def workforce_analytics(df, outdir):
    """
    General HR analytics dashboard: attrition rate cut by several dimensions.
    Works generically as long as the columns exist; skips ones that don't.
    """
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df["Attrition"]):
        df["AttritionFlag"] = df["Attrition"].apply(
            lambda v: 1 if str(v).strip().lower() in ("yes", "1", "true") else 0
        ).astype(int)
    else:
        df["AttritionFlag"] = df["Attrition"].astype(int)

    overall_rate = df["AttritionFlag"].mean() * 100
    print(f"\n[ANALYTICS] Overall attrition rate: {overall_rate:.2f}%")

    dims = [d for d in ["Department", "JobRole", "OverTime", "MaritalStatus", "BusinessTravel"]
            if d in df.columns]

    n = len(dims) + 1  # +1 for income/tenure distribution
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 5 * nrows))
    axes = axes.flatten()

    for i, dim in enumerate(dims):
        rate = df.groupby(dim)["AttritionFlag"].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.values, y=rate.index, ax=axes[i], color="#DD8452")
        axes[i].set_title(f"Attrition Rate by {dim}", fontweight="bold")
        axes[i].set_xlabel("Attrition Rate (%)")

    # Income distribution: attrition vs stay
    idx = len(dims)
    if "MonthlyIncome" in df.columns and idx < len(axes):
        sns.kdeplot(data=df, x="MonthlyIncome", hue="Attrition", fill=True, ax=axes[idx])
        axes[idx].set_title("Monthly Income Distribution by Attrition", fontweight="bold")

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    path = os.path.join(outdir, "workforce_analytics_dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

    # Correlation heatmap (numeric features only)
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["AttritionFlag"], errors="ignore")
    numeric_df["Attrition"] = df["AttritionFlag"]
    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=False)
    ax.set_title("Correlation Heatmap - Numeric Features", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[SAVED] {path}")

    return overall_rate


# =======================================================================================
# 6. MAIN PIPELINE
# =======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Hybrid ML Employee Attrition Prediction & Workforce Analytics")
    parser.add_argument("--data", type=str, default=None,
                         help="Path to your HR CSV file (must contain an 'Attrition' column). "
                              "If omitted, a synthetic dataset is generated automatically.")
    parser.add_argument("--outdir", type=str, default="outputs",
                         help="Folder where plots, model, and reports will be saved.")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("=" * 90)
    print(" HYBRID ML EMPLOYEE ATTRITION PREDICTION & WORKFORCE ANALYTICS ".center(90, "="))
    print("=" * 90)
    print(f"[INFO] XGBoost available   : {HAS_XGB}")
    print(f"[INFO] SMOTE (imblearn) avail: {HAS_SMOTE}")

    # 1. Load data
    df = load_data(args.data)
    print(f"\n[INFO] Dataset shape: {df.shape}")
    print(df.head())

    # Save the (possibly synthetic) raw dataset for reference
    raw_path = os.path.join(outdir, "employee_dataset.csv")
    df.to_csv(raw_path, index=False)
    print(f"[SAVED] {raw_path}")

    # 2. Workforce analytics (on raw data, before encoding)
    overall_rate = workforce_analytics(df, outdir)

    # 3. Preprocess
    X, y, preprocessor, numeric_cols, categorical_cols = preprocess(df)

    # 4. Train + evaluate all models (including the HYBRID stacking ensemble)
    train_out = run_training(X, y, preprocessor, outdir)

    results_df = train_out["results_df"]
    print("\n" + "=" * 60)
    print(" FINAL MODEL LEADERBOARD (sorted by ROC-AUC) ".center(60, "="))
    print("=" * 60)
    print(results_df.to_string(index=False))

    # 5. Plots
    plot_model_comparison(results_df, outdir)
    plot_roc_curves(train_out["roc_curves"], outdir)
    plot_confusion_matrix(train_out["confusion_matrix"], outdir)
    imp_df = plot_feature_importance(train_out["fitted_pipelines"], numeric_cols, categorical_cols, outdir)

    # 6. Save hybrid model + reports
    model_path = os.path.join(outdir, "hybrid_attrition_model.joblib")
    joblib.dump(train_out["hybrid_pipe"], model_path)
    print(f"\n[SAVED] Trained hybrid model -> {model_path}")

    results_csv = os.path.join(outdir, "model_leaderboard.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"[SAVED] {results_csv}")

    report_path = os.path.join(outdir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("HYBRID MODEL - CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(train_out["classification_report"])
        f.write(f"\n\nOverall workforce attrition rate: {overall_rate:.2f}%\n")
        f.write(f"5-fold CV ROC-AUC: {train_out['cv_scores'].mean():.4f} "
                f"(+/- {train_out['cv_scores'].std():.4f})\n")
    print(f"[SAVED] {report_path}")

    # 7. Predictions on the held-out test set
    X_test = train_out["X_test"]
    preds_df = X_test.copy()
    preds_df["Actual_Attrition"] = train_out["y_test"].values
    preds_df["Predicted_Attrition"] = train_out["y_pred_hybrid"]
    preds_path = os.path.join(outdir, "test_set_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"[SAVED] {preds_path}")

    print("\n" + "=" * 90)
    print(" DONE. All outputs saved to: {} ".format(os.path.abspath(outdir)).center(90, "="))
    print("=" * 90)


if __name__ == "__main__":
    main()
