"""Reproduce the AIM 350 survey classification comparison from the original dataset ZIP.

Dataset DOI: https://doi.org/10.17045/STHLMUNI.28850645
Example: python -m src.reproduce_study --zip 28850645.zip
This is an exploratory, observational predictive analysis; not evidence of causal harm.
"""
import argparse
import csv
import io
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "Age", "Year", "Programme", "Grade", "Gender",
    "Swedish_native_language", "Whole_school_in_sweden",
    "Years_in_swedish_school", "P_struggle", "P_procrastinate",
    "P_time", "P_knowledge_or_grades", "P_results", "P_stress",
    "P_relevance", "S_rules", "S_education_about_AI",
    "S_teachers_help_use_AI",
] + [f"Known_AI_{i}" for i in range(1, 23)]
NUMERIC = [
    "Age", "Years_in_swedish_school", "P_struggle", "P_procrastinate",
    "P_time", "P_knowledge_or_grades", "P_results", "P_stress",
    "P_relevance", "S_rules", "S_education_about_AI",
    "S_teachers_help_use_AI",
]


def read_source_zip(path):
    """The publisher CSV has every row wrapped in another quotation layer."""
    with ZipFile(path) as archive:
        member = next((n for n in archive.namelist()
                       if n.endswith("repo_data_with_factors.csv")), None)
        if member is None:
            raise ValueError("Missing repo_data_with_factors.csv in source archive")
        raw = archive.read(member).decode("utf-8-sig").splitlines()
    rows = []
    for line in raw:
        # Strip the outer quoted wrapper, undo doubled quote escapes,
        # then parse the inner comma-separated record.
        field = line.strip()
        if field.startswith('"') and field.endswith('"'):
            field = field[1:-1]
        field = field.replace('""', '"')
        rows.append(next(csv.reader([field])))
    columns = rows[0]
    if columns[0] == "":
        columns = columns[1:]
        rows = [r[1:] for r in rows[1:]]
    else:
        rows = rows[1:]
    frame = pd.DataFrame(rows, columns=columns).replace({"NA": np.nan, "": np.nan})
    for column in frame.columns:
        if column not in ("ID", "gender_binary"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True,
                        help="Original 28850645.zip download from dataset DOI")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    if not args.zip.is_file():
        parser.error(f"Dataset archive not found: {args.zip}")
    df = read_source_zip(args.zip)
    missing = set(FEATURES + ["Use_school"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected fields: {sorted(missing)}")

    # Codebook: Use_school=1 daily, 2 several times/week, 3 several/month,
    # 4 rarely, 5 never. Exclude 0 (don't know), 6, and missing.
    df = df[df["Use_school"].isin([1, 2, 3, 4, 5])].copy()
    y = df["Use_school"].isin([1, 2]).astype(int)
    X = df[FEATURES].replace(999, np.nan).copy()
    categoric = [f for f in FEATURES if f not in NUMERIC]
    pre = ColumnTransformer([
        ("n", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("c", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), categoric),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, estimator in models.items():
        pipeline = Pipeline([("pre", pre), ("model", estimator)])
        folds = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        matrix = confusion_matrix(y_test, pred)
        records.append({
            "model": name, "train_n": len(X_train), "test_n": len(X_test),
            "cv_f1": folds.mean(), "cv_f1_sd": folds.std(),
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob),
            "confusion": json.dumps(matrix.tolist()),
        })
        ConfusionMatrixDisplay(matrix, display_labels=["Low", "High"]).plot(
            cmap="Blues", colorbar=False)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(args.out / (name.lower().replace(" ", "_") + "_confusion.png"),
                    dpi=160)
        plt.close()
    table = pd.DataFrame(records)
    table.to_csv(args.out / "model_comparison.csv", index=False)
    table.plot(x="model", y=["accuracy", "f1", "roc_auc"], kind="bar", rot=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(args.out / "model_comparison.png", dpi=160)
    plt.close()
    (args.out / "methodology.json").write_text(json.dumps({
        "original_rows": 1266, "included_rows": len(df),
        "high_frequency_rows": int(y.sum()),
        "low_frequency_rows": int((1-y).sum()),
        "binary_target": "Use_school: 1-2=high, 3-5=low",
        "excluded_target_codes": [0, 6, "missing"],
        "feature_columns": FEATURES, "missing_code_replaced": 999,
        "seed": 42, "test_size": 0.2,
        "validation": "5-fold stratified F1 on train only",
        "warning": "Associations only, not causal effects or cheating evidence",
    }, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
