"""Reproducible classification runner for AIM 350.

Requires a locally obtained, appropriately licensed survey CSV.
Provide an explicit target column and positive class. Never infer labels automatically.
Example:
 python -m src.train_models --data data/survey.csv --target ai_use_level --positive high
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", required=True, help="Existing binary target column, chosen from codebook")
    parser.add_argument("--positive", required=True, help="Value indicating higher AI use")
    parser.add_argument("--exclude", nargs="*", default=[], help="Target-defining/leaky or identifying columns")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}. See data/README.md")
    df = pd.read_csv(args.data).drop_duplicates()
    if args.target not in df:
        parser.error(f"Target column '{args.target}' not in CSV. Check the survey codebook.")
    absent = [x for x in args.exclude if x not in df.columns]
    if absent:
        parser.error(f"Unknown excluded columns: {absent}")
    df = df.dropna(subset=[args.target]).copy()
    observed = df[args.target].astype(str).unique().tolist()
    if len(observed) != 2 or args.positive not in observed:
        parser.error(f"Expected exactly two target values including '{args.positive}'; observed {observed}. "
                     "Create and document a justified binary label before training.")
    y = (df[args.target].astype(str) == args.positive).astype(int)
    if y.value_counts().min() < 6:
        parser.error("Each class must have at least six observations for 5-fold CV.")
    X = df.drop(columns=[args.target] + args.exclude)
    if X.empty or X.shape[1] == 0:
        parser.error("No predictor columns remain.")
    # Leakage screening requires human review of the source questionnaire/codebook.
    print("Review predictors carefully for target leakage and respondent identifiers:")
    print(", ".join(X.columns))
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categoric = X.columns.difference(numeric).tolist()
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                   ("encode", OneHotEncoder(handle_unknown="ignore"))]), categoric),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=args.seed),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=args.seed),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                                                 random_state=args.seed, n_jobs=-1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])
        validation = cross_val_score(pipe, X_train, y_train, scoring="f1", cv=cv)
        pipe.fit(X_train, y_train)
        prediction = pipe.predict(X_test)
        probability = pipe.predict_proba(X_test)[:, 1]
        scores.append({
            "model": name, "train_rows": len(X_train), "test_rows": len(X_test),
            "cv_f1_mean": float(validation.mean()), "cv_f1_std": float(validation.std()),
            "accuracy": float(accuracy_score(y_test, prediction)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probability)),
        })
        display = ConfusionMatrixDisplay(confusion_matrix(y_test, prediction),
                                         display_labels=["Lower use", "Higher use"])
        display.plot(cmap="Blues", colorbar=False)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(args.output / (name.lower().replace(" ", "_") + "_confusion.png"), dpi=160)
        plt.close()
    pd.DataFrame(scores).to_csv(args.output / "model_comparison.csv", index=False)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "dataset": str(args.data), "target": args.target, "positive": args.positive,
        "excluded_columns": args.exclude, "predictor_columns": X.columns.tolist(),
        "n_samples": len(df), "seed": args.seed, "split": "stratified 80/20",
        "validation": "five-fold stratified CV on training set only",
        "note": "Observational survey: predictive association does not establish causality."
    }, indent=2), encoding="utf-8")
    print(pd.DataFrame(scores).to_string(index=False))
    print(f"Saved reproducible outputs under {args.output}")


if __name__ == "__main__":
    main()
