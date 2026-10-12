"""Unknown-aware semantic credential assessment (exploratory, leakage-safe).

Run from project root:
  python experiments/experiment_semantic_unknown_handling.py

Requires sibling files experiment_group_aware_validation.py and
experiment_robust_generalization.py and numpy, pandas, openpyxl, scikit-learn.

Compares A (baseline), B (legacy reference membership), C (explicit UNKNOWN)
on identical pre-specified partitions. A semantic relationship seen in positive
TRAINING rows is 'observed_positive'; one not seen is 'unknown'. There are NO
verified-invalid academic rules here, hence 'invalid' IS NOT assigned.
This is not an independently validated ontology. NEVER use test labels to form
reference sets, tune thresholds, or choose a winning model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_group_aware_validation import (
    load_data, build_and_audit_pairs, CATEGORICAL, NUMERIC, RELATIONS,
    make_pipeline,
)
from experiment_robust_generalization import (
    strict_semantic_components, select_component_holdout, TUPLE_COLUMNS,
)

SEED = 42
RELATION_STATUS = [f"status_{name}" for _, name in RELATIONS]
RELATION_UNKNOWN = [f"unknown_{name}" for _, name in RELATIONS]
C_NUMERIC = ["semantic_observed_fraction", "semantic_unknown_fraction"]


class ExplicitUnknownSemantics(BaseEstimator, TransformerMixin):
    """Training-derived support; unknown != invalid (no observed invalid rules)."""

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Training labels required to build train-only references")
        labels = np.asarray(y)
        if len(X) != len(labels):
            raise ValueError("X/y length mismatch")
        positive = X.iloc[np.flatnonzero(labels == 1)]
        if positive.empty:
            raise ValueError("No positive examples in training")
        self.reference_ = {
            cols: set(map(tuple, positive[list(cols)].fillna("__MISSING__").astype(str).to_numpy()))
            for cols, _ in RELATIONS
        }
        return self

    def transform(self, X):
        result = X.copy()
        for (cols, _), status, unknown in zip(RELATIONS, RELATION_STATUS, RELATION_UNKNOWN):
            keys = map(tuple, result[list(cols)].fillna("__MISSING__").astype(str).to_numpy())
            known = np.asarray([key in self.reference_[cols] for key in keys], dtype=bool)
            result[status] = np.where(known, "observed_positive", "unknown")
            result[unknown] = (~known).astype(float)
        result[C_NUMERIC[1]] = result[RELATION_UNKNOWN].mean(axis=1)
        result[C_NUMERIC[0]] = 1.0 - result[C_NUMERIC[1]]
        return result


def unknown_pipeline(prototype):
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    processor = ColumnTransformer([
        ("categorical", cat, CATEGORICAL + RELATION_STATUS),
        ("numeric", num, NUMERIC + RELATION_UNKNOWN + C_NUMERIC),
    ])
    return Pipeline([
        ("semantic_explicit_unknown", ExplicitUnknownSemantics()),
        ("preprocess", processor),
        ("model", clone(prototype)),
    ])


def make_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "SVM": SVC(probability=True, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    }


def metrics(estimator, X_test, y_test):
    pred = estimator.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    try:
        score = estimator.predict_proba(X_test)[:, 1]
    except (AttributeError, ValueError):
        score = np.full(len(pred), np.nan)
    auc = (float(roc_auc_score(y_test, score))
           if len(np.unique(y_test)) == 2 and np.isfinite(score).all() else None)
    return pred, score, {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "ROC_AUC": auc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    ap = argparse.ArgumentParser(description="Explicit-unknown exploratory semantic evaluation")
    root = HERE.parent
    ap.add_argument("--dataset", type=Path, default=root / "dataset" / "cross_institutional_dataset (1).xlsx")
    ap.add_argument("--results", type=Path, default=root / "results" / "semantic_unknown_handling")
    args = ap.parse_args()

    data = load_data(args.dataset)
    pairs = build_and_audit_pairs(data).astype(str).to_numpy()
    X = data[CATEGORICAL + NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    y = data.label.astype(int).to_numpy()

    scenarios = {}
    # Extra matched paired holdout for the comparison to the previous group-aware run.
    grp = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    scenarios["paired_holdout"] = next(grp.split(X, y, groups=pairs))

    components, tuples = strict_semantic_components(X, pairs)
    tr, te, reason = select_component_holdout(components, y)
    skipped = []
    if tr is None:
        skipped.append({"scenario": "strict_unseen_semantic_tuples", "reason": reason})
    else:
        scenarios["strict_unseen_semantic_tuples"] = (tr, te)
    for source, target in (("UHN", "PMKM"), ("PMKM", "UHN")):
        scenarios[f"transfer_{source}_to_{target}"] = (
            np.flatnonzero(X.institution.to_numpy() == source),
            np.flatnonzero(X.institution.to_numpy() == target),
        )

    experiments = {
        "A_baseline": lambda m: make_pipeline(m, False),
        "B_legacy_reference": lambda m: make_pipeline(m, True),
        "C_explicit_unknown": unknown_pipeline,
    }

    report = {
        "records": int(len(data)), "pairs": int(len(set(pairs))),
        "experiment_C_statuses": ["observed_positive", "unknown"],
        "invalid_status_available": False,
        "invalid_status_explanation": (
            "No independently verified invalid knowledge relations are available; "
            "negative class examples alone do not establish that any individual "
            "relation is intrinsically invalid."
        ),
        "semantic_reference": "positive training labels only, not external ontology",
        "threshold_selection": "fixed model defaults; no test-based optimization",
        "split_audit": {}, "skipped": skipped,
        "limitations": [
            "C makes unknown explicit but remains a training-derived supervised lookup.",
            "C is a representation ablation, not an independent academic rule verifier.",
            "Strict semantic holdout may be a single partition; no repeated confidence claims.",
            "Compare paired scenarios within each scenario only; do not tune on test set.",
        ],
    }
    comparison, predictions, coverage = [], [], []
    for scenario, (train_idx, test_idx) in scenarios.items():
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Empty subset in {scenario}")
        pair_overlap = len(set(pairs[train_idx]) & set(pairs[test_idx]))
        if pair_overlap:
            raise AssertionError(f"Pair leakage in {scenario}: {pair_overlap}")
        train_tuples = {tuples[i] for i in train_idx}
        tuple_overlap = len(train_tuples & {tuples[i] for i in test_idx})
        if scenario == "strict_unseen_semantic_tuples" and tuple_overlap:
            raise AssertionError("Unseen tuple scenario has tuple overlap")
        if set(y[train_idx]) != {0, 1} or set(y[test_idx]) != {0, 1}:
            raise ValueError(f"Both labels required in each subset: {scenario}")
        report["split_audit"][scenario] = {
            "train_records": int(len(train_idx)), "test_records": int(len(test_idx)),
            "train_pairs": int(len(set(pairs[train_idx]))),
            "test_pairs": int(len(set(pairs[test_idx]))),
            "pair_overlap": pair_overlap, "semantic_tuple_overlap": tuple_overlap,
            "unseen_test_tuple_fraction": float(np.mean([tuples[i] not in train_tuples for i in test_idx])),
            "train_positive": int(np.sum(y[train_idx] == 1)),
            "test_positive": int(np.sum(y[test_idx] == 1)),
        }
        x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        for exp, factory in experiments.items():
            for model_name, prototype in make_models().items():
                fitted = factory(prototype)
                fitted.fit(x_train, y_train)
                pred, scores, met = metrics(fitted, x_test, y_test)
                comparison.append({"Scenario": scenario, "Experiment": exp,
                                   "Model": model_name, "Train_N": len(train_idx),
                                   "Test_N": len(test_idx), **met})
                common = {"Scenario": scenario, "Experiment": exp, "Model": model_name}
                if exp == "C_explicit_unknown":
                    enriched = fitted.named_steps["semantic_explicit_unknown"].transform(x_test)
                    for rel, name in zip(RELATION_UNKNOWN, RELATION_STATUS):
                        coverage.append({**common, "Relation": name,
                                         "Unknown_fraction": float(enriched[rel].mean()),
                                         "Observed_positive_fraction": float(1.0-enriched[rel].mean())})
                for loc, original_i in enumerate(test_idx):
                    record = {**common, "credential_id": str(data.iloc[original_i].credential_id),
                              "Institution": str(data.iloc[original_i].institution),
                              "manipulation_type": str(data.iloc[original_i].manipulation_type),
                              "Pair_ID": str(pairs[original_i]), "True_label": int(y_test[loc]),
                              "Predicted_label": int(pred[loc]),
                              "Score_positive": float(scores[loc]) if np.isfinite(scores[loc]) else None}
                    if exp == "C_explicit_unknown":
                        for col in RELATION_STATUS + RELATION_UNKNOWN + C_NUMERIC:
                            record[col] = enriched.iloc[loc][col]
                    predictions.append(record)
                print(f"{scenario:32} {exp:22} {model_name:20} F1={met['F1']:.4f} "
                      f"TP={met['TP']} FP={met['FP']}")

    args.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison).to_csv(args.results / "unknown_handling_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.results / "unknown_handling_predictions.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.results / "unknown_handling_reference_coverage.csv", index=False)
    (args.results / "unknown_handling_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved results to: {args.results}")
    print("WARNING: 'unknown' means unsupported by TRAINING lookup, not academically invalid.")


if __name__ == "__main__":
    main()
