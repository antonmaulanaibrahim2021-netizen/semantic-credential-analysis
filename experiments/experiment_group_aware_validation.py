"""Leakage-aware paired credential validation (Experiment A vs B).

Run from the existing project: python experiments/experiment_group_aware_validation.py
Dependencies: pandas, openpyxl, numpy, scikit-learn >= 1.2.

IMPORTANT: semantic rules are learned from *training* records with label=1,
inside each split/fold. This is SUPERVISED training-derived reference learning,
NOT independent externally validated academic rules. The results should not be
claimed to validate an externally sourced semantic knowledge base.
Original data and existing results are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

SEED = 42
CATEGORICAL = ["institution", "program", "education_level", "accreditation_status",
               "graduation_predicate", "graduation_date", "faculty_program_name"]
NUMERIC = ["gpa"]
SEMANTIC = ["institution_program_valid", "program_education_valid",
            "program_accreditation_valid", "semantic_consistency_score"]
RELATIONS = [(("institution", "program"), SEMANTIC[0]),
             (("program", "education_level"), SEMANTIC[1]),
             (("program", "accreditation_status"), SEMANTIC[2])]
ALLOWED_CHANGE = {"accreditation_swap": "accreditation_status",
                  "cross_institution_program": "program", "education_level_swap": "education_level"}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Combined_Labeled", header=2)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    needed = set(CATEGORICAL + NUMERIC + ["credential_id", "label", "manipulation_type"])
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if df["credential_id"].duplicated().any():
        raise ValueError("Duplicated credential_id: aborting")
    if set(df["label"].dropna().unique()) != {0, 1}:
        raise ValueError("Labels must be binary 0/1")
    return df


def build_and_audit_pairs(df: pd.DataFrame) -> pd.Series:
    """Dataset-specific pairing. Validate each pair against manipulation type.

    UHN: UHN-INV-NNNN -> UHN-NNNN.
    PMKM: PMKM-INV-NNNN -> PMKM-(NNNN+746); confirmed on supplied data,
    but never silently assumed: each record is validated below.
    """
    ids = set(df["credential_id"].astype(str))
    genuine = df[df.label == 1].set_index("credential_id")
    invalid = df[df.label == 0]
    mapping = {}
    for _, rec in invalid.iterrows():
        invalid_id = str(rec.credential_id)
        m = re.fullmatch(r"(UHN|PMKM)-INV-(\d+)", invalid_id)
        if not m:
            raise ValueError(f"Unexpected invalid ID format: {invalid_id}")
        institution, number = m.groups()
        original_number = int(number) + (746 if institution == "PMKM" else 0)
        original_id = f"{institution}-{original_number:04d}"
        if original_id not in ids or original_id not in genuine.index:
            raise ValueError(f"Pair not found for {invalid_id} => {original_id}")
        base = genuine.loc[original_id]
        expected = ALLOWED_CHANGE.get(str(rec.manipulation_type))
        if expected is None:
            raise ValueError(f"Unknown manipulation type for {invalid_id}")
        differing = []
        for col in CATEGORICAL + NUMERIC:
            a, b = base[col], rec[col]
            if (pd.isna(a) and pd.isna(b)) or a == b:
                continue
            differing.append(col)
        if differing != [expected]:
            raise ValueError(f"Pair audit failed: {invalid_id} => {original_id}; "
                             f"changed={differing}, expected={[expected]}")
        mapping[invalid_id] = original_id
    if len(set(mapping.values())) != len(invalid):
        raise ValueError("Some invalid records map to the same genuine record")
    pair_ids = pd.Series(index=df.index, dtype="object")
    for idx, row in df.iterrows():
        credential_id = str(row.credential_id)
        pair_ids.loc[idx] = credential_id if row.label == 1 else mapping[credential_id]
    group = pd.DataFrame({"pair": pair_ids, "label": df.label}).groupby("pair").label.agg(["count", "nunique"])
    if not ((group["count"] == 2) & (group["nunique"] == 2)).all():
        raise ValueError("Not all pairs contain exactly one genuine and one invalid record")
    return pair_ids


class TrainingReferenceSemantics(BaseEstimator, TransformerMixin):
    """Derive semantic lookup from genuine TRAINING records, never testing rows.

    Uses training y intentionally; not a fully independent ontology.
    All unknown combinations receive 0 (unknown/not in reference), which
    MUST NOT be interpreted as proof of invalidity in deployment.
    """
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("TrainingReferenceSemantics requires training labels in fit")
        data = X.copy().reset_index(drop=True)
        labels = np.asarray(y)
        if len(labels) != len(data):
            raise ValueError("Label alignment failed")
        normal = data.loc[labels == 1]
        if normal.empty:
            raise ValueError("No positive reference records in training set")
        self.reference_ = {}
        for cols, _ in RELATIONS:
            self.reference_[cols] = set(map(tuple, normal[list(cols)].fillna("__MISSING__").astype(str).to_numpy()))
        return self

    def transform(self, X):
        result = X.copy()
        for cols, output in RELATIONS:
            values = map(tuple, result[list(cols)].fillna("__MISSING__").astype(str).to_numpy())
            result[output] = [int(t in self.reference_[cols]) for t in values]
        result[SEMANTIC[3]] = result[SEMANTIC[:3]].mean(axis=1)
        return result


def preprocessing(semantic: bool):
    cat = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                    ("scaler", StandardScaler())])
    return ColumnTransformer([("cat", cat, CATEGORICAL),
                              ("num", num, NUMERIC + (SEMANTIC if semantic else []))])


def make_pipeline(model, semantic: bool):
    steps = []
    if semantic:
        steps.append(("semantic_training_reference", TrainingReferenceSemantics()))
    steps += [("preprocess", preprocessing(semantic)), ("model", clone(model))]
    return Pipeline(steps)


def metrics(model, X_test, y_test):
    pred = model.predict(X_test)
    result = {"Accuracy": accuracy_score(y_test, pred),
              "Precision": precision_score(y_test, pred, zero_division=0),
              "Recall": recall_score(y_test, pred, zero_division=0),
              "F1-score": f1_score(y_test, pred, zero_division=0)}
    if len(np.unique(y_test)) == 2:
        prob = model.predict_proba(X_test)[:, 1]
        result["ROC-AUC"] = roc_auc_score(y_test, prob)
    else:
        result["ROC-AUC"] = np.nan
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--results", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dataset_path = args.dataset or (root / "dataset" / "cross_institutional_dataset (1).xlsx")
    out = args.results or (root / "results" / "group_aware_validation")
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(dataset_path)
    pair_id = build_and_audit_pairs(data)
    X = data[CATEGORICAL + NUMERIC].copy()
    for col in CATEGORICAL:
        X[col] = X[col].fillna("__MISSING__").astype(str)
    y = data.label.astype(int).to_numpy()
    group = pair_id.to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=group))
    assert not set(group[train_idx]).intersection(set(group[test_idx]))
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {"Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
              "Decision Tree": DecisionTreeClassifier(random_state=SEED),
              "SVM": SVC(probability=True, random_state=SEED),
              "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)}
    rows, folds, importances = [], [], []
    for exp, use_semantic in [("A_baseline", False), ("B_training_reference_semantics", True)]:
        for name, prototype in models.items():
            estimator = make_pipeline(prototype, use_semantic)
            estimator.fit(X.iloc[train_idx], y[train_idx])
            row = {"Experiment": exp, "Model": name, **metrics(estimator, X.iloc[test_idx], y[test_idx])}
            cv_scores = []
            # CV only within the training subset, with pair-group isolation in each fold
            Xt, yt, gt = X.iloc[train_idx], y[train_idx], group[train_idx]
            for fold, (tr, va) in enumerate(cv.split(Xt, yt, groups=gt), start=1):
                assert not set(gt[tr]).intersection(set(gt[va]))
                fitted = make_pipeline(prototype, use_semantic)
                fitted.fit(Xt.iloc[tr], yt[tr])
                scores = metrics(fitted, Xt.iloc[va], yt[va])
                folds.append({"Experiment": exp, "Model": name, "Fold": fold, **scores})
                cv_scores.append(scores["F1-score"])
            row["CV_F1_mean"] = float(np.mean(cv_scores))
            row["CV_F1_std"] = float(np.std(cv_scores))
            rows.append(row)
            if use_semantic and name == "Random Forest":
                names = estimator.named_steps["preprocess"].get_feature_names_out()
                weights = estimator.named_steps["model"].feature_importances_
                importances = sorted(zip(names, weights), key=lambda x: -x[1])
            print(exp, name, f"F1={row['F1-score']:.4f}", f"CV={row['CV_F1_mean']:.4f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "group_aware_model_comparison.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "group_aware_cv_folds.csv", index=False)
    pd.DataFrame(importances, columns=["Feature", "Importance"]).to_csv(
        out / "group_aware_semantic_RF_importance.csv", index=False)
    assignments = data[["credential_id", "institution", "label"]].copy()
    assignments["pair_id"] = group
    assignments["split"] = "train"
    assignments.loc[assignments.index[test_idx], "split"] = "test"
    assignments.to_csv(out / "pair_split_audit.csv", index=False)
    info = {"records": len(data), "pairs": len(np.unique(group)),
            "train_records": len(train_idx), "test_records": len(test_idx),
            "train_pairs": len(set(group[train_idx])), "test_pairs": len(set(group[test_idx])),
            "pair_overlap": len(set(group[train_idx]) & set(group[test_idx])),
            "semantic_reference": "learned from genuine training records using train labels; not external domain rules",
            "cv_policy": "five-fold GroupKFold within training only; preprocessing fit inside folds",
            "warnings": ["Training-derived references are not externally validated semantic knowledge.",
                         "This is a paired holdout test, not institution transfer or fully novel-semantic-group validation."]}
    (out / "validation_audit.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved to: {out}\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
