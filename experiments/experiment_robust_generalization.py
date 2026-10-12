"""Robust credential generalization audit: unseen semantic tuples and institution transfer.

Place in Semantic_Credential_Q1/experiments alongside experiment_group_aware_validation.py.
Run from project root: python experiments/experiment_robust_generalization.py
Requires: numpy, pandas, openpyxl, scikit-learn>=1.3.

Training-reference semantic indicators are SUPERVISED lookup features constructed from
positive TRAINING examples only. They are NOT externally validated ontology rules.
Outputs are saved separately under results/robust_generalization.
No dataset or previous result file is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_group_aware_validation import (  # shared, audited definitions
    load_data, build_and_audit_pairs, CATEGORICAL, NUMERIC, make_pipeline
)

SEED = 42
TUPLE_COLUMNS = ["program", "education_level", "accreditation_status"]
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "SVM": SVC(probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
}


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def strict_semantic_components(X: pd.DataFrame, pairs: np.ndarray):
    """Connected components enforce BOTH pair isolation and tuple disjointness.

    Any pair or semantic tuple connecting two rows forces the rows into one component.
    This prevents test semantic tuples appearing in train, even for manipulated twins.
    """
    uf = UnionFind()
    tokens = [tuple(map(str, row)) for row in X[TUPLE_COLUMNS].to_numpy()]
    for pair, tup in zip(pairs, tokens):
        uf.union(("pair", str(pair)), ("tuple", tup))
    components = np.asarray([str(uf.find(("pair", str(p)))) for p in pairs])
    return components, tokens


def select_component_holdout(components, y, fraction=0.2):
    """Attempt disjoint holdout; return diagnostic if connected graph prevents it."""
    uniq, inverse = np.unique(components, return_inverse=True)
    per_group = np.bincount(inverse)
    if len(uniq) < 2:
        return None, None, f"Only {len(uniq)} connected component; strict split impossible"
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(uniq))
    target = fraction * len(components)
    selected, count = set(), 0
    # Greedy selection without taking nearly all records.
    for i in order:
        n = int(per_group[i])
        if count + n <= max(target * 1.25, target + 2) and count + n < len(components):
            selected.add(int(i))
            count += n
        if count >= target * 0.8:
            break
    if not selected:
        return None, None, "No component small enough for an approximately 20% test partition"
    test = np.flatnonzero(np.isin(inverse, list(selected)))
    train = np.flatnonzero(~np.isin(inverse, list(selected)))
    if len(train) == 0 or len(test) == 0:
        return None, None, "Empty train/test subset"
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        return None, None, "Strict split lacks both labels in training or testing"
    return train, test, "ok"


def score(model, X_test, y_test):
    pred = model.predict(X_test)
    result = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, pred, zero_division=0)),
        "TN": int(confusion_matrix(y_test, pred, labels=[0, 1])[0, 0]),
        "FP": int(confusion_matrix(y_test, pred, labels=[0, 1])[0, 1]),
        "FN": int(confusion_matrix(y_test, pred, labels=[0, 1])[1, 0]),
        "TP": int(confusion_matrix(y_test, pred, labels=[0, 1])[1, 1]),
    }
    if len(np.unique(y_test)) == 2:
        try:
            result["ROC-AUC"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        except (AttributeError, ValueError):
            result["ROC-AUC"] = float("nan")
    else:
        result["ROC-AUC"] = float("nan")
    return pred, result


def run_partition(name, tr, te, X, y, pairs, tuples, rows, predictions, audit):
    if len(tr) == 0 or len(te) == 0 or len(np.unique(y[tr])) < 2:
        audit[name] = {"status": "not_estimable", "reason": "empty partition or one-class training"}
        return
    pair_overlap = set(pairs[tr]) & set(pairs[te])
    tuple_overlap = set(tuples[i] for i in tr) & set(tuples[i] for i in te)
    if pair_overlap:
        raise AssertionError(f"Pair overlap in {name}: {len(pair_overlap)}")
    if name == "strict_unseen_semantic_tuples" and tuple_overlap:
        raise AssertionError(f"Semantic tuple overlap in strict split: {len(tuple_overlap)}")
    audit[name] = {
        "status": "evaluated", "train_records": int(len(tr)), "test_records": int(len(te)),
        "train_label_counts": {str(v): int((y[tr] == v).sum()) for v in [0, 1]},
        "test_label_counts": {str(v): int((y[te] == v).sum()) for v in [0, 1]},
        "pair_overlap": len(pair_overlap), "semantic_tuple_overlap": len(tuple_overlap),
        "unseen_test_tuple_fraction": float(np.mean([tuples[i] not in set(tuples[j] for j in tr) for i in te])),
    }
    for experiment, semantic in (("A_baseline", False), ("B_training_reference_semantics", True)):
        for model_name, model_proto in MODELS.items():
            model = make_pipeline(clone(model_proto), semantic)
            model.fit(X.iloc[tr], y[tr])
            pred, met = score(model, X.iloc[te], y[te])
            rows.append({"Scenario": name, "Experiment": experiment, "Model": model_name, **met,
                         "Train_N": len(tr), "Test_N": len(te)})
            for i, yp in zip(te, pred):
                predictions.append({"Scenario": name, "Experiment": experiment, "Model": model_name,
                                    "credential_id": str(X.index[i]), "True_label": int(y[i]),
                                    "Predicted_label": int(yp), "Institution": X.iloc[i]["institution"],
                                    "Semantic_tuple": " | ".join(tuples[i])})
            print(f"{name:<32} {experiment:<31} {model_name:<20} F1={met['F1-score']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    root = HERE.parent
    ap.add_argument("--dataset", type=Path, default=root / "dataset" / "cross_institutional_dataset (1).xlsx")
    ap.add_argument("--results", type=Path, default=root / "results" / "robust_generalization")
    args = ap.parse_args()
    df = load_data(args.dataset)
    pairs = build_and_audit_pairs(df).astype(str).to_numpy()
    X = df[CATEGORICAL + NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna("__MISSING__").astype(str)
    X.index = df.credential_id.astype(str).to_numpy()  # traceable predictions, excluded as a feature
    y = df.label.astype(int).to_numpy()
    components, tuples = strict_semantic_components(X, pairs)
    audit = {"records": len(df), "pairs": len(set(pairs)),
             "distinct_semantic_tuples": len(set(tuples)),
             "semantic_graph_components": len(set(components)),
             "semantic_definition": TUPLE_COLUMNS,
             "feature_reference_policy": "Only genuine TRAINING records, derived using TRAINING labels; not independent knowledge rules",
             "cautions": ["Do not treat unseen training-derived tuple as inherently invalid.",
                          "Cross-institution transfer may test on unknown programs and cannot prove deployment readiness.",
                          "Strict unseen-tuple test is NOT estimated if pair/tuple graph cannot support a valid split."]}
    rows, predictions = [], []
    tr, te, reason = select_component_holdout(components, y)
    if tr is None:
        audit["strict_unseen_semantic_tuples"] = {"status": "not_estimable", "reason": reason}
        print("Strict semantic-group split not estimable:", reason)
    else:
        run_partition("strict_unseen_semantic_tuples", tr, te, X, y, pairs, tuples, rows, predictions, audit)
    for src, dst in (("UHN", "PMKM"), ("PMKM", "UHN")):
        tr = np.flatnonzero(X.institution.to_numpy() == src)
        te = np.flatnonzero(X.institution.to_numpy() == dst)
        run_partition(f"transfer_{src}_to_{dst}", tr, te, X, y, pairs, tuples, rows, predictions, audit)
    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "robust_generalization_comparison.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "robust_generalization_predictions.csv", index=False)
    audit["evaluated_scenarios"] = sorted(set(r["Scenario"] for r in rows))
    (out / "robust_generalization_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved output to:", out)
    print("Scenarios evaluated:", audit["evaluated_scenarios"])


if __name__ == "__main__":
    main()
