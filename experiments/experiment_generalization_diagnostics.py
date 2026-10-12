"""Post-hoc diagnostic companion to experiment_robust_generalization.py.

Run from project root:
    python experiments/experiment_generalization_diagnostics.py

Requires the existing experiment_group_aware_validation.py and
experiment_robust_generalization.py in experiments/. Does not modify original
input data, existing predictions, or models. Fits the same partitions again to
capture P(class=1) and inspect errors, covariate support and reference coverage.

IMPORTANT: Results are descriptive on held-out test data, NOT a method for
selecting model thresholds or claiming generalization. Semantic reference
features are built using positive TRAINING labels, not external rules.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment_robust_generalization import (
    MODELS, TUPLE_COLUMNS, strict_semantic_components,
    select_component_holdout, SEED,
)
from experiment_group_aware_validation import (
    load_data, build_and_audit_pairs, CATEGORICAL, NUMERIC,
    RELATIONS, SEMANTIC, make_pipeline,
)


def string_values(series):
    return series.fillna('__MISSING__').astype(str)


def split_statistics(train: pd.DataFrame, test: pd.DataFrame):
    """Unseen-value fractions and total-variation distance per categorical field."""
    rows = []
    for col in CATEGORICAL:
        a, b = string_values(train[col]), string_values(test[col])
        known = set(a)
        freq_a, freq_b = a.value_counts(normalize=True), b.value_counts(normalize=True)
        categories = freq_a.index.union(freq_b.index)
        tv = 0.5 * float((freq_a.reindex(categories, fill_value=0) - freq_b.reindex(categories, fill_value=0)).abs().sum())
        rows.append({'Attribute': col, 'Train_unique': len(known),
                     'Test_unique': b.nunique(), 'Unseen_test_fraction': float((~b.isin(known)).mean()),
                     'Total_variation_distance': tv})
    return rows


def reference_diagnostics(pipeline, X_test):
    trans = pipeline.named_steps['semantic_training_reference']
    enriched = trans.transform(X_test)
    result = {}
    for cols, name in RELATIONS:
        result[name + '_known_fraction'] = float(enriched[name].mean())
    result['semantic_consistency_score_mean'] = float(enriched[SEMANTIC[3]].mean())
    result['all_three_relations_unseen_fraction'] = float((enriched[SEMANTIC[:3]].sum(axis=1) == 0).mean())
    return enriched, result


def main():
    root = HERE.parent
    ap = argparse.ArgumentParser(description='Generalization error diagnostics without threshold tuning')
    ap.add_argument('--dataset', type=Path, default=root/'dataset'/'cross_institutional_dataset (1).xlsx')
    ap.add_argument('--results', type=Path, default=root/'results'/'generalization_diagnostics')
    args = ap.parse_args()

    df = load_data(args.dataset)
    pair = build_and_audit_pairs(df).astype(str).to_numpy()
    X = df[CATEGORICAL + NUMERIC].copy()
    for c in CATEGORICAL:
        X[c] = string_values(X[c])
    y = df.label.astype(int).to_numpy()
    comp, tuples = strict_semantic_components(X, pair)
    tr, te, reason = select_component_holdout(comp, y)
    splits = {}
    warnings = []
    if tr is None:
        warnings.append('Strict unseen-semantic scenario not estimable: ' + reason)
    else:
        splits['strict_unseen_semantic_tuples'] = (tr, te)
    for src, dst in (('UHN', 'PMKM'), ('PMKM', 'UHN')):
        tr_i = np.flatnonzero(X.institution.to_numpy() == src)
        te_i = np.flatnonzero(X.institution.to_numpy() == dst)
        splits[f'transfer_{src}_to_{dst}'] = (tr_i, te_i)

    performance, predictions, errors, drift, coverage = [], [], [], [], []
    scenario_audit = {}
    for scenario, (tr, te) in splits.items():
        train, test = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        overlap = len(set(pair[tr]).intersection(pair[te]))
        if overlap:
            raise RuntimeError(f'{scenario}: {overlap} overlapping genuine-invalid pairs')
        tup_overlap = len(set(tuples[i] for i in tr).intersection(tuples[i] for i in te))
        if scenario == 'strict_unseen_semantic_tuples' and tup_overlap:
            raise RuntimeError('Strict semantic tuples overlap')
        if len(set(ytr)) != 2 or len(set(yte)) != 2:
            warnings.append(f'{scenario}: only one class, skipped')
            continue
        scenario_audit[scenario] = {
            'train_records': len(tr), 'test_records': len(te),
            'train_positive': int(sum(ytr == 1)), 'test_positive': int(sum(yte == 1)),
            'pair_overlap': overlap, 'semantic_tuple_overlap': tup_overlap,
            'unseen_test_tuple_fraction': float(np.mean([tuples[i] not in set(tuples[j] for j in tr) for i in te]))
        }
        for row in split_statistics(train, test):
            drift.append({'Scenario': scenario, **row})

        for exp, semantic in [('A_baseline', False), ('B_training_reference_semantics', True)]:
            for name, proto in MODELS.items():
                model = make_pipeline(clone(proto), semantic)
                model.fit(train, ytr)
                pred = model.predict(test)
                prob = model.predict_proba(test)[:, 1]
                tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
                common = {'Scenario': scenario, 'Experiment': exp, 'Model': name}
                performance.append({**common, 'Accuracy': float(accuracy_score(yte, pred)),
                                    'F1': float(f1_score(yte, pred, zero_division=0)),
                                    'ROC_AUC': float(roc_auc_score(yte, prob)),
                                    'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
                                    'Predicted_positive_fraction': float(np.mean(pred == 1)),
                                    'Mean_score_true_0': float(np.mean(prob[yte == 0])),
                                    'Mean_score_true_1': float(np.mean(prob[yte == 1])),
                                    'Scores_above_0_5_fraction': float(np.mean(prob >= 0.5))})
                if semantic:
                    enriched, diagnostics = reference_diagnostics(model, test)
                    coverage.append({**common, **diagnostics})
                else:
                    enriched = None
                for local_i, global_i in enumerate(te):
                    p = int(pred[local_i]); truth = int(yte[local_i])
                    record = {**common, 'credential_id': str(df.iloc[global_i].credential_id),
                              'Institution': str(df.iloc[global_i].institution),
                              'manipulation_type': str(df.iloc[global_i].manipulation_type),
                              'True_label': truth, 'Predicted_label': p,
                              'Score_positive': float(prob[local_i]),
                              'Error_type': 'TP' if truth == 1 and p == 1 else
                                            'TN' if truth == 0 and p == 0 else
                                            'FP' if truth == 0 else 'FN',
                              'Semantic_tuple': ' | '.join(tuples[global_i])}
                    if enriched is not None:
                        for feature in SEMANTIC:
                            record[feature] = float(enriched.iloc[local_i][feature])
                    predictions.append(record)
                tmp = pd.DataFrame(predictions[-len(te):])
                for manipulation, part in tmp.groupby('manipulation_type', dropna=False):
                    errors.append({**common, 'manipulation_type': manipulation, 'N': len(part),
                                   'TN': int((part.Error_type == 'TN').sum()),
                                   'FP': int((part.Error_type == 'FP').sum()),
                                   'FN': int((part.Error_type == 'FN').sum()),
                                   'TP': int((part.Error_type == 'TP').sum())})
                print(f'{scenario:33} {exp:31} {name:20} F1={performance[-1]["F1"]:.4f} TP={tp} FP={fp}')

    out = args.results
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(performance).to_csv(out/'diagnostic_model_summary.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'diagnostic_predictions_with_scores.csv', index=False)
    pd.DataFrame(errors).to_csv(out/'diagnostic_errors_by_manipulation.csv', index=False)
    pd.DataFrame(drift).to_csv(out/'diagnostic_feature_shift.csv', index=False)
    pd.DataFrame(coverage).to_csv(out/'diagnostic_semantic_reference_coverage.csv', index=False)
    audit = {'record_count': len(df), 'pair_count': len(set(pair)), 'seed': SEED,
             'scenarios': scenario_audit, 'warnings': warnings,
             'limitations': [
                 'Uses class labels from TRAINING to construct semantic reference: not an independent ontology.',
                 'Test-set scores and labels are diagnostics only; thresholds must not be tuned on this test set.',
                 'Strict unseen semantic split is a single available partition, not repeated generalization estimation.',
                 'Feature shift is descriptive, not proof of causation or calibration quality.',
                 'Do not declare unknown semantic tuples invalid merely because they were absent in training.'
             ]}
    (out/'diagnostic_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\nSaved diagnostics to:', out)

if __name__ == '__main__':
    main()
