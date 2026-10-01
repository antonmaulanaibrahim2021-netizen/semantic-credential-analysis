from pathlib import Path
import sys

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from preprocessing.feature_encoding import (
    load_dataset,
    clean_dataset
)


RESULT_PATH = PROJECT_ROOT / "results"


MODELS = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}


def create_group(df):

    df["group_key"] = (
        df["program"].astype(str)
        + "_"
        + df["education_level"].astype(str)
        + "_"
        + df["accreditation_status"].astype(str)
        + "_"
        + df["graduation_predicate"].astype(str)
    )

    return df


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(
            y_test,
            y_prob
        )
    else:
        roc = None


    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "F1-score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "ROC-AUC": roc
    }


def run_group_split():

    df = load_dataset()

    df = clean_dataset(df)

    df = create_group(df)


    X = df.drop(
        columns=[
            "credential_id",
            "label",
            "manipulation_type",
            "group_key"
        ],
        errors="ignore"
    )

    y = df["label"]

    groups = df["group_key"]


    splitter = GroupShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=42
    )


    results = []


    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]


        print("=" * 60)
        print("Fold:", fold)
        print("Train:", X_train.shape)
        print("Test :", X_test.shape)


        for name, model in MODELS.items():

            print("Running:", name)

            metrics = evaluate_model(
                model,
                X_train,
                X_test,
                y_train,
                y_test
            )


            results.append(
                {
                    "Experiment": "Semantic Group Split",
                    "Fold": fold,
                    "Model": name,
                    **metrics
                }
            )


    RESULT_PATH.mkdir(
        exist_ok=True
    )


    result_df = pd.DataFrame(results)


    result_df.to_csv(
        RESULT_PATH / "group_split_model_comparison.csv",
        index=False
    )


    summary_df = (
        result_df
        .groupby("Model")
        [
            [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC-AUC"
            ]
        ]
        .mean()
        .reset_index()
    )


    summary_df.to_csv(
        RESULT_PATH / "group_split_summary.csv",
        index=False
    )


    print("\nExperiment completed.")
    print(
        "Saved:",
        RESULT_PATH / "group_split_model_comparison.csv"
    )
    print(
        "Saved:",
        RESULT_PATH / "group_split_summary.csv"
    )


if __name__ == "__main__":
    run_group_split()
