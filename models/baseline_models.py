"""
BASELINE MACHINE LEARNING MODELS

Project:
Semantic_Credential_Q1

Experiment A:
Original Credential Attribute Model

Models:
- Logistic Regression
- Decision Tree
- Support Vector Machine
- Random Forest

Output:
results/
- baseline_model_comparison.csv
- baseline_classification_report.csv
- baseline_feature_importance.csv
- confusion_matrix/
"""


from pathlib import Path
import sys


import pandas as pd
import numpy as np


from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


import matplotlib.pyplot as plt
import seaborn as sns



# ==========================================================
# PROJECT PATH
# ==========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = (
    PROJECT_ROOT
    / "results"
)


RESULT_PATH.mkdir(
    exist_ok=True
)


CONF_MATRIX_PATH = (
    RESULT_PATH
    / "confusion_matrix"
)


CONF_MATRIX_PATH.mkdir(
    exist_ok=True
)



# ==========================================================
# IMPORT PREPROCESSING
# ==========================================================


sys.path.append(
    str(PROJECT_ROOT)
)


from preprocessing.feature_encoding import (
    encode_baseline_features
)



# ==========================================================
# MODEL CONFIGURATION
# ==========================================================


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




# ==========================================================
# TRAINING FUNCTION
# ==========================================================


def train_baseline_models():


    print("="*70)
    print("BASELINE MACHINE LEARNING EXPERIMENT")
    print("="*70)



    # --------------------------------------
    # Load encoded dataset
    # --------------------------------------

    X, y, encoder = (
        encode_baseline_features()
    )



    print(
        "\nFeature matrix:",
        X.shape
    )


    # --------------------------------------
    # Train test split
    # --------------------------------------

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

    )


    print(
        "\nTraining:",
        X_train.shape
    )

    print(
        "Testing:",
        X_test.shape
    )



    results = []

    reports = []



    # Cross validation

    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42

    )



    # --------------------------------------
    # Training loop
    # --------------------------------------

    for name, model in MODELS.items():


        print("\nTraining:", name)


        model.fit(
            X_train,
            y_train
        )


        y_pred = model.predict(
            X_test
        )


        y_prob = model.predict_proba(
            X_test
        )[:,1]



        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        precision = precision_score(
            y_test,
            y_pred
        )


        recall = recall_score(
            y_test,
            y_pred
        )


        f1 = f1_score(
            y_test,
            y_pred
        )


        auc = roc_auc_score(
            y_test,
            y_prob
        )



        cv_score = cross_val_score(

            model,

            X,

            y,

            cv=cv,

            scoring="f1"

        )



        results.append(

            {

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1-score": f1,

            "ROC-AUC": auc,

            "CV_F1_mean": cv_score.mean(),

            "CV_F1_std": cv_score.std()

            }

        )



        # classification report

        report = classification_report(

            y_test,

            y_pred,

            output_dict=True

        )


        report_df = pd.DataFrame(
            report
        ).transpose()


        report_df["Model"] = name


        reports.append(
            report_df
        )



        # confusion matrix

        cm = confusion_matrix(

            y_test,

            y_pred

        )


        plt.figure(
            figsize=(5,4)
        )


        sns.heatmap(

            cm,

            annot=True,

            fmt="d"

        )


        plt.title(
            f"Confusion Matrix - {name}"
        )


        plt.xlabel(
            "Predicted"
        )


        plt.ylabel(
            "Actual"
        )


        plt.tight_layout()


        plt.savefig(

            CONF_MATRIX_PATH
            /
            f"{name.replace(' ','_')}.png",

            dpi=300

        )


        plt.close()



    # ==================================================
    # SAVE RESULTS
    # ==================================================


    result_df = pd.DataFrame(
        results
    )


    result_df = result_df.sort_values(

        by="F1-score",

        ascending=False

    )


    result_df.to_csv(

        RESULT_PATH
        /
        "baseline_model_comparison.csv",

        index=False

    )



    report_final = pd.concat(
        reports
    )


    report_final.to_csv(

        RESULT_PATH
        /
        "baseline_classification_report.csv",

        index=False

    )



    print("\nRESULT SUMMARY")
    print(result_df)


    print(
        "\nSaved:"
    )


    print(
        RESULT_PATH
        /
        "baseline_model_comparison.csv"
    )



    return result_df




# ==========================================================
# MAIN
# ==========================================================


if __name__ == "__main__":


    train_baseline_models()