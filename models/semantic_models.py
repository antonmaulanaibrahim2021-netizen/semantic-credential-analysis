"""
SEMANTIC KNOWLEDGE ENHANCED MACHINE LEARNING MODELS

Project:
Semantic_Credential_Q1


Experiment B:
Credential Attributes + Semantic Knowledge Features


Models:
- Logistic Regression
- Decision Tree
- SVM
- Random Forest


Output:
results/
- semantic_model_comparison.csv
- semantic_classification_report.csv
- semantic_feature_importance.csv
- semantic_confusion_matrix/
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


from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


from sklearn.pipeline import Pipeline


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
from sklearn.impute import SimpleImputer



import matplotlib.pyplot as plt
import seaborn as sns



# ==========================================================
# PATH
# ==========================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = (
    PROJECT_ROOT /
    "results"
)


CONFUSION_PATH = (
    RESULT_PATH /
    "semantic_confusion_matrix"
)


CONFUSION_PATH.mkdir(
    exist_ok=True
)



# ==========================================================
# IMPORT SEMANTIC FEATURE ENGINEERING
# ==========================================================


sys.path.append(
    str(PROJECT_ROOT)
)


from preprocessing.semantic_feature_engineering import (
    prepare_semantic_dataset
)



# ==========================================================
# CONFIGURATION
# ==========================================================


TARGET = "label"


DROP_COLUMNS = [

    "credential_id",

    "label",

    "manipulation_type",

    "institution_program_key",

    "program_education_key",

    "program_accreditation_key"

]



NUMERIC_FEATURES = [

    "gpa",

    "semantic_consistency_score",

    "institution_program_valid",

    "program_education_valid",

    "program_accreditation_valid"

]



CATEGORICAL_FEATURES = [

    "institution",

    "program",

    "education_level",

    "accreditation_status",

    "graduation_predicate",

    "graduation_date",

    "faculty_program_name"

]



# ==========================================================
# MODELS
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
# PREPROCESSOR
# ==========================================================


def build_preprocessor():


    categorical_pipeline = Pipeline(

        steps=[

            (

                "onehot",

                OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False

                )

            )

        ]

    )



    numerical_pipeline = Pipeline(

        steps=[

            (

                "scaler",

                StandardScaler()

            )

        ]

    )



    processor = ColumnTransformer(

        transformers=[


            (

                "categorical",

                categorical_pipeline,

                CATEGORICAL_FEATURES

            ),


            (

                "numeric",

                numerical_pipeline,

                NUMERIC_FEATURES

            )

        ]

    )


    return processor




# ==========================================================
# FEATURE PREPARATION
# ==========================================================


def prepare_features(df):


    X = df.drop(

        columns=DROP_COLUMNS,

        errors="ignore"

    )


    y = df[TARGET]


    return X, y




# ==========================================================
# TRAINING
# ==========================================================


def train_semantic_models():


    print("="*70)

    print(
        "SEMANTIC KNOWLEDGE ENHANCED MODEL"
    )

    print("="*70)



    # Load semantic dataset

    df = prepare_semantic_dataset()



    print(

        "\nDataset:",

        df.shape

    )



    X, y = prepare_features(df)



    print(

        "Raw features:",

        X.shape

    )



    processor = build_preprocessor()



    X_encoded = processor.fit_transform(X)



    feature_names = (

        processor

        .get_feature_names_out()

    )



    X_encoded = pd.DataFrame(

        X_encoded,

        columns=feature_names

    )



    print(

        "Encoded features:",

        X_encoded.shape

    )



    # split


    X_train, X_test, y_train, y_test = train_test_split(

        X_encoded,

        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )
    # Handle missing values before model training
    imputer = SimpleImputer(
        strategy="most_frequent"
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns
    )
    # untuk cross validation    
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_encoded),
        columns=X_encoded.columns
    )
    print(
        "Remaining NaN:",
        X_train.isna().sum().sum()
    )





    results = []

    reports = []



    cv = StratifiedKFold(

        n_splits=5,

        shuffle=True,

        random_state=42

    )



    # training loop


    for name, model in MODELS.items():


        print(

            "\nTraining:",

            name

        )


        model.fit(

            X_train,

            y_train

        )



        prediction = model.predict(

            X_test

        )


        probability = model.predict_proba(

            X_test

        )[:,1]



        accuracy = accuracy_score(

            y_test,

            prediction

        )


        precision = precision_score(

            y_test,

            prediction

        )


        recall = recall_score(

            y_test,

            prediction

        )


        f1 = f1_score(

            y_test,

            prediction

        )


        auc = roc_auc_score(

            y_test,

            probability

        )



        cv_score = cross_val_score(

            model,

            X_imputed,

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



        # report


        report = pd.DataFrame(

            classification_report(

                y_test,

                prediction,

                output_dict=True

            )

        ).transpose()



        report["Model"] = name


        reports.append(report)



        # confusion matrix


        cm = confusion_matrix(

            y_test,

            prediction

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

            f"Semantic CM - {name}"

        )


        plt.xlabel(

            "Prediction"

        )


        plt.ylabel(

            "Actual"

        )


        plt.tight_layout()


        plt.savefig(

            CONFUSION_PATH /

            f"{name.replace(' ','_')}.png",

            dpi=300

        )


        plt.close()



        # feature importance

        if hasattr(

            model,

            "feature_importances_"

        ):


            importance = pd.DataFrame(

                {

                    "Feature":

                    feature_names,


                    "Importance":

                    model.feature_importances_

                }

            )


            importance["Model"] = name


            importance.to_csv(

                RESULT_PATH /

                f"semantic_feature_importance_{name.replace(' ','_')}.csv",

                index=False

            )



    # SAVE RESULTS


    result_df = pd.DataFrame(results)


    result_df = result_df.sort_values(

        by="F1-score",

        ascending=False

    )


    result_df.to_csv(

        RESULT_PATH /

        "semantic_model_comparison.csv",

        index=False

    )



    report_df = pd.concat(

        reports

    )


    report_df.to_csv(

        RESULT_PATH /

        "semantic_classification_report.csv",

        index=False

    )



    print("\nRESULT")

    print(result_df)



    print(

        "\nSaved semantic model results."

    )



    return result_df





# ==========================================================
# MAIN
# ==========================================================


if __name__ == "__main__":


    train_semantic_models()