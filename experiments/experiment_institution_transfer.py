from pathlib import Path
import sys

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



# ======================================
# PROJECT PATH
# ======================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from preprocessing.feature_encoding import (
    load_dataset,
    clean_dataset
)


RESULT_PATH = PROJECT_ROOT / "results"



# ======================================
# MODELS
# ======================================

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



# ======================================
# PREPROCESSOR
# ======================================

def create_preprocessor(X):


    categorical_features = [

        "institution",

        "program",

        "education_level",

        "accreditation_status",

        "faculty_program_name"

    ]


    numeric_features = [

        "gpa"

    ]


    categorical_features = [

        c for c in categorical_features

        if c in X.columns

    ]


    numeric_features = [

        c for c in numeric_features

        if c in X.columns

    ]



    return ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical_features

            ),


            (
                "numeric",

                "passthrough",

                numeric_features

            )

        ]

    )



# ======================================
# EVALUATION
# ======================================

def evaluate_model(

        model,

        X_train,

        X_test,

        y_train,

        y_test

):


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                create_preprocessor(X_train)

            ),

            (
                "model",

                model

            )

        ]

    )


    pipeline.fit(

        X_train,

        y_train

    )


    y_pred = pipeline.predict(

        X_test

    )


    if hasattr(

        pipeline,

        "predict_proba"

    ):

        y_prob = pipeline.predict_proba(

            X_test

        )[:,1]


        roc = roc_auc_score(

            y_test,

            y_prob

        )


    else:

        roc = None



    return {


        "Accuracy":

            accuracy_score(

                y_test,

                y_pred

            ),



        "Precision":

            precision_score(

                y_test,

                y_pred,

                zero_division=0

            ),



        "Recall":

            recall_score(

                y_test,

                y_pred,

                zero_division=0

            ),



        "F1-score":

            f1_score(

                y_test,

                y_pred,

                zero_division=0

            ),



        "ROC-AUC":

            roc

    }



# ======================================
# INSTITUTION TRANSFER EXPERIMENT
# ======================================

def run_institution_transfer():


    print("Loading dataset...")


    df = load_dataset()


    df = clean_dataset(df)



    print("\nInstitution distribution:")

    print(

        df["institution"]

        .value_counts()

    )



    X = df.drop(

        columns=[

            "credential_id",

            "label",

            "manipulation_type"

        ],

        errors="ignore"

    )


    y = df["label"]



    institutions = (

        df["institution"]

        .unique()

    )


    if len(institutions) < 2:

        raise ValueError(

            "Dataset membutuhkan minimal dua institusi"

        )



    results = []



    for train_inst in institutions:


        test_inst = [

            i for i in institutions

            if i != train_inst

        ]



        test_inst = test_inst[0]



        print("\n")

        print("="*60)

        print(

            "Train Institution:",

            train_inst

        )

        print(

            "Test Institution:",

            test_inst

        )



        train_idx = (

            df["institution"]

            == train_inst

        )


        test_idx = (

            df["institution"]

            == test_inst

        )



        X_train = X[train_idx]

        X_test = X[test_idx]


        y_train = y[train_idx]

        y_test = y[test_idx]



        for name, model in MODELS.items():


            print(

                "Running:",

                name

            )



            metrics = evaluate_model(

                model,

                X_train,

                X_test,

                y_train,

                y_test

            )



            results.append(

                {

                    "Experiment":

                    "Institution Transfer",


                    "Train Institution":

                    train_inst,


                    "Test Institution":

                    test_inst,


                    "Model":

                    name,


                    **metrics

                }

            )



    RESULT_PATH.mkdir(

        exist_ok=True

    )



    result_df = pd.DataFrame(

        results

    )


    result_df.to_csv(

        RESULT_PATH /

        "institution_transfer_comparison.csv",

        index=False

    )



    summary_df = (

        result_df

        .groupby(

            "Model"

        )

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

        RESULT_PATH /

        "institution_transfer_summary.csv",

        index=False

    )



    print("\nCompleted")

    print(

        "Saved institution transfer results"

    )




if __name__ == "__main__":

    run_institution_transfer()