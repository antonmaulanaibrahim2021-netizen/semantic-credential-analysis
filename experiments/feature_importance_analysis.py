"""
Semantic Feature Importance Analysis

Output:
results/semantic_feature_importance.csv
"""

from pathlib import Path
import sys

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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
        col for col in categorical_features
        if col in X.columns
    ]


    numeric_features = [
        col for col in numeric_features
        if col in X.columns
    ]


    preprocessor = ColumnTransformer(

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


    return preprocessor



# ======================================
# FEATURE IMPORTANCE
# ======================================

def run_feature_importance():


    print("Loading dataset...")


    df = load_dataset()

    df = clean_dataset(df)


    print("Dataset shape:", df.shape)



    X = df.drop(

        columns=[

            "credential_id",

            "label",

            "manipulation_type"

        ],

        errors="ignore"

    )


    y = df["label"]



    preprocessor = create_preprocessor(X)



    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42

    )



    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor

            ),

            (
                "model",

                model

            )

        ]

    )



    print("Training Random Forest...")


    pipeline.fit(
        X,
        y
    )


    # ==================================
    # Extract feature names
    # ==================================

    encoder = (
        pipeline
        .named_steps["preprocessor"]
        .named_transformers_["categorical"]
    )


    categorical_names = (
        encoder
        .get_feature_names_out()
        .tolist()
    )


    numeric_names = (

        X.select_dtypes(
            exclude="object"
        )

        .columns

        .tolist()

    )


    feature_names = (
        categorical_names
        +
        numeric_names
    )


    importance = (

        pipeline
        .named_steps["model"]
        .feature_importances_

    )


    result = pd.DataFrame(

        {

            "Feature":

                feature_names,


            "Importance":

                importance

        }

    )


    result = result.sort_values(

        by="Importance",

        ascending=False

    )


    result["Rank"] = range(

        1,

        len(result)+1

    )


    result = result[

        [

            "Rank",

            "Feature",

            "Importance"

        ]

    ]



    RESULT_PATH.mkdir(

        exist_ok=True

    )


    result.head(20).to_csv(

        RESULT_PATH /

        "semantic_feature_importance.csv",

        index=False

    )


    print("\nCompleted")

    print(

        "Saved:",

        RESULT_PATH /

        "semantic_feature_importance.csv"

    )



if __name__ == "__main__":

    run_feature_importance()