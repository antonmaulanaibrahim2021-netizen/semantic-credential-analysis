"""
FEATURE ENCODING MODULE

Project:
Semantic_Credential_Q1

Experiment A:
Baseline Credential Attribute Model

Purpose:
- Load dataset
- Clean data
- Remove leakage features
- Encode categorical features
- Prepare ML-ready dataset
"""


from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline


# ==========================================================
# PATH CONFIGURATION
# ==========================================================

# preprocessing/
#       feature_encoding.py
#
# parent[0] = preprocessing
# parent[1] = Semantic_Credential_Q1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATA_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "cross_institutional_dataset (1).xlsx"
)


# ==========================================================
# COLUMN CONFIGURATION
# ==========================================================


TARGET_COLUMN = "label"


# tidak digunakan untuk training
DROP_COLUMNS = [
    "credential_id",
    "label",
    "manipulation_type"
]


NUMERIC_FEATURES = [
    "gpa"
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
# LOAD DATA
# ==========================================================

def load_dataset():

    print("\nLoading dataset...")


    sheet_name = "Combined_Labeled"


    # coba baca normal
    df = pd.read_excel(
        DATA_PATH,
        sheet_name=sheet_name,
        header=0
    )


    print("\nRaw columns:")
    print(df.columns.tolist())


    # cek apakah header masih berupa deskripsi
    if "label" not in [
        str(c).strip().lower()
        for c in df.columns
    ]:


        print(
            "\nHeader tidak ditemukan, mencoba header kedua..."
        )


        df = pd.read_excel(
            DATA_PATH,
            sheet_name=sheet_name,
            header=2
        )


    # normalisasi nama kolom

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )


    print("\nFinal columns:")
    print(df.columns.tolist())


    print(
        "\nDataset shape:",
        df.shape
    )


    return df

# ==========================================================
# CLEAN DATA
# ==========================================================

def clean_dataset(df):

    df = df.copy()


    # categorical missing

    for col in CATEGORICAL_FEATURES:

        if col in df.columns:

            df[col] = (
                df[col]
                .fillna("UNKNOWN")
                .astype(str)
            )


    # numerical missing

    if "gpa" in df.columns:

        df["gpa"] = (
            df["gpa"]
            .fillna(
                df["gpa"].median()
            )
        )


    return df



# ==========================================================
# SPLIT FEATURE TARGET
# ==========================================================

def split_xy(df):

    X = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )


    y = df[TARGET_COLUMN]


    return X, y



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


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            ),

            (
                "num",
                numerical_pipeline,
                NUMERIC_FEATURES
            )

        ]

    )


    return preprocessor



# ==========================================================
# MAIN ENCODING FUNCTION
# ==========================================================

def encode_baseline_features():


    print("="*70)
    print("BASELINE FEATURE ENCODING")
    print("="*70)


    df = load_dataset()


    df = clean_dataset(df)


    X, y = split_xy(df)


    print(
        "\nFeature sebelum encoding:",
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
        "Feature setelah encoding:",
        X_encoded.shape
    )


    print(
        "\nLabel distribution:"
    )

    print(
        y.value_counts()
    )


    return (
        X_encoded,
        y,
        processor
    )



# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":


    X, y, processor = (
        encode_baseline_features()
    )


    print("\nPreview:")
    print(
        X.head()
    )