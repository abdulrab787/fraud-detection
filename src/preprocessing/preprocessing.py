import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering from transaction timestamp.
    """
    df = df.copy()

    df["Time"] = pd.to_datetime(df["Time"], unit="s")

    df["hour"] = df["Time"].dt.hour
    df["day"] = df["Time"].dt.day
    df["month"] = df["Time"].dt.month
    df["dayofweek"] = df["Time"].dt.dayofweek

    df.drop(columns=["Time"], inplace=True)

    return df


def preprocess_data(train_path: str, test_path: str):
    """
    Full preprocessing pipeline (industry-style).
    Returns processed train, test, and fitted preprocessor.
    """

    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Separate target
    y = train["IsFraud"]
    X = train.drop(columns=["IsFraud"])
    # Add time features
    X = add_time_features(X)
    test = add_time_features(test)

    # Drop obvious ID-like columns (leakage risk)
    drop_cols = ["Unnamed: 0", "cc_num", "first", "last", "street", "zip"]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    # Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"Numeric cols: {num_cols}")
    print(f"Categorical cols: {cat_cols}")

    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # High-cardinality categorical columns (target encoding)
    high_cardinality_cols = [c for c in cat_cols if X[c].nunique() > 10]

    # Low-cardinality categorical columns (one-hot)
    low_cardinality_cols = [c for c in cat_cols if X[c].nunique() <= 10]

    # Pipelines
    cat_high_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_encoder", TargetEncoder())
    ])

    cat_low_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine everything
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat_high", cat_high_pipeline, high_cardinality_cols),
            ("cat_low", cat_low_pipeline, low_cardinality_cols),
        ],
        remainder="drop"
    )

    # Fit on training data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting preprocessing pipeline...")
    preprocessor.fit(X_train, y_train)

    # Transform data
    X_train_p = preprocessor.transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(test)

    # Save preprocessor for later use
    joblib.dump(preprocessor, "../models/preprocessor.pkl")

    print("Preprocessing complete. Saved preprocessor.pkl")

    return X_train_p, X_val_p, y_train, y_val, X_test_p
