import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import MODELS_DIR
from preprocessing.preprocessing import preprocess_data
from config import TRAIN_PATH, TEST_PATH, MODELS_DIR

# 1️⃣ Load preprocessed data
X_train, X_val, y_train, y_val, X_test, preprocessor = preprocess_data(
    TRAIN_PATH,
    TEST_PATH
)


# Save validation set for SHAP / threshold tuning
joblib.dump(X_val, MODELS_DIR / "X_val.pkl")
joblib.dump(y_val, MODELS_DIR / "y_val.pkl")
# Save preprocessor for inference
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

# 2️⃣ Start MLflow Experiment
mlflow.set_experiment("fraud_detection_mlflow")

#  MODEL 1: Logistic Regression 
with mlflow.start_run(run_name="logreg_baseline"):

    model = LogisticRegression(max_iter=3000, class_weight="balanced")

    mlflow.log_param("max_iter", 3000)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(model, "logreg_model")

    joblib.dump(model, MODELS_DIR / "logreg_baseline.pkl")

#  MODEL 2: XGBoost 
with mlflow.start_run(run_name="xgboost_tuned"):

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss"
    }

    for k, v in params.items():
        mlflow.log_param(k, v)

    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(model, "xgb_model")

    joblib.dump(model, MODELS_DIR / "xgb_fraud_model_tuned.pkl")

#  MODEL 3: LightGBM 
with mlflow.start_run(run_name="lightgbm_tuned"):

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced"
    }

    for k, v in params.items():
        mlflow.log_param(k, v)

    model = lgb.LGBMClassifier(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    mlflow.sklearn.log_model(model, "lgb_model")

    # Save LightGBM correctly
    model.booster_.save_model(str(MODELS_DIR / "lgb_fraud_model_tuned.txt"))

print("MLflow training completed!")