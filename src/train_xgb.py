import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

from preprocessing.preprocessing import preprocess_data
from config import MODELS_DIR, TRAIN_PATH, TEST_PATH, PREPROCESSOR_PATH, MODEL_PATH

# Load & preprocess data
X_train, X_val, y_train, y_val, X_test, preprocessor = preprocess_data(
    TRAIN_PATH,
    TEST_PATH
)

print("Data shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# Compute scale_pos_weight (VERY IMPORTANT)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")

# Train XGBoost (Imbalance-Aware)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.02,
    "max_depth": 2,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "scale_pos_weight": min(scale_pos_weight, 50),
    "tree_method": "hist"
}

evals = [(dtrain, "train"), (dval, "validation")]

print("\nTraining XGBoost (Imbalance-Aware)...")

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=3000,
    evals=evals,
    early_stopping_rounds=100,
    verbose_eval=50
)


# Validation Predictions
val_probs = model.predict(dval)
val_preds = (val_probs > 0.5).astype(int)


# Evaluation Metrics
acc = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
auc = roc_auc_score(y_val, val_probs)

print("\n===== XGBOOST RESULTS =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

cm = confusion_matrix(y_val, val_preds)
print("\nConfusion Matrix:")
print(cm)

# Save Model
joblib.dump(model, MODELS_DIR / "xgb_baseline.pkl")
print(f"Model saved to: {MODELS_DIR / 'xgb_baseline.pkl'}")