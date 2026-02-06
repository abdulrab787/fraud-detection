import os
from xml.parsers.expat import model
import joblib
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

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

# Compute imbalance ratio
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos


print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

# LightGBM Parameters (strong default)
lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": scale_pos_weight,
}

# Convert to LightGBM Dataset
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Train with Early Stopping
print("Training LightGBM with Early Stopping...")

lgb_model = lgb.train(
    params=lgb_params,
    train_set=train_data,
    valid_sets=[train_data, val_data],
    valid_names=["train", "validation"],
    num_boost_round=3000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50)
    ]
)

# Validation Predictions
val_probs = lgb_model.predict(X_val)
val_preds = (val_probs >= 0.5).astype(int)

# Evaluation Metrics
acc = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
auc = roc_auc_score(y_val, val_probs)

print("\n===== LIGHTGBM RESULTS =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

cm = confusion_matrix(y_val, val_preds)
print("\nConfusion Matrix:")
print(cm)

# Save Model
joblib.dump(lgb_model, MODELS_DIR / "lgb_fraud_model.pkl")
print(f"Model saved to: {MODELS_DIR / 'lgb_fraud_model.pkl'}")
