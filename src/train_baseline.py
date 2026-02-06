import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from preprocessing.preprocessing import preprocess_data
from config import TRAIN_PATH, TEST_PATH, PREPROCESSOR_PATH, MODEL_PATH


# Load & preprocess data
X_train, X_val, y_train, y_val, X_test, preprocessor = preprocess_data(
    TRAIN_PATH,
    TEST_PATH
)

print("Data shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Baseline Model: Logistic Regression (Class Weights)
print("\nTraining Baseline Logistic Regression...")

baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)

# Validation Predictions
val_preds = baseline_model.predict(X_val)
val_probs = baseline_model.predict_proba(X_val)[:, 1]

# Evaluation Metrics
acc = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
auc = roc_auc_score(y_val, val_probs)

print("\n===== BASELINE RESULTS =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_val, val_preds)
print("\nConfusion Matrix:")
print(cm)

# Save Preprocessor
joblib.dump(preprocessor, PREPROCESSOR_PATH)
print(f"\nPreprocessor saved to: {PREPROCESSOR_PATH}")

# Save Model
joblib.dump(baseline_model, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")