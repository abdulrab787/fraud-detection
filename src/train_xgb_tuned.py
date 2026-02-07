import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

from preprocessing.preprocessing import preprocess_data
from config import MODELS_DIR, TRAIN_PATH, TEST_PATH


# Load & preprocess data
X_train, X_val, y_train, y_val, X_test, preprocessor = preprocess_data(
    TRAIN_PATH,
    TEST_PATH
)

# Load best params (must be a dict)
best_params = joblib.load(MODELS_DIR / "best_xgb_params.pkl")

print("\nTraining FINAL tuned XGBoost...")

xgb_model = XGBClassifier(**best_params)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)
thresholds = np.linspace(0.001, 0.20, 200)
best_f1 = 0
best_t = 0.5

for t in thresholds:
    val_preds = (xgb_model.predict_proba(X_val)[:, 1] >= t).astype(int)
    f1 = f1_score(y_val, val_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t
print(f"\nBest F1: {best_f1:.4f} at threshold: {best_t:.4f}")


# Evaluate
val_preds = xgb_model.predict(X_val)
val_probs = xgb_model.predict_proba(X_val)[:, 1]

# Apply best threshold
best_threshold = 0.0290  # from your threshold search
final_preds = (val_probs >= best_threshold).astype(int)

print("\n===== THRESHOLD-OPTIMIZED RESULTS =====")
print(f"Precision: {precision_score(y_val, final_preds):.4f}")
print(f"Recall:    {recall_score(y_val, final_preds):.4f}")
print(f"F1-score:  {f1_score(y_val, final_preds):.4f}")

print("\nConfusion Matrix (Threshold Optimized):")
print(confusion_matrix(y_val, final_preds))

acc = accuracy_score(y_val, val_preds)
precision = precision_score(y_val, val_preds, zero_division=0)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)
auc = roc_auc_score(y_val, val_probs)

print("\n===== TUNED XGBOOST RESULTS =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

cm = confusion_matrix(y_val, val_preds)
print("\nConfusion Matrix:")
print(cm)

# Save final tuned model
save_path = MODELS_DIR / "xgb_fraud_model_tuned.pkl"
joblib.dump(xgb_model, save_path)

print(f"Model saved to: {save_path}")