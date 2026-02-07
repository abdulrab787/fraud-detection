import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
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


# Load trained base models
print("\nLoading base models...")

logreg = joblib.load(MODELS_DIR / "logreg_baseline.pkl")
xgb = joblib.load(MODELS_DIR / "xgb_fraud_model_tuned.pkl")
lgb_model = lgb.Booster(model_file=str(MODELS_DIR / "lgb_fraud_model_tuned.txt"))

print("Loaded all base models.")


# Create meta-features (validation set)
print("\nGenerating meta-features...")

meta_val = pd.DataFrame({
    "logreg_prob": logreg.predict_proba(X_val)[:, 1],
    "xgb_prob": xgb.predict_proba(X_val)[:, 1],
    "lgb_prob": lgb_model.predict(X_val)
})


# Train Meta-Model (Stacking)
print("\nTraining meta-model (Logistic Regression)...")

meta_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

meta_model.fit(meta_val, y_val)


# Evaluate Stacking Model
stack_probs = meta_model.predict_proba(meta_val)[:, 1]
stack_preds = (stack_probs >= 0.5).astype(int)

acc = accuracy_score(y_val, stack_preds)
precision = precision_score(y_val, stack_preds, zero_division=0)
recall = recall_score(y_val, stack_preds)
f1 = f1_score(y_val, stack_preds)
auc = roc_auc_score(y_val, stack_probs)

print("\n===== STACKING RESULTS =====")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, stack_preds))


# Retrain meta-model on FULL training data
print("\nRetraining meta-model on full data...")

meta_train = pd.DataFrame({
    "logreg_prob": logreg.predict_proba(X_train)[:, 1],
    "xgb_prob": xgb.predict_proba(X_train)[:, 1],
    "lgb_prob": lgb_model.predict(X_train)
})

meta_model.fit(meta_train, y_train)


# Create meta-features for test set
meta_test = pd.DataFrame({
    "logreg_prob": logreg.predict_proba(X_test)[:, 1],
    "xgb_prob": xgb.predict_proba(X_test)[:, 1],
    "lgb_prob": lgb_model.predict(X_test)
})


# Save artifacts
save_path = MODELS_DIR / "stacking_meta_model.pkl"
joblib.dump(meta_model, save_path)
print(f"Model saved to: {save_path}")

save_test_path = MODELS_DIR / "meta_test_features.pkl"
joblib.dump(meta_test, save_test_path)
print(f"Test features saved to: {save_test_path}")