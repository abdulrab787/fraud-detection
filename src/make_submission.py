import pandas as pd
import joblib
import numpy as np
import lightgbm as lgb

from preprocessing.preprocessing import add_time_features
from config import MODELS_DIR, TEST_PATH


# Load test data
print("Loading test data...")
test = pd.read_csv(TEST_PATH)

# Keep transaction ID for submission
if "TransactionID" in test.columns:
    submission_ids = test["TransactionID"]
else:
    submission_ids = test.index


# Feature engineering
test = add_time_features(test)

drop_cols = ["Unnamed: 0", "cc_num", "first", "last", "street", "zip"]
test = test.drop(columns=[c for c in drop_cols if c in test.columns])


# Load saved preprocessor
print("Loading preprocessor...")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

X_test = preprocessor.transform(test)


# Load base models
print("Loading base models...")

logreg = joblib.load(MODELS_DIR / "logreg_baseline.pkl")
xgb = joblib.load(MODELS_DIR / "xgb_fraud_model_tuned.pkl")

# Load LightGBM correctly
lgb_model = lgb.Booster(model_file=str(MODELS_DIR / "lgb_fraud_model_tuned.txt"))


# Create meta-features for test
print("Generating meta-features for test set...")

meta_test = pd.DataFrame({
    "logreg_prob": logreg.predict_proba(X_test)[:, 1],
    "xgb_prob": xgb.predict_proba(X_test)[:, 1],
    "lgb_prob": lgb_model.predict(X_test)
})


# Load stacking meta-model
print("Loading stacking meta-model...")
meta_model = joblib.load(MODELS_DIR / "stacking_meta_model.pkl")

final_probs = meta_model.predict_proba(meta_test)[:, 1]
final_preds = (final_probs >= 0.5).astype(int)


# Create submission file
submission = pd.DataFrame({
    "id": submission_ids,
    "is_fraud": final_preds
})

output_path = MODELS_DIR / "stacking_submission.csv"
submission.to_csv(output_path, index=False)

print(f"Submission saved to: {output_path}")
print("First 5 rows:")
print(submission.head())
