import joblib
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from preprocessing.preprocessing import preprocess_data
from config import MODELS_DIR, TRAIN_PATH, TEST_PATH


# Load & preprocess data
X_train, X_val, y_train, y_val, X_test, preprocessor = preprocess_data(
    TRAIN_PATH,
    TEST_PATH
)

# Compute scale_pos_weight (critical for fraud)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print(f"\nScale Pos Weight: {scale_pos_weight:.2f}")


# Objective function for Optuna
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": -1,
        "random_state": 42,
    }

    model = XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)

    return auc


# Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

print("\n===== BEST XGBOOST PARAMETERS =====")
print(study.best_params)
print(f"\nBest AUC: {study.best_value:.6f}")

joblib.dump(study.best_params, MODELS_DIR / "best_xgb_params.pkl")
print("Saved best params to best_xgb_params.pkl")


# Retrain final model properly
best_params = study.best_params

best_params.update({
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "scale_pos_weight": scale_pos_weight,
    "n_jobs": -1,
    "random_state": 42,
})

final_model = XGBClassifier(**best_params)

final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# Save tuned model
save_path = MODELS_DIR / "xgb_fraud_model_optuna.pkl"
joblib.dump(final_model, save_path)

print(f"Model saved to: {save_path}")