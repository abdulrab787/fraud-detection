import joblib
import numpy as np
import lightgbm as lgb
import optuna

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

print("Data shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# Compute imbalance ratio
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
scale_pos_weight = min(scale_pos_weight, 50)

print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)


# Objective function for Optuna
def objective(trial):

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0),

        "scale_pos_weight": scale_pos_weight,
        "feature_pre_filter": False,
    }

    lgb_model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )

    preds = lgb_model.predict(X_val)
    auc = roc_auc_score(y_val, preds)
    return auc


# Run Optuna Study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

print("\n===== BEST PARAMETERS =====")
print(study.best_params)
print(f"\nBest AUC: {study.best_value:.6f}")


# Retrain final model properly
best_params = study.best_params

best_params.update({
    "objective": "binary",
    "metric": "auc",
    "feature_pre_filter": False,
    "scale_pos_weight": scale_pos_weight,
})

final_train = lgb.Dataset(X_train, label=y_train)
final_val = lgb.Dataset(X_val, label=y_val)

final_model = lgb.train(
    params=best_params,
    train_set=final_train,
    valid_sets=[final_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=50)
    ]
)

# Save tuned model
save_path = MODELS_DIR / "lgb_fraud_model_tuned.txt"
final_model.save_model(str(save_path))
print(f"Model saved to: {save_path}")
