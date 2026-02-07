# 💳 Credit Card Fraud Detection — End-to-End ML Pipeline (Industry-Grade)

🔍 **Problem:**  
Detect fraudulent credit card transactions in a highly imbalanced dataset where fraudulent cases are rare but extremely costly.

📌 **Goal:**  
Build a **reproducible, production-style ML system** with:
- Robust preprocessing
- Strong feature engineering  
- Multiple models (baseline → advanced)  
- Stacking ensemble  
- Threshold tuning (business-ready)  
- Explainability (SHAP)  
- Experiment tracking (MLflow)  
- Automated inference & submission pipeline  

---

## 🏆 Performance Summary

| Model | Validation Performance |
|------|------------------------|
| Logistic Regression | Baseline |
| Random Forest | Lower than baseline |
| Tuned XGBoost | **Best single model** ✅ |
| Tuned LightGBM | Competitive |
| Stacking Ensemble | **Best overall (after threshold tuning)** 🚀 |

*(Exact numbers can be updated from your logs if you want.)*


## 📂 Project Structure
credit-card-fraud/
│
├── data/
│ ├── raw/
│ │ ├── fraudTrain.csv
│ │ └── fraudTest.csv
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_baseline_models.ipynb
│ ├── 04_xgboost_tuning.ipynb
│ ├── 05_lightgbm_tuning.ipynb
│ ├── 06_stacking_ensemble.ipynb
│ ├── 09_threshold_tuning.ipynb
│ └── 10_shap_explainability.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── train_with_mlflow.py
│ └── make_submission.py
│
├── models/
│ ├── preprocessor.pkl
│ ├── xgb_fraud_model_tuned.pkl
│ ├── lgb_fraud_model_tuned.pkl
│ ├── logreg_baseline.pkl
│ ├── stacking_meta_model.pkl
│ └── best_threshold.txt
│
├── submissions/
│ └── stacking_submission.csv
│
├── mlruns/ # MLflow experiment tracking
├── requirements.txt
└── README.md


## 🔧 Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **LightGBM**
- **SHAP (Explainable AI)**
- **MLflow (Experiment Tracking)**
- **Joblib (Model Serialization)**


## 🔄 End-to-End Workflow

### **1️⃣ Exploratory Data Analysis (EDA)**

Key insights:
- Strong class imbalance (fraud is very rare)
- Transaction time patterns were informative  
- Certain merchant and location features correlated with fraud  
- Missing values handled systematically  

📍 Notebook: `notebooks/01_eda.ipynb`

---

### **2️⃣ Preprocessing & Feature Engineering (Industry-Grade)**

Key steps:
- Created **time-based features** (hour, day, weekday)
- Removed high-cardinality identifiers (`cc_num`, names, etc.)
- Imputed missing values appropriately  
- One-hot encoded categorical features  
- Standardized numeric features  
- Saved full pipeline as `models/preprocessor.pkl`

📍 Notebook: `notebooks/02_preprocessing.ipynb`  
📍 Script: `src/preprocessing.py`

---

### **3️⃣ Modeling (Baseline → Advanced)**

Models trained:
- Logistic Regression (baseline)
- Random Forest  
- Tuned XGBoost (**best single model**)  
- Tuned LightGBM  

📍 Notebooks:
- `03_baseline_models.ipynb`
- `04_xgboost_tuning.ipynb`
- `05_lightgbm_tuning.ipynb`

---

### **4️⃣ Stacking Ensemble (Production-Level Approach)**

Base models:
- Logistic Regression  
- XGBoost  
- LightGBM  

Meta-model:
- Logistic Regression trained on predicted probabilities

📍 Notebook: `06_stacking_ensemble.ipynb`

---

### **5️⃣ Threshold Tuning (Business-Ready)**

Instead of using a default 0.5 cutoff, I:
- Tuned decision threshold on validation set  
- Optimized for **F1-score** (better for fraud detection)
- Visualized F1 vs Threshold  
- Saved best threshold in `models/best_threshold.txt`

📍 Notebook: `09_threshold_tuning.ipynb`

---

### **6️⃣ Model Explainability (SHAP)**

Used **SHAP TreeExplainer** on tuned XGBoost model to provide:
- Global feature importance  
- Local explanations for individual transactions  
- Summary plots for interpretability  

📍 Notebook: `10_shap_explainability.ipynb`



### **7️⃣ Experiment Tracking (MLflow)**

Logged:
- Model parameters  
- Validation metrics (Accuracy, F1)  
- Trained model artifacts  

### **8️⃣ Automated Inference & Submission
python -m src.make_submission
Generates:
submissions/stacking_submission.csv

🚀 How to Run This Project
pip install -r requirements.txt

#Train models with MLflow
python -m src.train_with_mlflow

#Generate predictions
python -m src.make_submission



🎯 What I Learned 

Built a reproducible ML pipeline

Handled class imbalance effectively

Used feature engineering to boost performance

Applied stacking ensemble learning

Tuned decision threshold instead of default 0.5

Added explainability with SHAP

Tracked experiments using MLflow

Created production-style inference script

📬 Contact

GitHub: [abdulrab787](https://github.com/abdulrab787)

Kaggle:[abdurrabnizamuddeen](https://www.kaggle.com/abdurrabnizamuddeen)

LinkedIn:[abdulrab89](www.linkedin.com/in/abdulrab89)
