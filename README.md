# Telco-Churn-Analysis
End-to-end Machine Learning solution to predict customer attrition using XGBoost, featuring automated data reengineering (SMOTE) and an interactive Streamlit dashboard for pro-active retention strategies.

## 📊 Business Problem
Acquiring new customers in telecom is 5x more expensive than retaining existing ones. High attrition directly impacts monthly recurring revenue. This application identifies high-risk churners early, allowing marketing teams to launch targeted retention campaigns before customers leave.

## ✨ Key Features
* **Automated Data Reengineering**: Effortlessly handles categorical transformation (One-Hot & Label Encoding), feature relevance scoring via Mutual Information, and addresses 26% severe class imbalance using SMOTE.
* **Predictive Excellence**: Leverages advanced gradient boosting to capture non-linear customer behaviors, achieving an optimized ROC-AUC score of ~0.824.
* **Interactive Dashboard**: Streamlit-based control panel for real-time customer risk profiling, instant churn probability scoring, and key risk driver visualization.
* **Smart Retention Alert**: Automatically flags high-risk accounts (>70% churn probability) and maps them to tailored operational recommendations (e.g., contract migration or targeted tech audits).

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **ML Model:** XGBoost Classifier
- **Data Preprocessing:** Pandas, NumPy, Scikit-Learn, Imbalanced-Learn (SMOTE)
- **Dashboard:** Streamlit
- **Visualization:** Matplotlib, Seaborn
