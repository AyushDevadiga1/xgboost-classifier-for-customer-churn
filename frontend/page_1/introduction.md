# About the Model: XGBoost

XGBoost (Extreme Gradient Boosting) is a widely used machine learning algorithm for structured, tabular data. It builds a sequence of decision trees, where each tree is trained to correct the mistakes of the ones before it. It includes built-in regularization to prevent overfitting and is one of the most reliable default choices for classification problems like this one.

---

## About the Dataset

The model is trained on the **Telco Customer Churn** dataset, a widely used benchmark dataset available on Kaggle. It contains 7,043 customer records across 21 original features.

| Categorical Features | Numerical Features | Target |
| :--- | :--- | :--- |
| `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges` | `Churn` (Yes / No) |

**Source:** [Kaggle — Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---