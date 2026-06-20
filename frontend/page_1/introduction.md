# Extreme Gradient Boosting Classifier ( XGBOOST )

According to its official documentation and community definition, XGBoost (Extreme Gradient Boosting) is an open-source software library providing a highly optimized gradient boosting framework. Designed for speed, flexibility, and portability, it builds an ensemble of sequentially trained decision trees to solve classification, regression, and ranking problems. The algorithm is widely recognized for its "regularized boosting" formulation ($L_1$ and $L_2$ penalties) which actively limits model complexity to curb overfitting while maximizing computational efficiency across single systems and distributed execution environments.

### About the Dataset
The model is trained on the classic **Telco Customer Churn** dataset available on Kaggle, tracking profiles from 7,043 customers across 21 core features.The dataset tracks 7,043 records across 21 features. Here is the compressed data schema organized by feature type:

| Categorical / Object Data | Numerical Data | Target Variable |
| :--- | :--- | :--- |
| `customerID`, `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | `SeniorCitizen` *(Binary)*, `tenure` *(Integer)*, `MonthlyCharges` *(Float)*, `TotalCharges` *(Float)* | `Churn` *(Binary)* |

🔗 **Dataset Source:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---