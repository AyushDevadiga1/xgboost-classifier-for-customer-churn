# Customer Churn Prediction

## What This Project Does

This app predicts whether a telecom customer is likely to cancel their subscription (churn), using a machine learning model trained on historical customer data. The goal is simple: catch at-risk customers early enough that a business could try to retain them, instead of finding out only after they've already left.

---

## Why Churn Prediction Matters

Losing a customer costs more than keeping one. Acquiring a new customer is consistently more expensive than retaining an existing one, which is why companies invest in identifying which customers are likely to leave — so they can step in with a retention offer, a support call, or a pricing change before it's too late.

---

## How the System Works

The project follows three stages:

**1. Data Preparation**
Raw customer data — contract type, monthly charges, internet service, tenure, and so on — is cleaned and converted into a format a model can learn from. Categorical fields like "Yes/No" are mapped to numbers, and a few columns with no real predictive value (like customer ID) are dropped.

**2. Model Training**
An XGBoost classifier — a tree-based model well suited to structured, tabular data like this — is trained on the cleaned data. Because churned customers are a minority class in this dataset (about 1 in 4), the model is explicitly weighted to pay more attention to correctly identifying churners rather than just maximizing overall accuracy.

**3. Evaluation**
The trained model is tested on data it has never seen, using metrics that account for the class imbalance — precision, recall, F1-score, and ROC-AUC — rather than plain accuracy, which can be misleading when one class is rarer than the other.

---

## How to Read the Evaluation Metrics

| Metric | What It Measures | Why It Matters Here |
| :--- | :--- | :--- |
| **Precision** | Of everyone flagged as a churn risk, how many actually churned | Low precision means wasted retention spend on customers who wouldn't have left anyway |
| **Recall** | Of everyone who actually churned, how many the model caught | Low recall means real churners slip through undetected |
| **F1-Score** | A balance between precision and recall | Useful for comparing models at a glance |
| **ROC-AUC** | How well the model separates churners from non-churners across all decision thresholds | A threshold-independent measure of overall model quality |

For this project, **recall on churners is prioritized** — missing a customer who was about to leave is generally costlier than occasionally flagging a loyal customer who wasn't at risk.

---

## About the Model

This project uses **XGBoost**, a gradient boosting algorithm that builds an ensemble of decision trees sequentially, where each new tree corrects the errors of the previous ones. It was chosen after comparing it against LightGBM, CatBoost, and a Random Forest baseline on the same data and evaluation criteria.

**Dataset:** IBM Telco Customer Churn dataset — 7,043 customers, 21 original attributes covering demographics, account details, and the services each customer subscribed to.