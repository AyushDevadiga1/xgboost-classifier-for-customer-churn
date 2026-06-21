# Make Predictions on Your Own Data

Upload a CSV file and the app will predict churn risk for each customer. You can upload data with or without the actual outcome — both are supported.

> **Before you upload:**
>
> - Download the sample CSV below to see the exact column structure expected.
> - Keep the column names and order exactly as shown in the sample.
> - The file must be a `.csv` with the same data types as the sample (numbers as numbers, text as text).
> - Remove any personally identifiable information (names, phone numbers, addresses) before uploading.

---

## Two Ways to Use This

**Inference mode** — upload customer data without a `Churn` column, and the app predicts churn risk for each row. This mirrors how the model would be used in production, where the actual outcome isn't known yet.

**Evaluation mode** — upload data that includes the `Churn` column, and the app will additionally show how accurate the model's predictions were against the real outcomes, along with the full set of evaluation charts.

The app detects which mode applies automatically based on whether a `Churn` column is present.

