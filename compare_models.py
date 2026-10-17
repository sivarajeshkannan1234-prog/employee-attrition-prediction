
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load cleaned dataset
df = pd.read_csv("data/employee_attrition_cleaned.csv")
df = df.drop(columns=["EmployeeNumber"], errors="ignore")

X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Use the same test split as model training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained models
models = {
    "Logistic Regression": load("models/attrition_model.joblib"),
    "Random Forest": load("models/random_forest_model.joblib")
}

results = []

for name, model in models.items():
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4)
    })

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

results_df.to_csv("data/model_comparison.csv", index=False)
print("\nComparison saved to data/model_comparison.csv")
