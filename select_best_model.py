
import pandas as pd
from joblib import load, dump

# Load the model comparison results
results = pd.read_csv("data/model_comparison.csv")

print("MODEL COMPARISON")
print(results.to_string(index=False))

# Select the model with the highest F1-score
best_row = results.sort_values(
    by=["F1 Score", "ROC-AUC"],
    ascending=False
).iloc[0]

best_name = best_row["Model"]

# Load the selected model
if best_name == "Logistic Regression":
    best_model = load("models/attrition_model.joblib")
else:
    best_model = load("models/random_forest_model.joblib")

# Save the selected model
dump(best_model, "models/best_model.joblib")

print("\nSelected model:", best_name)
print("F1-score:", best_row["F1 Score"])
print("ROC-AUC:", best_row["ROC-AUC"])
print("Best model saved successfully!")
