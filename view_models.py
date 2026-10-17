from joblib import load

# Load the saved models
lr_model = load("models/attrition_model.joblib")
rf_model = load("models/random_forest_model.joblib")

print("LOGISTIC REGRESSION MODEL")
print(lr_model)

print("\nRANDOM FOREST MODEL")
print(rf_model)

print("\nRandom Forest number of trees:")
print(
    len(rf_model.named_steps["classifier"].estimators_)
)
