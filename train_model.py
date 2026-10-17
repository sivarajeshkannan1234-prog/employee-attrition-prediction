import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/employee_attrition_cleaned.csv")

# Remove employee identifier if present
df = df.drop(columns=["EmployeeNumber"], errors="ignore")

# Separate features and target
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully!")
print("Features:", X.shape)
print("Target:", y.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("\nTraining and testing split completed!")


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from joblib import dump

# 1. Load dataset
df = pd.read_csv("data/employee_attrition_cleaned.csv")

# 2. Remove employee ID if present
df = df.drop(columns=["EmployeeNumber"], errors="ignore")

# 3. Separate features and target
X = df.drop(columns=["Attrition"])
y = df["Attrition"].map({"Yes": 1, "No": 0})

# 4. Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include="number").columns
categorical_columns = X.select_dtypes(exclude="number").columns

# 5. Prepare preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Create and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Evaluate the model
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Save trained model
dump(model, "models/attrition_model.joblib")

print("\nModel trained and saved successfully!")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from joblib import dump

# Train a Random Forest model using the same preprocessor
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the Random Forest model
print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))

print("\nClassification Report:")
print(classification_report(
    y_test, rf_pred, zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("ROC-AUC:", round(
    roc_auc_score(y_test, rf_prob), 4
))

# Save the Random Forest model
dump(rf_model, "models/random_forest_model.joblib")
print("\nRandom Forest model saved successfully!")


