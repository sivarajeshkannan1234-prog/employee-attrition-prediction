
import streamlit as st
import pandas as pd
from joblib import load

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    return load("models/best_model.joblib")

model = load_model()

# Dashboard title
st.title("📊 Employee Attrition Prediction")
st.write(
    "Enter employee details to predict the likelihood "
    "of employee attrition."
)

st.info(
    "This is a machine-learning estimate for project "
    "demonstration, not a definitive HR decision."
)

# Employee details
st.header("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 65, 30)
    monthly_income = st.number_input(
        "Monthly Income", 1000, 100000, 30000
    )
    distance = st.number_input(
        "Distance From Home", 1, 30, 5
    )
    overtime = st.selectbox(
        "OverTime", ["No", "Yes"]
    )

with col2:
    department = st.selectbox(
        "Department",
        [
            "Research & Development",
            "Sales",
            "Human Resources"
        ]
    )
    job_role = st.selectbox(
        "Job Role",
        [
            "Laboratory Technician",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Research Director",
            "Human Resources"
        ]
    )
    job_satisfaction = st.selectbox(
        "Job Satisfaction", [1, 2, 3, 4]
    )
    work_life_balance = st.selectbox(
        "Work-Life Balance", [1, 2, 3, 4]
    )

with col3:
    years_at_company = st.number_input(
        "Years at Company", 0, 40, 3
    )
    years_in_current_role = st.number_input(
        "Years in Current Role", 0, 20, 2
    )
    years_since_promotion = st.number_input(
        "Years Since Last Promotion", 0, 20, 1
    )
    total_working_years = st.number_input(
        "Total Working Years", 0, 45, 5
    )

# Predict button
if st.button("Predict Attrition", type="primary"):

    # Start with every feature used during training
    input_df = pd.DataFrame(
        [model.feature_names_in_]
    ) if False else None

    # Build input using the original dataset columns
    original_df = pd.read_csv(
        "data/employee_attrition_cleaned.csv",
        nrows=1
    )

    input_data = {}

    for column in original_df.drop(
        columns=["Attrition", "EmployeeNumber"],
        errors="ignore"
    ).columns:
        if column == "Age":
            input_data[column] = age
        elif column == "MonthlyIncome":
            input_data[column] = monthly_income
        elif column == "DistanceFromHome":
            input_data[column] = distance
        elif column == "OverTime":
            input_data[column] = overtime
        elif column == "Department":
            input_data[column] = department
        elif column == "JobRole":
            input_data[column] = job_role
        elif column == "JobSatisfaction":
            input_data[column] = job_satisfaction
        elif column == "WorkLifeBalance":
            input_data[column] = work_life_balance
        elif column == "YearsAtCompany":
            input_data[column] = years_at_company
        elif column == "YearsInCurrentRole":
            input_data[column] = years_in_current_role
        elif column == "YearsSinceLastPromotion":
            input_data[column] = years_since_promotion
        elif column == "TotalWorkingYears":
            input_data[column] = total_working_years
        else:
            # Fill remaining fields with a representative
            # value from the dataset
            input_data[column] = original_df.iloc[0][column]

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Attrition Probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.warning(
            "The model predicts a higher likelihood of attrition."
        )
    else:
        st.success(
            "The model predicts a lower likelihood of attrition."
        )

    st.caption(
        "Predictions are estimates and should not be used "
        "as the sole basis for employment decisions."
    )



# ===== EMPLOYEE ATTRITION ANALYTICS =====

st.divider()
st.header("📈 Employee Attrition Analytics")

# Load cleaned dataset
df = pd.read_csv("data/employee_attrition_cleaned.csv")

# Summary metrics
total_employees = len(df)
employees_left = (df["Attrition"] == "Yes").sum()
attrition_rate = employees_left / total_employees * 100
employees_stayed = total_employees - employees_left

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", total_employees)
col2.metric("Employees Left", employees_left)
col3.metric("Attrition Rate", f"{attrition_rate:.2f}%")

# Attrition distribution chart
st.subheader("Employee Attrition Distribution")

attrition_counts = (
    df["Attrition"]
    .value_counts()
    .rename_axis("Status")
    .reset_index(name="Employees")
)

st.bar_chart(
    attrition_counts,
    x="Status",
    y="Employees"
)

# Attrition by overtime
st.subheader("Attrition by Overtime")

overtime_data = pd.crosstab(
    df["OverTime"],
    df["Attrition"]
)

st.bar_chart(overtime_data)

# Attrition by job satisfaction
st.subheader("Attrition by Job Satisfaction")

satisfaction_data = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"]
)

st.bar_chart(satisfaction_data)

st.caption(
    "Analytics describe the dataset and do not establish "
    "that any factor causes employee attrition."
)

# ===== MODEL PERFORMANCE =====

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

st.divider()
st.header("🤖 Machine Learning Model Performance")

# Load dataset and selected model
df_model = pd.read_csv("data/employee_attrition_cleaned.csv")
df_model = df_model.drop(
    columns=["EmployeeNumber"], errors="ignore"
)

X = df_model.drop(columns=["Attrition"])
y = df_model["Attrition"].map({"Yes": 1, "No": 0})

# Re-create the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Evaluate saved best model
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

col1, col2 = st.columns(2)
col1.metric("Test Accuracy", f"{accuracy:.2%}")
col2.metric("ROC-AUC Score", f"{roc_auc:.4f}")

# Confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual: Stayed", "Actual: Left"],
    columns=["Predicted: Stayed", "Predicted: Left"]
)

st.dataframe(cm_df)

# Classification report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Stayed", "Left"],
    output_dict=True,
    zero_division=0
)

st.dataframe(
    pd.DataFrame(report).transpose().round(3)
)

st.caption(
    "Metrics are calculated on the held-out test set. "
    "They measure model performance, not certainty about an individual."
)
