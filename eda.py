import pandas as pd

# Load the employee dataset
file_path = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

# Display the first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset shape:")
print(df.shape)

# Display column names
print("\nColumn names:")
print(df.columns.tolist())

# Display data types
print("\nData types:")
print(df.dtypes)


# Step 3: Data quality checks

print("\n--- DATA QUALITY CHECK ---")

# 1. Check number of missing values
print("\nMissing values:")
print(df.isnull().sum())

# 2. Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# 3. Check dataset information
print("\nDataset information:")
df.info()

# 4. Check target column distribution
print("\nAttrition distribution:")
print(df["Attrition"].value_counts())

# 5. Check summary statistics
print("\nSummary statistics:")
print(df.describe())



# Step 3.3: Data cleaning

# Create a copy of the original dataset
clean_df = df.copy()

# Remove duplicate rows, if any
clean_df = clean_df.drop_duplicates()

# Check missing values
print("Missing values before cleaning:")
print(clean_df.isnull().sum().sum())

# Check constant columns
constant_columns = [
    col for col in clean_df.columns
    if clean_df[col].nunique() <= 1
]

print("\nConstant columns:")
print(constant_columns)

# Remove constant columns if they have no analytical value
clean_df = clean_df.drop(columns=constant_columns)

# Check final dataset
print("\nCleaned dataset shape:")
print(clean_df.shape)

print("\nMissing values after duplicate removal:")
print(clean_df.isnull().sum().sum())

# Save cleaned dataset
clean_df.to_csv("data/employee_attrition_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")



import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
clean_df = pd.read_csv("data/employee_attrition_cleaned.csv")

# Convert Attrition into readable labels
clean_df["Attrition"] = clean_df["Attrition"].astype(str)

print("Attrition counts:")
print(clean_df["Attrition"].value_counts())

# Chart 1: Employee attrition distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=clean_df, x="Attrition")
plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("data/attrition_distribution.png")
plt.show()

# Chart 2: Attrition by overtime
plt.figure(figsize=(8, 5))
sns.countplot(data=clean_df, x="OverTime", hue="Attrition")
plt.title("Employee Attrition by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("data/attrition_by_overtime.png")
plt.show()

# Chart 3: Attrition by job satisfaction
plt.figure(figsize=(8, 5))
sns.countplot(data=clean_df, x="JobSatisfaction", hue="Attrition")
plt.title("Employee Attrition by Job Satisfaction")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("data/attrition_by_satisfaction.png")
plt.show()

print("EDA charts created successfully!")
