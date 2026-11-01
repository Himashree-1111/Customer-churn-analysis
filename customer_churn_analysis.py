import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path =r"C:\Users\ELCOT\Downloads\Telco-Customer-Churn.csv"
df=pd.read_csv(file_path)

#Display First 5 rows
print(df.head())#Display dataset dimensions
print("\nDataset Shape:",df.shape)

#isplay Column names
print("\nColumns Names:")
print(df.columns.tolist())


#Step 2 :Data quality  checks
print("\n---DATASET INFORMATION---")
df.info()
print("\n---MISSING VALUES---")
print(df.isnull().sum())
print("\n---DUPLICATE ROWS---")
print("Duplicate rows:", df.duplicated().sum())
print("\n---CUSTOMER CHURN COUNTS---")
print(df["Churn"].value_counts())
print("\n---NUMERICAL SUMMARY---")
print(df.describe())

# STEP 3: DATA CLEANING

print("\n--- DATA CLEANING STARTED ---")

# Create a working copy
df_clean = df.copy()

# Remove duplicate rows
df_clean = df_clean.drop_duplicates()

# Convert TotalCharges to numeric
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"], errors="coerce"
)

# Check missing values after conversion
print("\nMissing values after conversion:")
print(df_clean.isnull().sum())

# Remove rows with missing TotalCharges
df_clean = df_clean.dropna(subset=["TotalCharges"])

# Check final dataset
print("\nCleaned dataset shape:", df_clean.shape)
print("Remaining duplicate rows:", df_clean.duplicated().sum())

print("\n--- DATA CLEANING COMPLETED ---")

# STEP 4: CUSTOMER CHURN KPI ANALYSIS

total_customers = len(df_clean)

churned_customers = (df_clean["Churn"] == "Yes").sum()
retained_customers = (df_clean["Churn"] == "No").sum()

churn_rate = (churned_customers / total_customers) * 100
retention_rate = (retained_customers / total_customers) * 100

average_monthly_charges = df_clean["MonthlyCharges"].mean()
average_tenure = df_clean["tenure"].mean()
total_monthly_revenue = df_clean["MonthlyCharges"].sum()

print("\n===== CUSTOMER CHURN KPI REPORT =====")

print("Total Customers:", total_customers)
print("Churned Customers:", churned_customers)
print("Retained Customers:", retained_customers)
print("Churn Rate:", round(churn_rate, 2), "%")
print("Retention Rate:", round(retention_rate, 2), "%")
print("Average Monthly Charges: $", round(average_monthly_charges, 2))
print("Average Customer Tenure:", round(average_tenure, 2), "months")
print("Total Monthly Charges: $", round(total_monthly_revenue, 2))

# STEP 5: CHURN BY CONTRACT TYPE

print("\n===== CHURN BY CONTRACT TYPE =====")

contract_churn = pd.crosstab(
    df_clean["Contract"],
    df_clean["Churn"],
    normalize="index"
) * 100

print(contract_churn.round(2))

# Calculate churn rate for each contract
churn_by_contract = (
    df_clean.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\nChurn Rate by Contract (%):")
print(churn_by_contract.round(2))

# STEP 7: CHURN BY CUSTOMER TENURE

def categorize_tenure(months):
    if months <= 12:
        return "0-12 Months"
    elif months <= 24:
        return "13-24 Months"
    elif months <= 48:
        return "25-48 Months"
    else:
        return "49+ Months"

df_clean["TenureGroup"] = df_clean["tenure"].apply(categorize_tenure)

tenure_churn = (
    df_clean.groupby("TenureGroup")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

tenure_order = [
    "0-12 Months",
    "13-24 Months",
    "25-48 Months",
    "49+ Months"
]

tenure_churn = tenure_churn.reindex(tenure_order)

print("\n===== CHURN BY TENURE =====")
print(tenure_churn.round(2))

# Create chart
plt.figure(figsize=(9, 5))
tenure_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Tenure Group")
plt.xlabel("Customer Tenure")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("churn_by_tenure.png", dpi=300)
plt.show()

# STEP 8: CHURN BY MONTHLY CHARGES

df_clean["ChargeGroup"] = pd.cut(
    df_clean["MonthlyCharges"],
    bins=[0, 35, 70, 100, float("inf")],
    labels=["Low", "Medium", "High", "Very High"]
)

charge_churn = (
    df_clean.groupby("ChargeGroup", observed=False)["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\n===== CHURN BY MONTHLY CHARGES =====")
print(charge_churn.round(2))

# Visualize results
plt.figure(figsize=(9, 5))
charge_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Monthly Charges")
plt.xlabel("Monthly Charge Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("churn_by_monthly_charges.png", dpi=300)
plt.show()

# STEP 9: CHURN BY PAYMENT METHOD

payment_churn = (
    df_clean.groupby("PaymentMethod")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\n===== CHURN BY PAYMENT METHOD =====")
print(payment_churn.round(2))

# Create visualization
plt.figure(figsize=(10, 6))
payment_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig("churn_by_payment_method.png", dpi=300)
plt.show()

# STEP 10: CHURN BY INTERNET SERVICE

internet_churn = (
    df_clean.groupby("InternetService")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values(ascending=False)
)

print("\n===== CHURN BY INTERNET SERVICE =====")
print(internet_churn.round(2))

# Create visualization
plt.figure(figsize=(9, 5))

internet_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("churn_by_internet_service.png", dpi=300)
plt.show()

# STEP 11: CHURN BY SENIOR CITIZEN STATUS

df_clean["CustomerAgeGroup"] = df_clean["SeniorCitizen"].map({
    0: "Non-Senior",
    1: "Senior"
})

senior_churn = (
    df_clean.groupby("CustomerAgeGroup")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
)

print("\n===== CHURN BY SENIOR CITIZEN STATUS =====")
print(senior_churn.round(2))

# Create visualization
plt.figure(figsize=(8, 5))

senior_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Senior Citizen Status")
plt.xlabel("Customer Group")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("churn_by_senior_status.png", dpi=300)
plt.show()

# STEP 12: KPI SUMMARY TABLE

kpi_summary = pd.DataFrame({
    "KPI": [
        "Total Customers",
        "Churned Customers",
        "Retained Customers",
        "Churn Rate (%)",
        "Retention Rate (%)",
        "Average Monthly Charges",
        "Average Tenure (Months)"
    ],
    "Value": [
        len(df_clean),
        (df_clean["Churn"] == "Yes").sum(),
        (df_clean["Churn"] == "No").sum(),
        round((df_clean["Churn"] == "Yes").mean() * 100, 2),
        round((df_clean["Churn"] == "No").mean() * 100, 2),
        round(df_clean["MonthlyCharges"].mean(), 2),
        round(df_clean["tenure"].mean(), 2)
    ]
})

print("\n===== CUSTOMER CHURN KPI SUMMARY =====")
print(kpi_summary.to_string(index=False))

# Save the KPI table
kpi_summary.to_csv("customer_churn_kpi_summary.csv", index=False)

print("\nKPI summary saved successfully!")

# STEP 13: EXPORT ANALYSIS TO EXCEL

with pd.ExcelWriter("Customer_Churn_Analysis_Report.xlsx",
                    engine="openpyxl") as writer:

    kpi_summary.to_excel(
        writer, sheet_name="KPI Summary", index=False
    )

    churn_by_contract = (
        df_clean.groupby("Contract")["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate (%)")
    )

    churn_by_contract.to_excel(
        writer, sheet_name="Contract Analysis", index=False
    )

    tenure_churn.reset_index(
        name="Churn Rate (%)"
    ).to_excel(
        writer, sheet_name="Tenure Analysis", index=False
    )

    charge_churn.reset_index(
        name="Churn Rate (%)"
    ).to_excel(
        writer, sheet_name="Charge Analysis", index=False
    )

    payment_churn.reset_index(
        name="Churn Rate (%)"
    ).to_excel(
        writer, sheet_name="Payment Analysis", index=False
    )

    internet_churn.reset_index(
        name="Churn Rate (%)"
    ).to_excel(
        writer, sheet_name="Internet Analysis", index=False
    )

    senior_churn.reset_index(
        name="Churn Rate (%)"
    ).to_excel(
        writer, sheet_name="Senior Citizen Analysis", index=False
    )

print("\nExcel report exported successfully!")


"""import os
print(os.getcwd())"""

total = len(df_clean)
churned = (df_clean["Churn"] == "Yes").sum()
print("Total customers:", total)
print("Churned customers:", churned)
print("Churn rate:", round(churned / total * 100, 2), "%")
