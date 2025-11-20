RAW_DATA_DIR = "Data/Raw"
INTERIM_DATA_DIR = "Data/Interim"
PROCESSED_DATA_DIR = "Data/Processed"
ARTIFACTS_DIR = "artifacts"

# EMI scenario rules
EMI_RULES = {
    "E-commerce Shopping EMI": {"amount_min": 10_000, "amount_max": 200_000, "tenure_min": 3, "tenure_max": 24},
    "Home Appliances EMI": {"amount_min": 20_000, "amount_max": 300_000, "tenure_min": 6, "tenure_max": 36},
    "Vehicle EMI": {"amount_min": 80_000, "amount_max": 1_500_000, "tenure_min": 12, "tenure_max": 84},
    "Personal Loan EMI": {"amount_min": 50_000, "amount_max": 1_000_000, "tenure_min": 12, "tenure_max": 60},
    "Education EMI": {"amount_min": 50_000, "amount_max": 500_000, "tenure_min": 6, "tenure_max": 48},
}

# quick lists (edit if your column names differ)
NUMERIC_COLS = [
    "Age","Monthly_Salary","Years_Of_Employment","Monthly_Rent","Family_Size","Dependents",
    "School_Fees","College_Fees","Travel_Expenses","Groceries_Utilities","Other_Monthly_Expenses",
    "Current_Emi_Amount","Credit_Score","Bank_Balance","Emergency_Fund",
    "Requested_Amount","Requested_Tenure","Max_Monthly_Emi"
]

CATEGORICAL_COLS = [
    "Gender","Marital_Status","Education","Employment_Type","Company_Type","House_Type",
    "Existing_Loans","Emi_Scenario"
]

# columns to drop after feature creation (example)
COLS_TO_DROP = [
    "Monthly_Rent","School_Fees","College_Fees","Travel_Expenses","Groceries_Utilities",
    "Other_Monthly_Expenses","Family_Size","Dependents"
]

# Targets
TARGET_REG = "Max_Monthly_Emi"
TARGET_CLF = "Emi_Eligibility"

# Split params
TRAIN_FRACTION = 0.70    # first split: train vs temp
TEMP_FRACTION = 0.30     # remainder (will be split into val & test)
VAL_FRACTION = 0.50      # fraction of temp -> val (0.5 means val=test)
TEST_FRACTION = 0.50     # fraction of temp -> test (kept for clarity)

# Random seed
RANDOM_STATE = 42