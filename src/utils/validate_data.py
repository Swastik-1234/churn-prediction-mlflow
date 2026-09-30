
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    print("Starting data validation...")
    failed_expectations = []
    total_checks = 0

    def check(name, condition):
        total_checks_ref.append(1)
        if not condition:
            failed_expectations.append(name)

    total_checks_ref = []

    # Required columns
    required_columns = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]

    print("Validating schema and required columns...")
    for col in required_columns:
        check(f"expect_column_to_exist:{col}", col in df.columns)

    # Stop early if columns are missing
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print(f" Data validation FAILED: missing columns {missing}")
        return False, failed_expectations

    print("Validating business logic constraints...")
    check("expect_column_values_to_not_be_null:customerID",
          df["customerID"].notnull().all())

    check("expect_column_values_to_be_in_set:gender",
          df["gender"].isin(["Male", "Female"]).all())

    check("expect_column_values_to_be_in_set:Partner",
          df["Partner"].isin(["Yes", "No"]).all())

    check("expect_column_values_to_be_in_set:Dependents",
          df["Dependents"].isin(["Yes", "No"]).all())

    check("expect_column_values_to_be_in_set:PhoneService",
          df["PhoneService"].isin(["Yes", "No"]).all())

    check("expect_column_values_to_be_in_set:Contract",
          df["Contract"].isin(["Month-to-month", "One year", "Two year"]).all())

    check("expect_column_values_to_be_in_set:InternetService",
          df["InternetService"].isin(["DSL", "Fiber optic", "No"]).all())

    print("Validating numeric ranges and business constraints...")
    
# Convert TotalCharges to numeric (it loads as string in raw Telco data)
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    check("expect_column_values_to_not_be_null:tenure",
        df["tenure"].notnull().all())
    

    check("expect_column_values_to_not_be_null:MonthlyCharges",
          df["MonthlyCharges"].notnull().all())

    check("expect_column_values_to_be_between:tenure",
          ((df["tenure"] >= 0) & (df["tenure"] <= 120)).all())

    check("expect_column_values_to_be_between:MonthlyCharges",
          ((df["MonthlyCharges"] >= 0) & (df["MonthlyCharges"] <= 200)).all())

    check("expect_column_values_to_be_between:TotalCharges",
      (df["TotalCharges"].dropna() >= 0).all())

    print("Validating data consistency...")
    pair_check = (df["TotalCharges"] >= df["MonthlyCharges"]).mean() >= 0.95
    check("expect_column_pair_values_A_to_be_greater_than_B:TotalCharges_MonthlyCharges",
          pair_check)

    total = len(total_checks_ref)
    passed = total - len(failed_expectations)
    is_valid = len(failed_expectations) == 0

    if is_valid:
        print(f" Data validation PASSED: {passed}/{total} checks successful")
    else:
        print(f" Data validation FAILED: {len(failed_expectations)}/{total} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return is_valid, failed_expectations