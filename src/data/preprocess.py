import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if target_col in df.columns and pd.api.types.is_string_dtype(df[target_col]):
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df