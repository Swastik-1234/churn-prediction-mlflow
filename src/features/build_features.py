import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """Convert a 2-category series to 0/1 integers using deterministic mappings."""
    valset = set(s.dropna().astype(str).unique())

    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1})
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1})
    if len(valset) == 2:
        sorted_vals = sorted(valset)
        return s.astype(str).map({sorted_vals[0]: 0, sorted_vals[1]: 1})

    return s  


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Transform raw customer data into ML-ready features."""
    df = df.copy()

    obj_cols = [c for c in df.select_dtypes(["object", "str"]).columns if c != target_col]
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols  = [c for c in obj_cols if df[c].dropna().nunique() >  2]

    
    for c in binary_cols:
        df[c] = _map_binary_series(df[c]).fillna(0).astype(int)

    bool_cols = df.select_dtypes("bool").columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df

    val