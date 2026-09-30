
import os
import glob
import json
import pandas as pd
import mlflow

def _find_model():
    paths = glob.glob("./mlruns/*/models/*/artifacts")
    if paths:
        return max(paths, key=os.path.getmtime)
    raise Exception("No trained model found in ./mlruns. Please run the training pipeline first.")


try:
    MODEL_DIR = _find_model()
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    raise Exception(f"Failed to load model: {e}")



try:
    feature_file = os.path.join("artifacts", "feature_columns.json")
    with open(feature_file) as f:
        FEATURE_COLS = json.load(f)
    print(f"Loaded {len(FEATURE_COLS)} feature columns from training")
except Exception as e:
    raise Exception(f"Failed to load feature columns from artifacts/feature_columns.json: {e}")


BINARY_MAP = {
    "gender":          {"Female": 0, "Male": 1},
    "Partner":         {"No": 0, "Yes": 1},
    "Dependents":      {"No": 0, "Yes": 1},
    "PhoneService":    {"No": 0, "Yes": 1},
    "PaperlessBilling":{"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    
    obj_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df



def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])  

    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if isinstance(preds, (list, tuple)) and len(preds) == 1:  
            result = preds[0]
        else:
            result = preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    if result == 1:
        return " Likely to Churn"
    else:
        return "Not Likely to Churn"
