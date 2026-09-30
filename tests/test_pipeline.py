import pandas as pd
import pytest

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data


def _raw_row(**overrides):
    row = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": "400.0",
        "Churn": "Yes",
    }
    row.update(overrides)
    return row


@pytest.fixture
def raw_df():
    # Every categorical column varies across rows so build_features's
    # nunique()-based binary/multi detection actually exercises each column
    # (a column that is constant across the whole dataframe is left
    # unencoded by build_features — not a scenario the real 7043-row Telco
    # dataset ever hits, but worth noting as a fixture design constraint).
    return pd.DataFrame([
        _raw_row(),
        _raw_row(
            customerID="0002-TEST", gender="Male", Partner="No", Dependents="Yes",
            PhoneService="No", MultipleLines="Yes", InternetService="DSL",
            OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes",
            TechSupport="Yes", StreamingTV="No", StreamingMovies="Yes",
            Contract="Two year", PaperlessBilling="No",
            PaymentMethod="Mailed check", Churn="No",
        ),
    ])


def test_preprocess_maps_churn_to_binary(raw_df):
    df = preprocess_data(raw_df)
    assert "customerID" not in df.columns
    assert df["Churn"].dtype.kind in "iu"
    assert set(df["Churn"].unique()) == {0, 1}


def test_preprocess_converts_total_charges_to_numeric(raw_df):
    df = preprocess_data(raw_df)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])


def test_build_features_encodes_categoricals(raw_df):
    df = preprocess_data(raw_df)
    df_enc = build_features(df, target_col="Churn")
    assert df_enc.select_dtypes(include=["object", "str"]).columns.empty
    assert "Churn" in df_enc.columns


def test_validate_telco_data_passes_on_clean_data(raw_df):
    is_valid, failed = validate_telco_data(raw_df)
    assert is_valid, failed


def test_validate_telco_data_fails_on_missing_columns():
    df = pd.DataFrame([{"customerID": "0001-TEST"}])
    is_valid, failed = validate_telco_data(df)
    assert not is_valid
    assert failed
