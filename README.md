# Telco Customer Churn Prediction

A machine learning pipeline that predicts whether a telecom customer is likely to churn, using XGBoost, tracked with MLflow, and served through a FastAPI + Gradio app.

## Architecture

```
Raw CSV → validate → preprocess → feature engineering → train (XGBoost)
                                                              ↓
                                                        MLflow (mlruns/)
                                                              ↓
                                          FastAPI (/predict) + Gradio UI (/ui)
```

## Stack

| Component | Technology |
|---|---|
| Model | XGBoost |
| Experiment tracking | MLflow (local filesystem backend) |
| Data validation | Custom checks (Great-Expectations-style naming) |
| Hyperparameter tuning | Optuna (optional, `src/models/tune.py`) |
| API | FastAPI |
| UI | Gradio (mounted at `/ui`) |
| Serving | Docker |

## Project structure

```
.
├── scripts/
│   ├── run_pipeline.py            # end-to-end: load → validate → preprocess → features → train → evaluate
│   └── prepare_processed_data.py  # standalone data-prep helper
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py               # standalone training helper (not used by run_pipeline.py)
│   │   ├── evaluate.py
│   │   └── tune.py                # Optuna hyperparameter search
│   ├── serving/
│   │   └── inference.py           # loads latest MLflow model, transforms input, predicts
│   ├── utils/
│   │   ├── utils.py
│   │   └── validate_data.py
│   └── app/
│       └── main.py                # FastAPI app + mounted Gradio UI
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── eda.ipynb
├── dockerfile
└── requirements.txt
```

## Quick start

### Prerequisites

- Python 3.11+
- Docker Desktop (only needed for the containerized serving step)
- The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (or use the IBM mirror below — same data)

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the dataset

```bash
mkdir -p data/raw
curl -sL -o data/raw/Telco-Customer-Churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

### 3. Run the training pipeline

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv
```

This loads the CSV, validates it, preprocesses it, engineers features, trains an XGBoost classifier, evaluates it, and logs everything (params, metrics, the model) to MLflow under `./mlruns`. It also writes `artifacts/feature_columns.json` and `artifacts/preprocessing.pkl`, which the serving layer needs.

Useful flags: `--threshold` (decision threshold, default 0.35), `--test_size`, `--experiment` (MLflow experiment name).

### 4. Run the API + UI locally

```bash
uvicorn src.app.main:app --reload --port 8000
```

- REST API: `POST http://localhost:8000/predict`
- Interactive docs: http://localhost:8000/docs
- Gradio UI: http://localhost:8000/ui

### 5. Run it in Docker

The Docker image bundles whatever's currently in `artifacts/` and `mlruns/`, so **train first** (step 3), then:

```bash
docker build -t churn-prediction -f dockerfile .
docker run -p 8000:8000 churn-prediction
```

Same endpoints as above, now containerized.

## Running tests

```bash
pytest tests/ -v
```

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1, "MonthlyCharges": 85.0, "TotalCharges": 85.0
  }'
```

Response:
```json
{"prediction": " Likely to Churn"}
```

## Model performance (on the full 7043-row Telco dataset, threshold=0.35)

| Metric | Score |
|---|---|
| Precision | 0.484 |
| Recall | 0.832 |
| F1 | 0.612 |
| ROC AUC | 0.836 |

The low decision threshold (0.35 vs the usual 0.5) is intentional — for churn prediction, missing an at-risk customer (false negative) is typically costlier than a false alarm, so recall is favored over precision.

## License

MIT
