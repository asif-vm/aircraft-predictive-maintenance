# Aircraft Predictive Maintenance

[![CI](https://github.com/asif-vm/aircraft-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/asif-vm/aircraft-predictive-maintenance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predict remaining useful life (RUL) from multivariate engine sensor readings and translate predictions into maintenance priority.

## Why this belongs in an AI/ML portfolio

It demonstrates leakage-safe group splitting, regression evaluation in operational units, critical-window recall, explainability artifacts, API serving, Docker and CI. It is deliberately different from RAG and financial forecasting.

## Data and stack

- Free [NASA PCoE C-MAPSS turbofan run-to-failure dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (FD001)
- pandas, scikit-learn and permutation importance
- FastAPI, Streamlit, Docker, pytest and GitHub Actions
- Optional local MLflow tracking and PSI-based feature-drift reports
- Deterministic C-MAPSS-shaped demo data for immediate offline execution

## Run

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python train.py
pytest -q
uvicorn api:app --reload
streamlit run app.py
```

For experiment tracking and drift monitoring:

```bash
pip install -r requirements-mlops.txt
python train.py --real --mlflow
python monitor.py data/train_FD001.csv
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Run `python train.py --real` to download and train on NASA FD001. Engines—not random rows—are separated between train and test to prevent leakage.

## Architecture

1. Ingest NASA FD001 or generate a schema-compatible demonstration set.
2. Calculate capped RUL from each engine's final observed cycle.
3. Split by engine ID so one engine never appears in both train and test.
4. Train and evaluate RUL in cycles plus recall inside the 30-cycle critical window.
5. Save the model, metrics, feature importance and reference sample; log optional MLflow runs and monitor PSI drift before FastAPI serving.

## Verified FD001 result

- 100 engines total: 80 train and 20 completely unseen test engines
- 13.13-cycle MAE and 18.75-cycle RMSE
- 80.0% recall inside the 30-cycle critical-maintenance window

## Resume bullets

- Trained a remaining-useful-life model on **100 NASA turbofan engines** using engine-level holdout validation, achieving **13.13-cycle MAE** across 20 unseen engines without cross-engine leakage.
- Designed maintenance-risk evaluation around a **30-cycle critical window**, attaining **80% critical recall** and translating predictions into actionable priority levels.
- Productionized inference through FastAPI and Docker with automated model tests, MLflow experiment tracking, explainability artifacts and PSI-based drift monitoring.

## Interview questions

1. **Why split by engine instead of rows?** Random rows leak each engine's degradation pattern into both train and test and inflate performance.
2. **Why cap RUL?** Early-life sensor behavior often does not support distinguishing very large RUL values; capping reduces noise and follows common C-MAPSS practice.
3. **Why measure critical recall?** Missing an engine close to failure is operationally more costly than flagging one slightly early.
