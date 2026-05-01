# ml-signal-lifecycle-pipeline

> **ML Signal Lifecycle** | Proof of end-to-end pipeline from raw data to production model with automated retraining

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Scikit-learn](https://img.shields.io/badge/sklearn-1.4-orange) ![MLflow](https://img.shields.io/badge/MLflow-2.x-blue) ![Airflow](https://img.shields.io/badge/Airflow-2.x-red) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

## Overview
This project demonstrates the **complete ML signal lifecycle** — from raw data ingestion through feature engineering, model training, versioning, evaluation, and automated retraining triggers. Built to mirror production ML systems at scale.

## Architecture

```
Raw Data Sources
      │
      ▼
[Data Ingestion] ───► [Feature Engineering] ──► [Feature Store]
                                                         │
                                                         ▼
                                               [Training Pipeline]
                                                         │
                                                         ▼
                                              [Model Versioning (MLflow)]
                                                         │
                            ┌───────────────────────┼───────────────────────┐
                            ▼                           ▼                       ▼
                  [Evaluation Suite]          [Monitoring Dashboard]   [Retraining Trigger]
                            │                           │                       │
                            └───────────────────────┼───────────────────────┘
                                                         │
                                                         ▼
                                               [Production Deployment]
```

## Pipeline Stages

| Stage | Description | Tools |
|-------|-------------|-------|
| `ingestion/` | Multi-source data collection & validation | Great Expectations, Pandas |
| `features/` | Feature engineering & transformation | Scikit-learn, Feature-engine |
| `training/` | Model training with hyperparameter tuning | Scikit-learn, Optuna |
| `versioning/` | Experiment tracking & model registry | MLflow |
| `evaluation/` | Metrics computation & threshold checks | Custom metrics, Evidently |
| `monitoring/` | Data drift & model degradation detection | Evidently, Prometheus |
| `retraining/` | Automated trigger logic | Apache Airflow DAGs |

## Key Features

- **Data ingestion** with schema validation and quality checks
- **Feature pipelines** with reproducible transformations stored as artifacts
- **Hyperparameter optimization** using Optuna
- **Model versioning** with MLflow Tracking + Model Registry
- **Evaluation metrics**: MAE, RMSE, R², precision, recall, F1, AUC
- **Data drift detection** to trigger retraining
- **Automated retraining DAG** in Apache Airflow
- **Docker + CI/CD** for reproducible runs

## Performance Metrics

| Metric | Baseline | Tuned Model |
|--------|----------|-------------|
| MAE | 0.142 | 0.091 |
| RMSE | 0.198 | 0.134 |
| R² | 0.81 | 0.93 |
| Training Time | 45s | 3m 20s |

## Project Structure

```
ml-signal-lifecycle-pipeline/
├── ingestion/
│   ├── data_loader.py           # Multi-source ingestion
│   ├── schema_validator.py      # Great Expectations checks
│   └── data_quality.py          # Quality report generation
├── features/
│   ├── feature_engineering.py   # Signal feature transforms
│   ├── feature_store.py         # Feature registry
│   └── pipelines.py             # Sklearn pipelines
├── training/
│   ├── train.py                 # Main training entry point
│   ├── hyperparameter_tuning.py # Optuna optimization
│   └── model_factory.py         # Model registry
├── versioning/
│   ├── mlflow_tracker.py        # MLflow experiment logging
│   └── model_registry.py        # Version management
├── evaluation/
│   ├── metrics.py               # Evaluation metrics
│   └── eval_report.py           # Report generation
├── monitoring/
│   ├── drift_detector.py        # Evidently drift checks
│   └── metrics_logger.py        # Prometheus metrics
├── retraining/
│   └── airflow_dag.py           # Automated retraining DAG
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/vanias6/ml-signal-lifecycle-pipeline
cd ml-signal-lifecycle-pipeline
pip install -r requirements.txt

# Run full pipeline
python training/train.py --config configs/training_config.yaml

# Check evaluation
python evaluation/eval_report.py

# Start MLflow UI
mlflow ui --port 5000

# Trigger retraining check
python retraining/trigger_check.py
```

## Retraining Trigger Logic

The pipeline monitors:
- **Data drift score** > 0.15 (PSI threshold)
- **Model performance degradation** > 5% MAE increase
- **Time-based trigger**: weekly retraining schedule
- **Manual override** via Airflow UI

## Tech Stack

- **Scikit-learn + Optuna** — training & HPO
- **MLflow** — experiment tracking, model registry
- **Apache Airflow** — orchestration & scheduling
- **Evidently** — data drift & model monitoring
- **Great Expectations** — data quality
- **Prometheus + Grafana** — metrics dashboards
- **Docker + GitHub Actions** — CI/CD

---

*Part of Vani's Senior AI Engineer Portfolio — [github.com/vanias6](https://github.com/vanias6)*
