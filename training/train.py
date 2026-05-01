"""Main training entry point for ML signal lifecycle pipeline."""
import argparse
import yaml
import logging
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import mlflow
import mlflow.sklearn

from ingestion.data_loader import DataLoader
from features.feature_engineering import build_feature_pipeline
from training.hyperparameter_tuning import run_optuna_study
from versioning.mlflow_tracker import MLflowTracker
from evaluation.metrics import compute_metrics
from monitoring.drift_detector import DriftDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train(config_path: str = "configs/training_config.yaml"):
    config = load_config(config_path)
    tracker = MLflowTracker(experiment_name=config["experiment"]["name"])

    logger.info("Loading and validating data...")
    loader = DataLoader(source=config["data"]["source"])
    df = loader.load()

    logger.info("Building feature pipeline...")
    feature_pipeline = build_feature_pipeline(config["features"])
    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["test_size"], random_state=42
    )

    logger.info("Running hyperparameter optimization...")
    best_params = run_optuna_study(
        X_train, y_train,
        model_type=config["training"]["model_type"],
        n_trials=config["training"]["n_trials"],
    )

    logger.info(f"Best params: {best_params}")

    with tracker.start_run(run_name="training-run") as run:
        tracker.log_params({**config["training"], **best_params})

        # Build final model
        from sklearn.ensemble import GradientBoostingRegressor
        model = Pipeline([
            ("features", feature_pipeline),
            ("model", GradientBoostingRegressor(**best_params)),
        ])
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        tracker.log_metrics(metrics)

        logger.info(f"Evaluation metrics: {metrics}")

        # Log and register model
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=config["experiment"]["model_name"],
        )

        # Check for drift (compare against baseline)
        detector = DriftDetector()
        drift_report = detector.check(X_test, reference_path=config["monitoring"]["reference_data"])
        tracker.log_artifact(drift_report.report_path)

        logger.info(f"Training complete | MAE={metrics['mae']:.4f} | R2={metrics['r2']:.4f}")
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML signal lifecycle model")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()
    train(args.config)
