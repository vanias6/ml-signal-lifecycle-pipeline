"""Airflow DAG for automated ML retraining based on drift and performance triggers."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import logging

logger = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.15
PERFORMANCE_DEGRADATION_THRESHOLD = 0.05


def check_drift_and_performance(**context) -> str:
    """Check if retraining is needed based on drift score and model performance."""
    from monitoring.drift_detector import DriftDetector
    from versioning.model_registry import ModelRegistry

    detector = DriftDetector()
    registry = ModelRegistry()

    drift_score = detector.get_latest_drift_score()
    current_mae = registry.get_production_metrics()["mae"]
    baseline_mae = registry.get_baseline_metrics()["mae"]
    performance_delta = (current_mae - baseline_mae) / baseline_mae

    logger.info(f"Drift score: {drift_score:.4f} | Performance delta: {performance_delta:.4f}")

    if drift_score > DRIFT_THRESHOLD or performance_delta > PERFORMANCE_DEGRADATION_THRESHOLD:
        logger.info("Retraining triggered!")
        return "trigger_retraining"
    else:
        logger.info("No retraining needed.")
        return "skip_retraining"


def run_retraining(**context):
    """Execute retraining pipeline."""
    from training.train import train
    metrics = train(config_path="configs/training_config.yaml")
    logger.info(f"Retraining complete: {metrics}")
    return metrics


def validate_new_model(**context):
    """Validate newly trained model before promoting to production."""
    from versioning.model_registry import ModelRegistry
    registry = ModelRegistry()
    new_model = registry.get_latest_model()
    metrics = new_model.metrics
    if metrics["mae"] < 0.1 and metrics["r2"] > 0.90:
        registry.promote_to_production(new_model)
        logger.info("New model promoted to production!")
    else:
        logger.warning(f"New model did not meet quality gates: {metrics}")


default_args = {
    "owner": "vanias6",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_signal_retraining_pipeline",
    default_args=default_args,
    description="Automated ML retraining DAG with drift and performance triggers",
    schedule_interval="@weekly",
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "retraining", "monitoring"],
) as dag:

    check_trigger = BranchPythonOperator(
        task_id="check_drift_and_performance",
        python_callable=check_drift_and_performance,
    )

    trigger_retraining = PythonOperator(
        task_id="trigger_retraining",
        python_callable=run_retraining,
    )

    validate_model = PythonOperator(
        task_id="validate_new_model",
        python_callable=validate_new_model,
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    check_trigger >> [trigger_retraining, skip_retraining]
    trigger_retraining >> validate_model >> end
    skip_retraining >> end
