import os
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(f"file://{os.path.abspath(MLFLOW_TRACKING_URI)}")

def start_experiment_run(params: dict, metrics: dict, model, artifacts: dict, experiment_name="FakeNewsDetection"):
    """
    Start an MLflow experiment run, log parameters, metrics, and artifacts.
    Args:
        params (dict): Parameters to log (model type, TF-IDF config, etc.)
        metrics (dict): Metrics to log (accuracy, precision, recall, F1-score)
        model: Trained model object
        artifacts (dict): Dict of artifact paths {'confusion_matrix': ..., 'classification_report': ..., 'model': ...}
        experiment_name (str): MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Log parameters
        for k, v in params.items():
            mlflow.log_param(k, v)
        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        # Log artifacts
        if 'confusion_matrix' in artifacts and os.path.exists(artifacts['confusion_matrix']):
            mlflow.log_artifact(artifacts['confusion_matrix'], artifact_path="plots")
        if 'classification_report' in artifacts and os.path.exists(artifacts['classification_report']):
            mlflow.log_artifact(artifacts['classification_report'], artifact_path="reports")
        # Log model
        if model is not None:
            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception:
                import joblib
                joblib.dump(model, "temp_model.pkl")
                mlflow.log_artifact("temp_model.pkl", artifact_path="model")
                os.remove("temp_model.pkl")
        # Log any additional artifacts
        for k, v in artifacts.items():
            if k not in ['confusion_matrix', 'classification_report', 'model'] and os.path.exists(v):
                mlflow.log_artifact(v)