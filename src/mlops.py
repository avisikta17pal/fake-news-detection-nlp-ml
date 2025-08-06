"""
MLOps utilities for experiment tracking, model versioning, and production monitoring.
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import wandb
import structlog
import os
import json
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Configure structured logging
logger = structlog.get_logger()

class ExperimentTracker:
    """
    Comprehensive experiment tracking using MLflow and Weights & Biases.
    """
    
    def __init__(self, experiment_name: str = "fake-news-detection", 
                 tracking_uri: str = "sqlite:///mlruns.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Setup Weights & Biases
        try:
            wandb.init(project="fake-news-detection", name=experiment_name)
        except Exception as e:
            logger.warning(f"Could not initialize Weights & Biases: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log model parameters."""
        mlflow.log_params(params)
        try:
            wandb.config.update(params)
        except:
            pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log model metrics."""
        mlflow.log_metrics(metrics, step=step)
        try:
            wandb.log(metrics, step=step)
        except:
            pass
    
    def log_model(self, model, model_name: str = "fake_news_model"):
        """Log the trained model."""
        mlflow.sklearn.log_model(model, model_name)
    
    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """Log artifacts (plots, reports, etc.)."""
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        try:
            wandb.finish()
        except:
            pass


class ModelVersioning:
    """
    Model versioning and registry management.
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
    
    def save_model_version(self, model, vectorizer, metadata: Dict[str, Any], 
                          version: str = None) -> str:
        """Save a new version of the model with metadata."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = os.path.join(self.registry_path, version)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))
        joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
        
        # Save metadata
        metadata.update({
            "version": version,
            "created_at": datetime.now().isoformat(),
            "model_type": type(model).__name__,
            "vectorizer_type": type(vectorizer).__name__
        })
        
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update latest version pointer
        with open(os.path.join(self.registry_path, "latest.txt"), "w") as f:
            f.write(version)
        
        logger.info(f"Model version {version} saved successfully")
        return version
    
    def load_model_version(self, version: str = "latest"):
        """Load a specific model version."""
        if version == "latest":
            try:
                with open(os.path.join(self.registry_path, "latest.txt"), "r") as f:
                    version = f.read().strip()
            except FileNotFoundError:
                raise ValueError("No latest model version found")
        
        model_dir = os.path.join(self.registry_path, version)
        if not os.path.exists(model_dir):
            raise ValueError(f"Model version {version} not found")
        
        # Load model and vectorizer
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
        
        # Load metadata
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded model version {version}")
        return model, vectorizer, metadata
    
    def list_versions(self) -> Dict[str, Dict[str, Any]]:
        """List all available model versions."""
        versions = {}
        for item in os.listdir(self.registry_path):
            if os.path.isdir(os.path.join(self.registry_path, item)):
                metadata_path = os.path.join(self.registry_path, item, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    versions[item] = metadata
        
        return versions


class ModelMonitoring:
    """
    Production model monitoring and drift detection.
    """
    
    def __init__(self):
        self.prediction_history = []
        self.performance_history = []
    
    def log_prediction(self, text: str, prediction: int, probability: float, 
                      actual: Optional[int] = None):
        """Log a prediction for monitoring."""
        prediction_record = {
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text),
            "prediction": prediction,
            "confidence": probability,
            "actual": actual,
            "is_correct": actual is not None and prediction == actual
        }
        
        self.prediction_history.append(prediction_record)
        
        # Keep only last 1000 predictions to prevent memory issues
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
    
    def detect_data_drift(self, current_features: np.ndarray, 
                         reference_features: np.ndarray) -> Dict[str, float]:
        """Detect data drift using statistical tests."""
        from scipy import stats
        
        drift_metrics = {}
        
        # KS test for distribution drift
        if current_features.shape[1] == reference_features.shape[1]:
            for i in range(min(10, current_features.shape[1])):  # Test first 10 features
                statistic, p_value = stats.ks_2samp(
                    reference_features[:, i], 
                    current_features[:, i]
                )
                drift_metrics[f"ks_test_feature_{i}"] = p_value
        
        # Mean and variance drift
        drift_metrics["mean_drift"] = np.mean(
            np.abs(np.mean(current_features, axis=0) - np.mean(reference_features, axis=0))
        )
        drift_metrics["variance_drift"] = np.mean(
            np.abs(np.var(current_features, axis=0) - np.var(reference_features, axis=0))
        )
        
        return drift_metrics
    
    def get_performance_metrics(self, window_size: int = 100) -> Dict[str, float]:
        """Calculate performance metrics over recent predictions."""
        if len(self.prediction_history) < window_size:
            return {}
        
        recent_predictions = self.prediction_history[-window_size:]
        
        # Filter predictions with actual labels
        labeled_predictions = [p for p in recent_predictions if p["actual"] is not None]
        
        if not labeled_predictions:
            return {}
        
        y_true = [p["actual"] for p in labeled_predictions]
        y_pred = [p["prediction"] for p in labeled_predictions]
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
            "avg_confidence": np.mean([p["confidence"] for p in labeled_predictions]),
            "prediction_count": len(labeled_predictions)
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(self.prediction_history),
            "performance_metrics": self.get_performance_metrics(),
            "recent_activity": {
                "last_hour": len([p for p in self.prediction_history 
                                if (datetime.now() - datetime.fromisoformat(p["timestamp"])).seconds < 3600]),
                "last_day": len([p for p in self.prediction_history 
                               if (datetime.now() - datetime.fromisoformat(p["timestamp"])).days < 1])
            }
        }
        
        return report


class AITesting:
    """
    AI testing framework for model validation.
    """
    
    def __init__(self):
        self.test_cases = []
    
    def add_test_case(self, name: str, input_text: str, expected_output: int, 
                     description: str = ""):
        """Add a test case for model validation."""
        self.test_cases.append({
            "name": name,
            "input": input_text,
            "expected": expected_output,
            "description": description
        })
    
    def run_tests(self, model, vectorizer) -> Dict[str, Any]:
        """Run all test cases against the model."""
        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test_case in self.test_cases:
            try:
                # Make prediction
                prediction, probability = self._predict_text(test_case["input"], model, vectorizer)
                
                # Check if prediction matches expected
                passed = prediction == test_case["expected"]
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["test_results"].append({
                    "name": test_case["name"],
                    "passed": passed,
                    "expected": test_case["expected"],
                    "actual": prediction,
                    "confidence": max(probability),
                    "description": test_case["description"]
                })
                
            except Exception as e:
                results["failed"] += 1
                results["test_results"].append({
                    "name": test_case["name"],
                    "passed": False,
                    "error": str(e)
                })
        
        return results
    
    def _predict_text(self, text: str, model, vectorizer):
        """Helper method to make predictions."""
        # Basic preprocessing
        text_cleaned = text.lower().strip()
        text_vectorized = vectorizer.transform([text_cleaned])
        prediction = model.predict(text_vectorized)[0]
        probability = model.predict_proba(text_vectorized)[0]
        return prediction, probability


def setup_default_test_cases() -> AITesting:
    """Setup default test cases for fake news detection."""
    testing = AITesting()
    
    # Real news test cases
    testing.add_test_case(
        "Real News - Scientific Study",
        "A new study published in Nature shows that regular exercise can reduce the risk of heart disease by up to 30%. The research involved over 10,000 participants across multiple countries.",
        0,
        "Should classify scientific study as real news"
    )
    
    testing.add_test_case(
        "Real News - Climate Change",
        "Climate change report indicates global temperatures have risen by 1.1°C since pre-industrial levels. Scientists warn of severe consequences if action is not taken.",
        0,
        "Should classify climate change report as real news"
    )
    
    # Fake news test cases
    testing.add_test_case(
        "Fake News - Alien Contact",
        "BREAKING: Aliens contact Earth government! Secret meeting held at Area 51. Sources say they want to share advanced technology in exchange for our natural resources.",
        1,
        "Should classify alien conspiracy as fake news"
    )
    
    testing.add_test_case(
        "Fake News - Celebrity Conspiracy",
        "SHOCKING: Celebrities are actually robots controlled by the government! Insider reveals all the secrets they don't want you to know.",
        1,
        "Should classify celebrity conspiracy as fake news"
    )
    
    return testing


if __name__ == "__main__":
    # Test the MLOps components
    print("Testing MLOps components...")
    
    # Test experiment tracking
    tracker = ExperimentTracker("test-experiment")
    tracker.log_parameters({"max_features": 5000, "C": 1.0})
    tracker.log_metrics({"accuracy": 0.95, "precision": 0.94})
    tracker.end_run()
    
    # Test model versioning
    versioning = ModelVersioning()
    versions = versioning.list_versions()
    print(f"Available model versions: {list(versions.keys())}")
    
    # Test monitoring
    monitoring = ModelMonitoring()
    monitoring.log_prediction("Test article", 1, 0.85, 1)
    report = monitoring.generate_monitoring_report()
    print(f"Monitoring report: {report}")
    
    # Test AI testing
    testing = setup_default_test_cases()
    print(f"Setup {len(testing.test_cases)} test cases") 