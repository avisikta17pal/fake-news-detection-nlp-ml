"""
Comprehensive test suite for fake news detection system.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data
from preprocessing import clean_text, prepare_data, get_text_statistics
from model import train_model, predict_text, get_feature_importance
from evaluation import evaluate_model, plot_confusion_matrix
from advanced_models import EnsembleFakeNewsClassifier, create_advanced_feature_pipeline
from mlops import ExperimentTracker, ModelVersioning, ModelMonitoring, AITesting


class TestDataLoader:
    """Test data loading functionality."""
    
    def test_load_data_success(self):
        """Test successful data loading."""
        # Create sample data
        sample_data = {
            'text': ['Sample text 1', 'Sample text 2'],
            'label': [0, 1]
        }
        df = pd.DataFrame(sample_data)
        
        # Mock pandas read_csv
        with patch('pandas.read_csv', return_value=df):
            result = load_data('dummy_path.csv')
            assert len(result) == 2
            assert 'text' in result.columns
            assert 'label' in result.columns
    
    def test_load_data_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            load_data('nonexistent_file.csv')


class TestPreprocessing:
    """Test text preprocessing functionality."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello, World! 123"
        cleaned = clean_text(text)
        assert "hello" in cleaned
        assert "world" in cleaned
        assert "123" not in cleaned  # Numbers should be removed
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_prepare_data(self):
        """Test data preparation pipeline."""
        df = pd.DataFrame({
            'text': ['Sample text 1', 'Sample text 2'],
            'label': [0, 1]
        })
        
        X, y = prepare_data(df)
        assert len(X) == 2
        assert len(y) == 2
        assert all(label in [0, 1] for label in y)
    
    def test_get_text_statistics(self):
        """Test text statistics calculation."""
        df = pd.DataFrame({
            'text': ['Short', 'This is a longer text for testing'],
            'label': [0, 1]
        })
        
        stats = get_text_statistics(df)
        assert 'original_mean_length' in stats
        assert 'original_median_length' in stats
        assert stats['original_max_length'] > stats['original_min_length']


class TestModel:
    """Test model training and prediction functionality."""
    
    def test_train_model(self):
        """Test model training."""
        X_train = pd.Series(['text 1', 'text 2', 'text 3'])
        y_train = pd.Series([0, 1, 0])
        
        model, vectorizer = train_model(X_train, y_train)
        
        assert model is not None
        assert vectorizer is not None
        assert hasattr(model, 'predict')
        assert hasattr(vectorizer, 'transform')
    
    def test_predict_text(self):
        """Test text prediction."""
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0]])
        
        prediction, probability = predict_text("test text", mock_model, mock_vectorizer)
        
        assert prediction == 1
        assert len(probability) == 2
        assert sum(probability) == pytest.approx(1.0)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.coef_ = np.array([[0.5, -0.3, 0.8]])
        
        mock_vectorizer = Mock()
        mock_vectorizer.get_feature_names_out.return_value = ['feature1', 'feature2', 'feature3']
        
        importance = get_feature_importance(mock_model, mock_vectorizer, top_n=2)
        
        assert len(importance) == 2
        assert all(isinstance(item, tuple) for item in importance)


class TestEvaluation:
    """Test model evaluation functionality."""
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Mock model and data
        mock_model = Mock()
        mock_model.predict.return_value = [0, 1, 0, 1]
        mock_model.predict_proba.return_value = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]]
        
        X_test = pd.Series(['text 1', 'text 2', 'text 3', 'text 4'])
        y_test = pd.Series([0, 1, 0, 1])
        
        results = evaluate_model(mock_model, X_test, y_test)
        
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert all(0 <= metric <= 1 for metric in [results['accuracy'], results['precision'], results['recall'], results['f1_score']])


class TestAdvancedModels:
    """Test advanced model architectures."""
    
    def test_ensemble_classifier(self):
        """Test ensemble classifier."""
        ensemble = EnsembleFakeNewsClassifier()
        
        # Mock data
        X_train = ['text 1', 'text 2', 'text 3']
        y_train = [0, 1, 0]
        
        # Mock vectorizer
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0], [0, 1], [1, 0]])
        
        # Test fitting
        ensemble.fit(X_train, y_train, mock_vectorizer)
        assert ensemble.vectorizer is not None
    
    def test_advanced_feature_pipeline(self):
        """Test advanced feature extraction."""
        feature_extractor = create_advanced_feature_pipeline()
        
        text = "This is a test article with some content."
        features = feature_extractor(text)
        
        assert 'char_count' in features
        assert 'word_count' in features
        assert 'polarity' in features
        assert 'subjectivity' in features


class TestMLOps:
    """Test MLOps functionality."""
    
    def test_experiment_tracker(self):
        """Test experiment tracking."""
        tracker = ExperimentTracker("test-experiment")
        
        # Test parameter logging
        params = {"max_features": 5000, "C": 1.0}
        tracker.log_parameters(params)
        
        # Test metric logging
        metrics = {"accuracy": 0.95, "precision": 0.94}
        tracker.log_metrics(metrics)
        
        tracker.end_run()
    
    def test_model_versioning(self):
        """Test model versioning."""
        versioning = ModelVersioning("test_registry")
        
        # Mock model and vectorizer
        mock_model = Mock()
        mock_vectorizer = Mock()
        metadata = {"accuracy": 0.95, "precision": 0.94}
        
        # Test saving version
        version = versioning.save_model_version(mock_model, mock_vectorizer, metadata)
        assert version is not None
        
        # Test listing versions
        versions = versioning.list_versions()
        assert len(versions) > 0
    
    def test_model_monitoring(self):
        """Test model monitoring."""
        monitoring = ModelMonitoring()
        
        # Test prediction logging
        monitoring.log_prediction("test text", 1, 0.85, 1)
        
        # Test report generation
        report = monitoring.generate_monitoring_report()
        assert 'timestamp' in report
        assert 'total_predictions' in report
    
    def test_ai_testing(self):
        """Test AI testing framework."""
        testing = AITesting()
        
        # Add test case
        testing.add_test_case(
            "Test Case 1",
            "This is a test article.",
            0,
            "Should classify as real news"
        )
        
        assert len(testing.test_cases) == 1
        
        # Mock model and vectorizer for testing
        mock_model = Mock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0]])
        
        # Run tests
        results = testing.run_tests(mock_model, mock_vectorizer)
        
        assert results['total_tests'] == 1
        assert results['passed'] == 1
        assert results['failed'] == 0


class TestPerformance:
    """Test performance characteristics."""
    
    def test_model_inference_speed(self):
        """Test model inference speed."""
        import time
        
        # Mock model and vectorizer
        mock_model = Mock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]
        
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.array([[1, 0, 0]])
        
        # Test inference speed
        start_time = time.time()
        for _ in range(100):
            predict_text("test text", mock_model, mock_vectorizer)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.1  # Should be faster than 100ms per prediction
    
    def test_memory_usage(self):
        """Test memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large dataset
        large_texts = [f"Text {i} with some content for testing purposes." for i in range(1000)]
        
        # Process texts
        processed_texts = [clean_text(text) for text in large_texts]
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Create sample data
        sample_data = {
            'text': [
                "Scientists discover new species in the ocean.",
                "BREAKING: Aliens contact Earth government!",
                "New study shows exercise reduces heart disease risk.",
                "SHOCKING: Celebrities are actually robots!"
            ],
            'label': [0, 1, 0, 1]
        }
        df = pd.DataFrame(sample_data)
        
        # Test preprocessing
        X, y = prepare_data(df)
        assert len(X) == 4
        assert len(y) == 4
        
        # Test model training
        model, vectorizer = train_model(X, y)
        assert model is not None
        assert vectorizer is not None
        
        # Test prediction
        test_text = "This is a test article."
        prediction, probability = predict_text(test_text, model, vectorizer)
        assert prediction in [0, 1]
        assert len(probability) == 2
        assert sum(probability) == pytest.approx(1.0)
        
        # Test evaluation
        results = evaluate_model(model, X, y, vectorizer)
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 