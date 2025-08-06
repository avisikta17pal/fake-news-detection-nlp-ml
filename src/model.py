"""
Machine learning model utilities for fake news detection project.
Enhanced to handle advanced features including sentiment analysis, readability scores, and linguistic features.
Integrated with MLflow for experiment tracking and model management.
"""

import joblib
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')
from src.experiment_tracker import start_experiment_run


def train_model(X_train, y_train, feature_df=None, save_dir='models', model_type='logistic', 
                enable_mlflow=True, experiment_name="fake_news_detection", track_experiment=False, experiment_params=None, experiment_metrics=None, experiment_artifacts=None):
    """
    Train a model with TF-IDF vectorization and optional advanced features.
    Integrated with MLflow for experiment tracking.
    
    Args:
        X_train (pd.Series): Training text features
        y_train (pd.Series): Training labels
        feature_df (pd.DataFrame): Additional features (sentiment, readability, etc.)
        save_dir (str): Directory to save the trained model
        model_type (str): Type of model to train ('logistic' or 'random_forest')
        enable_mlflow (bool): Whether to enable MLflow tracking
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        tuple: (trained_model, tfidf_vectorizer, feature_scaler)
    """
    print("Training model...")
    print(f"Model type: {model_type}")
    print(f"Using advanced features: {feature_df is not None}")
    print(f"MLflow tracking: {'Enabled' if enable_mlflow else 'Disabled'}")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize TF-IDF vectorizer
    print("Initializing TF-IDF vectorizer...")
    tfidf_params = {
        'max_features': 5000,  # Limit features to prevent overfitting
        'ngram_range': (1, 2),  # Use unigrams and bigrams
        'min_df': 2,  # Minimum document frequency
        'max_df': 0.95,  # Maximum document frequency
        'stop_words': 'english'
    }
    
    tfidf_vectorizer = TfidfVectorizer(**tfidf_params)
    
    # Fit and transform the training data
    print("Vectorizing training data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Number of features: {len(tfidf_vectorizer.get_feature_names_out())}")
    
    # Combine TF-IDF features with additional features if available
    if feature_df is not None and not feature_df.empty:
        print("Combining TF-IDF with additional features...")
        
        # Scale the additional features
        feature_scaler = StandardScaler()
        X_train_features_scaled = feature_scaler.fit_transform(feature_df)
        
        # Combine TF-IDF and additional features
        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_features_scaled])
        
        print(f"Combined features shape: {X_train_combined.shape}")
        print(f"TF-IDF features: {X_train_tfidf.shape[1]}")
        print(f"Additional features: {feature_df.shape[1]}")
        
    else:
        X_train_combined = X_train_tfidf.toarray()
        feature_scaler = None
        print("Using only TF-IDF features")
    
    # Initialize and train model
    print(f"Training {model_type} model...")
    
    if model_type == 'logistic':
        model_params = {
            'random_state': 42,
            'max_iter': 1000,
            'C': 1.0,  # Regularization parameter
            'solver': 'liblinear'  # Good for binary classification
        }
        model = LogisticRegression(**model_params)
    elif model_type == 'random_forest':
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Start MLflow run if enabled
    if enable_mlflow:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"{model_type}_model_training"):
            # Log parameters
            mlflow.log_params({
                "model_type": model_type,
                "tfidf_max_features": tfidf_params['max_features'],
                "tfidf_ngram_range": str(tfidf_params['ngram_range']),
                "tfidf_min_df": tfidf_params['min_df'],
                "tfidf_max_df": tfidf_params['max_df'],
                "use_advanced_features": feature_df is not None,
                "advanced_features_count": feature_df.shape[1] if feature_df is not None else 0,
                "training_samples": len(X_train),
                **model_params
            })
            
            # Train the model
            model.fit(X_train_combined, y_train)
            
            # Make predictions on training data
            y_train_pred = model.predict(X_train_combined)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log metrics
            mlflow.log_metrics({
                "train_accuracy": train_accuracy,
                "cv_mean_accuracy": cv_mean,
                "cv_std_accuracy": cv_std,
                "tfidf_features": X_train_tfidf.shape[1],
                "total_features": X_train_combined.shape[1]
            })
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_type}_model")
            
            # Log vectorizer and scaler as artifacts
            vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
            joblib.dump(tfidf_vectorizer, vectorizer_path)
            mlflow.log_artifact(vectorizer_path, "vectorizer")
            
            if feature_scaler:
                scaler_path = os.path.join(save_dir, 'feature_scaler.pkl')
                joblib.dump(feature_scaler, scaler_path)
                mlflow.log_artifact(scaler_path, "scaler")
            
            print(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
    else:
        # Train the model without MLflow
        model.fit(X_train_combined, y_train)
        
        # Make predictions on training data
        y_train_pred = model.predict(X_train_combined)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Perform cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(model, X_train_combined, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    # Save the model and vectorizer locally
    model_path = os.path.join(save_dir, f'fake_news_model_{model_type}.pkl')
    vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(save_dir, 'feature_scaler.pkl') if feature_scaler else None
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    if feature_scaler:
        joblib.dump(feature_scaler, scaler_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")
    if feature_scaler:
        print(f"Feature scaler saved to: {scaler_path}")
    
    # After training and evaluation
    if track_experiment:
        start_experiment_run(
            params=experiment_params or {},
            metrics=experiment_metrics or {},
            model=model,
            artifacts=experiment_artifacts or {},
            experiment_name=experiment_name
        )

    return model, tfidf_vectorizer, feature_scaler


def load_model(model_path='models/fake_news_model_logistic.pkl', 
               vectorizer_path='models/tfidf_vectorizer.pkl',
               scaler_path='models/feature_scaler.pkl'):
    """
    Load the trained model, vectorizer, and feature scaler.
    
    Args:
        model_path (str): Path to the saved model
        vectorizer_path (str): Path to the saved vectorizer
        scaler_path (str): Path to the saved feature scaler
        
    Returns:
        tuple: (loaded_model, loaded_vectorizer, loaded_scaler)
    """
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
        print("Model, vectorizer, and scaler loaded successfully!")
        return model, vectorizer, scaler
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model has been trained first.")
        return None, None, None


def predict_text(text, model, vectorizer, scaler=None, feature_df=None):
    """
    Predict whether a given text is fake or real news.
    
    Args:
        text (str): Input text to classify
        model: Trained model
        vectorizer: Trained TF-IDF vectorizer
        scaler: Trained feature scaler (optional)
        feature_df: Additional features (optional)
        
    Returns:
        tuple: (prediction, probability)
    """
    # Clean the text (assuming it's already cleaned or apply basic cleaning)
    if isinstance(text, str):
        # Apply basic cleaning if needed
        text_cleaned = text.lower().strip()
    else:
        text_cleaned = str(text).lower().strip()
    
    # Vectorize the text
    text_vectorized = vectorizer.transform([text_cleaned])
    
    # Combine with additional features if available
    if feature_df is not None and not feature_df.empty and scaler is not None:
        # Scale the additional features
        features_scaled = scaler.transform(feature_df)
        
        # Combine TF-IDF and additional features
        text_combined = np.hstack([text_vectorized.toarray(), features_scaled])
    else:
        text_combined = text_vectorized.toarray()
    
    # Make prediction
    prediction = model.predict(text_combined)[0]
    probability = model.predict_proba(text_combined)[0]
    
    return prediction, probability


def get_feature_importance(model, vectorizer, scaler=None, feature_df=None, top_n=20):
    """
    Get the most important features for classification.
    
    Args:
        model: Trained model
        vectorizer: Trained TF-IDF vectorizer
        scaler: Trained feature scaler (optional)
        feature_df: Additional features (optional)
        top_n (int): Number of top features to return
        
    Returns:
        list: List of tuples (feature_name, coefficient/importance)
    """
    # Get TF-IDF feature names
    tfidf_feature_names = vectorizer.get_feature_names_out()
    
    # Get additional feature names if available
    additional_feature_names = []
    if feature_df is not None and not feature_df.empty:
        additional_feature_names = list(feature_df.columns)
    
    # Combine feature names
    all_feature_names = list(tfidf_feature_names) + additional_feature_names
    
    # Get feature importance based on model type
    if hasattr(model, 'coef_'):
        # Logistic Regression
        importance_scores = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        # Random Forest
        importance_scores = model.feature_importances_
    else:
        print("Model doesn't support feature importance extraction")
        return []
    
    # Create list of (feature, importance) pairs
    feature_importance = list(zip(all_feature_names, importance_scores))
    
    # Sort by absolute importance value
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return feature_importance[:top_n]


def evaluate_model_performance(model, vectorizer, X_test, y_test, scaler=None, feature_df_test=None):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        vectorizer: Trained TF-IDF vectorizer
        X_test: Test features
        y_test: Test labels
        scaler: Trained feature scaler (optional)
        feature_df_test: Additional test features (optional)
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("Evaluating model performance...")
    
    # Vectorize test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Combine with additional features if available
    if feature_df_test is not None and not feature_df_test.empty and scaler is not None:
        # Scale the additional features
        features_scaled = scaler.transform(feature_df_test)
        
        # Combine TF-IDF and additional features
        X_test_combined = np.hstack([X_test_tfidf.toarray(), features_scaled])
    else:
        X_test_combined = X_test_tfidf.toarray()
    
    # Make predictions
    y_pred = model.predict(X_test_combined)
    y_pred_proba = model.predict_proba(X_test_combined)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    return results


if __name__ == "__main__":
    # Test the model functions with sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    sample_texts = [
        "This is a real news article about technology and innovation.",
        "Fake news article with misleading information and false claims.",
        "Another real news story about climate change and environmental issues.",
        "More fake news spreading conspiracy theories and misinformation.",
        "Real news about scientific discoveries and research findings.",
        "Fake news with sensationalist headlines and unverified sources."
    ]
    
    sample_labels = [0, 1, 0, 1, 0, 1]  # 0=Real, 1=Fake
    
    # Create sample additional features
    sample_features = pd.DataFrame({
        'textblob_polarity': [0.2, -0.3, 0.1, -0.5, 0.3, -0.4],
        'flesch_reading_ease': [65.2, 45.8, 70.1, 35.2, 68.9, 42.1],
        'pos_noun': [5, 3, 6, 2, 7, 3],
        'entity_person': [1, 0, 0, 0, 1, 0]
    })
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sample_texts, sample_labels, test_size=0.3, random_state=42
    )
    
    # Split features accordingly
    feature_train = sample_features.iloc[:len(X_train)]
    feature_test = sample_features.iloc[len(X_train):]
    
    print("Testing model training with advanced features...")
    
    # Train model with advanced features
    model, vectorizer, scaler = train_model(X_train, y_train, feature_train, model_type='logistic')
    
    # Evaluate model
    results = evaluate_model_performance(model, vectorizer, X_test, y_test, scaler, feature_test)
    
    # Test prediction
    test_text = "This is a test article to classify."
    prediction, probability = predict_text(test_text, model, vectorizer, scaler)
    print(f"\nTest prediction for '{test_text}':")
    print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
    print(f"Probability: {probability}")
    
    # Get feature importance
    top_features = get_feature_importance(model, vectorizer, scaler, feature_train, top_n=10)
    print(f"\nTop 10 most important features:")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}") 