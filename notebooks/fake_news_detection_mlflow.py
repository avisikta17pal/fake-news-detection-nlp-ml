"""
Fake News Detection with Advanced NLP Features and MLflow Tracking
=================================================================

This script replicates the full pipeline from the original MLflow notebook, including:
- Data loading
- Preprocessing with advanced features (lemmatization, sentiment, readability, etc.)
- Feature extraction (TF-IDF + advanced features)
- Model training (Logistic Regression and Random Forest)
- Evaluation (accuracy, precision, recall, F1, confusion matrix, classification report)
- MLflow experiment tracking (parameters, metrics, artifacts, models)
- Sample predictions

Run as a standalone script:
    python fake_news_detection_mlflow.py
"""

# =============================
# 1. Imports and Configuration
# =============================
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_loader import load_data
from preprocessing import prepare_data, extract_text_features
from model import train_model, predict_text, get_feature_importance, evaluate_model_performance
from evaluation import generate_evaluation_report

# =============================
# 2. Configuration
# =============================
USE_ADVANCED_FEATURES = True
PRESERVE_ENTITIES = True
EXPERIMENT_NAME = "fake_news_detection_mlflow"
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
SAMPLE_DATA = {
    'text': [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean. The research team used advanced underwater drones to explore depths previously inaccessible to humans.",
        "BREAKING: Aliens contact Earth government! Secret meeting held at Area 51. Sources say they want to share advanced technology in exchange for our natural resources.",
        "New study shows that regular exercise can reduce the risk of heart disease by up to 30%. The research involved over 10,000 participants across multiple countries.",
        "SHOCKING: Celebrities are actually robots controlled by the government! Insider reveals all the secrets they don't want you to know.",
        "Climate change report indicates global temperatures have risen by 1.1°C since pre-industrial levels. Scientists warn of severe consequences if action is not taken.",
        "CONSPIRACY: The moon landing was filmed in Hollywood! NASA admits to staging the entire event to win the space race."
    ],
    'label': [0, 1, 0, 1, 0, 1]
}

# =============================
# 3. Data Loading
# =============================
def load_or_create_data():
    print("\n# === Data Loading ===")
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
    else:
        print(f"❌ Dataset not found at {DATA_PATH}. Using sample data.")
        df = pd.DataFrame(SAMPLE_DATA)
    print(f"Data shape: {df.shape}")
    return df

# =============================
# 4. Preprocessing
# =============================
def preprocess_data(df):
    print("\n# === Preprocessing ===")
    X, y, feature_df = prepare_data(df, use_advanced_features=USE_ADVANCED_FEATURES, preserve_entities=PRESERVE_ENTITIES)
    print(f"Text features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    if feature_df is not None:
        print(f"Advanced features shape: {feature_df.shape}")
    return X, y, feature_df

# =============================
# 5. Train-Test Split
# =============================
def split_data(X, y, feature_df):
    print("\n# === Train-Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    feature_df_train = feature_df.iloc[X_train.index] if feature_df is not None else None
    feature_df_test = feature_df.iloc[X_test.index] if feature_df is not None else None
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, feature_df_train, feature_df_test

# =============================
# 6. Model Training and MLflow Logging
# =============================
def train_and_log_model(model_type, X_train, y_train, feature_df_train, X_test, y_test, feature_df_test):
    print(f"\n# === Training {model_type.replace('_', ' ').title()} Model ===")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{model_type}_run"):
        model, vectorizer, scaler = train_model(
            X_train, y_train,
            feature_df=feature_df_train,
            model_type=model_type,
            enable_mlflow=True,
            experiment_name=EXPERIMENT_NAME
        )
        # Evaluation
        results = evaluate_model_performance(
            model, vectorizer, X_test, y_test,
            scaler=scaler, feature_df_test=feature_df_test
        )
        generate_evaluation_report(results)
        # Log evaluation metrics
        mlflow.log_metrics({
            'test_accuracy': results['accuracy'],
            'test_precision': results['precision'],
            'test_recall': results['recall'],
            'test_f1_score': results['f1_score']
        })
        # Log top features
        top_features = get_feature_importance(model, vectorizer, scaler, feature_df_train, top_n=20)
        feature_imp_path = f"outputs/{model_type}_top_features.txt"
        os.makedirs("outputs", exist_ok=True)
        with open(feature_imp_path, "w") as f:
            for feat, imp in top_features:
                f.write(f"{feat}: {imp:.4f}\n")
        mlflow.log_artifact(feature_imp_path, "feature_importance")
        # Log model
        mlflow.sklearn.log_model(model, f"{model_type}_model")
        print(f"MLflow run complete for {model_type}.")
    return model, vectorizer, scaler, results

# =============================
# 7. Sample Predictions
# =============================
def run_sample_predictions(model, vectorizer, scaler, feature_df=None):
    print("\n# === Sample Predictions ===")
    sample_texts = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
        "BREAKING: Aliens contact Earth government! Secret meeting held at Area 51.",
        "New study shows that regular exercise can reduce the risk of heart disease.",
        "SHOCKING: Celebrities are actually robots controlled by the government!"
    ]
    for i, text in enumerate(sample_texts, 1):
        text_features_df = None
        if feature_df is not None:
            features = extract_text_features(text, use_advanced_features=True)
            text_features_df = pd.DataFrame([features])
        prediction, probability = predict_text(text, model, vectorizer, scaler, text_features_df)
        result = "🔴 FAKE" if prediction == 1 else "🟢 REAL"
        confidence = max(probability) * 100
        print(f"\n{i}. {result} (Confidence: {confidence:.1f}%)")
        print(f"   Text: {text}")
        print(f"   Probabilities: Real={probability[0]:.3f}, Fake={probability[1]:.3f}")
        if text_features_df is not None:
            print(f"   Sentiment (TextBlob): {text_features_df.iloc[0].get('textblob_polarity', 0):.3f}")
            print(f"   Readability (Flesch): {text_features_df.iloc[0].get('flesch_reading_ease', 0):.1f}")

# =============================
# 8. Main Pipeline
# =============================
def main():
    print("\n==============================")
    print("FAKE NEWS DETECTION PIPELINE WITH MLFLOW")
    print("==============================")
    # Data
    df = load_or_create_data()
    # Preprocessing
    X, y, feature_df = preprocess_data(df)
    # Split
    X_train, X_test, y_train, y_test, feature_df_train, feature_df_test = split_data(X, y, feature_df)
    # Logistic Regression
    log_model, log_vectorizer, log_scaler, log_results = train_and_log_model(
        'logistic', X_train, y_train, feature_df_train, X_test, y_test, feature_df_test)
    # Random Forest
    rf_model, rf_vectorizer, rf_scaler, rf_results = train_and_log_model(
        'random_forest', X_train, y_train, feature_df_train, X_test, y_test, feature_df_test)
    # Sample predictions (using Logistic Regression by default)
    run_sample_predictions(log_model, log_vectorizer, log_scaler, feature_df)
    print("\n🎉 All steps completed! Check the outputs/ directory and MLflow UI for results.")

if __name__ == "__main__":
    main()