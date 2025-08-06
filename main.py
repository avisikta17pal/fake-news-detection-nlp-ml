#!/usr/bin/env python3
"""
Main script for Fake News Detection using NLP and Machine Learning

This script demonstrates the complete pipeline from data loading to model evaluation.
Integrated with MLflow for experiment tracking and model management.

Usage:
    python main.py

Author: Avisikta Pal
Email: avisiktapalofficial2006@gmail.com
"""

import os
import sys
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Optional: Enable logging instead of print
# import logging
# logging.basicConfig(level=logging.INFO)
# log = logging.getLogger(__name__)
# def loginfo(msg): log.info(msg)

# Add src directory to path
sys.path.append('src')

# Import core modules
from data_loader import load_data
from preprocessing import prepare_data
from visualization import generate_wordcloud, plot_label_distribution, plot_article_length_distribution
from model import train_model, predict_text
from evaluation import evaluate_model, generate_evaluation_report

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

# ✅ MLflow Windows fix
mlflow.set_tracking_uri(f"file:///{Path(os.getcwd()).as_posix()}/mlruns")


def main() -> None:
    """
    Main function that runs the complete fake news detection pipeline.
    """
    print("🚨 FAKE NEWS DETECTION PIPELINE")
    print("=" * 50)

    # MLflow configuration
    ENABLE_MLFLOW = True
    EXPERIMENT_NAME = "fake_news_detection"

    if ENABLE_MLFLOW:
        mlflow.set_experiment(EXPERIMENT_NAME)
        print(f"🔧 MLflow tracking enabled for experiment: {EXPERIMENT_NAME}")
    else:
        print("⚠️ MLflow tracking disabled")

    # Step 1: Data Loading
    print("\n📊 STEP 1: LOADING DATA")
    print("-" * 30)

    data_path = "data/train.csv"
    if os.path.exists(data_path):
        df = load_data(data_path)
    else:
        print(f"❌ Dataset not found at {data_path}")
        print("📝 Creating sample data for demonstration...")

        # Create sample data
        sample_data = {
            'text': [
                "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
                "BREAKING: Aliens contact Earth government! Secret meeting held at Area 51.",
                "New study shows that regular exercise can reduce the risk of heart disease.",
                "SHOCKING: Celebrities are actually robots controlled by the government!",
                "Climate change report indicates global temperatures have risen.",
                "CONSPIRACY: The moon landing was filmed in Hollywood!"
            ],
            'label': [0, 1, 0, 1, 0, 1]  # 0=Real, 1=Fake
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Sample dataset created! Shape: {df.shape}")

    # Step 2: Data Preprocessing
    print("\n🧹 STEP 2: DATA PREPROCESSING")
    print("-" * 30)
    X, y = prepare_data(df)
    print("✅ Preprocessing completed!")

    # Step 3: Data Visualization
    print("\n📈 STEP 3: DATA VISUALIZATION")
    print("-" * 30)
    plot_label_distribution(df)
    plot_article_length_distribution(df)
    generate_wordcloud(df)
    print("✅ Visualizations completed!")

    # Step 4: Train-Test Split
    print("\n🔀 STEP 4: TRAIN-TEST SPLIT")
    print("-" * 30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Step 5: Model Training
    print("\n🤖 STEP 5: MODEL TRAINING")
    print("-" * 30)
    model, vectorizer, scaler = train_model(
        X_train, y_train,
        enable_mlflow=ENABLE_MLFLOW,
        experiment_name=EXPERIMENT_NAME
    )
    print("✅ Model training completed!")

    # Step 6: Model Evaluation
    print("\n📊 STEP 6: MODEL EVALUATION")
    print("-" * 30)
    results = evaluate_model(model, X_test, y_test, vectorizer, enable_mlflow=ENABLE_MLFLOW)
    generate_evaluation_report(results)
    print("✅ Model evaluation completed!")

    # Step 7: Sample Predictions
    print("\n🧪 STEP 7: SAMPLE PREDICTIONS")
    print("-" * 30)

    test_texts = [
        "Scientists discover new species of deep-sea creatures in the Pacific Ocean.",
        "BREAKING: Aliens contact Earth government! Secret meeting held at Area 51.",
        "New study shows that regular exercise can reduce the risk of heart disease.",
        "SHOCKING: Celebrities are actually robots controlled by the government!"
    ]

    print("\nSample predictions:")
    for i, text in enumerate(test_texts, 1):
        prediction, probability = predict_text(text, model, vectorizer)
        result = "🔴 FAKE" if prediction == 1 else "🟢 REAL"
        confidence = max(probability) * 100

        print(f"\n{i}. {result} (Confidence: {confidence:.1f}%)")
        print(f"   Text: {text}")
        print(f"   Probabilities: Real={probability[0]:.3f}, Fake={probability[1]:.3f}")

    # Step 8: Summary
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("📁 Check the following directories for outputs:")
    print("   - models/: Trained model files")
    print("   - outputs/: Visualizations and reports")

    print("\n📊 Model Performance Summary:")
    print(f"   - Accuracy: {results['accuracy']:.4f}")
    print(f"   - Precision: {results['precision']:.4f}")
    print(f"   - Recall: {results['recall']:.4f}")
    print(f"   - F1-Score: {results['f1_score']:.4f}")

    print("\n✅ All steps completed successfully!")
    print("🚀 The fake news detection system is ready to use!")


if __name__ == "__main__":
    main()
