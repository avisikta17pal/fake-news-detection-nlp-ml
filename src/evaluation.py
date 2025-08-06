"""
Model evaluation utilities for fake news detection project.
Integrated with MLflow for experiment tracking and artifact logging.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import os
import mlflow


def evaluate_model(model, X_test, y_test, vectorizer=None, save_dir='outputs', enable_mlflow=True):
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    Integrated with MLflow for experiment tracking.
    
    Args:
        model: Trained model
        X_test: Test features (text data)
        y_test: Test labels
        vectorizer: TF-IDF vectorizer (if needed for text data)
        save_dir (str): Directory to save evaluation plots
        enable_mlflow (bool): Whether to enable MLflow tracking
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print("=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Make predictions
    if vectorizer is not None:
        X_test_vectorized = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vectorized)
        y_pred_proba = model.predict_proba(X_test_vectorized)
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print(f"\n📊 EVALUATION METRICS:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Log metrics to MLflow if enabled
    if enable_mlflow and mlflow.active_run() is not None:
        mlflow.log_metrics({
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1
        })
        print("✅ Metrics logged to MLflow")
    
    # Detailed classification report
    print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
    report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'], output_dict=True)
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Save classification report as artifact
    report_path = os.path.join(save_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    if enable_mlflow and mlflow.active_run() is not None:
        mlflow.log_artifact(report_path, "evaluation")
    
    # Create and log confusion matrix
    cm_path = plot_confusion_matrix(y_test, y_pred, save_dir)
    if enable_mlflow and mlflow.active_run() is not None and cm_path:
        mlflow.log_artifact(cm_path, "evaluation")
    
    # Create and log ROC curve
    roc_path = plot_roc_curve(y_test, y_pred_proba[:, 1], save_dir)
    if enable_mlflow and mlflow.active_run() is not None and roc_path:
        mlflow.log_artifact(roc_path, "evaluation")
    
    # Create and log Precision-Recall curve
    pr_path = plot_precision_recall_curve(y_test, y_pred_proba[:, 1], save_dir)
    if enable_mlflow and mlflow.active_run() is not None and pr_path:
        mlflow.log_artifact(pr_path, "evaluation")
    
    # Create and log prediction distribution
    dist_path = plot_prediction_distribution(y_test, y_pred, y_pred_proba, save_dir)
    if enable_mlflow and mlflow.active_run() is not None and dist_path:
        mlflow.log_artifact(dist_path, "evaluation")
    
    # Store results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'y_true': y_test
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, save_dir='outputs'):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot file
    """
    print("Creating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add text annotations
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.7, f'{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print confusion matrix details
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📈 CONFUSION MATRIX DETAILS:")
    print(f"True Negatives (Real → Real):  {tn}")
    print(f"False Positives (Real → Fake): {fp}")
    print(f"False Negatives (Fake → Real): {fn}")
    print(f"True Positives (Fake → Fake):  {tp}")
    
    return plot_path


def plot_roc_curve(y_true, y_pred_proba, save_dir='outputs'):
    """
    Plot and save ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot file
    """
    print("Creating ROC curve...")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC AUC: {roc_auc:.4f}")
    return plot_path


def plot_precision_recall_curve(y_true, y_pred_proba, save_dir='outputs'):
    """
    Plot and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot file
    """
    print("Creating Precision-Recall curve...")
    
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Add baseline (random classifier)
    baseline = len(y_true[y_true == 1]) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', 
                label=f'Random classifier (precision = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'precision_recall_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-Recall AUC: {pr_auc:.4f}")
    return plot_path


def plot_prediction_distribution(y_true, y_pred, y_pred_proba, save_dir='outputs'):
    """
    Plot prediction distribution and probability histograms.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        save_dir (str): Directory to save the plot
        
    Returns:
        str: Path to the saved plot file
    """
    print("Creating prediction distribution plots...")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Prediction vs Actual
    ax1 = axes[0, 0]
    prediction_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    prediction_crosstab = pd.crosstab(prediction_df['Actual'], prediction_df['Predicted'])
    sns.heatmap(prediction_crosstab, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Prediction vs Actual', fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. Probability distribution for Real news
    ax2 = axes[0, 1]
    real_probs = y_pred_proba[y_true == 0, 1]  # Probability of being fake
    fake_probs = y_pred_proba[y_true == 1, 1]  # Probability of being fake
    
    ax2.hist(real_probs, bins=20, alpha=0.7, label='Real News', color='blue')
    ax2.hist(fake_probs, bins=20, alpha=0.7, label='Fake News', color='red')
    ax2.set_xlabel('Probability of being Fake')
    ax2.set_ylabel('Count')
    ax2.set_title('Probability Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence distribution
    ax3 = axes[1, 0]
    confidence = np.max(y_pred_proba, axis=1)
    ax3.hist(confidence, bins=20, alpha=0.7, color='green')
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error analysis
    ax4 = axes[1, 1]
    errors = y_true != y_pred
    correct = y_true == y_pred
    
    error_probs = y_pred_proba[errors, 1]
    correct_probs = y_pred_proba[correct, 1]
    
    ax4.hist(correct_probs, bins=20, alpha=0.7, label='Correct Predictions', color='green')
    ax4.hist(error_probs, bins=20, alpha=0.7, label='Incorrect Predictions', color='red')
    ax4.set_xlabel('Probability of being Fake')
    ax4.set_ylabel('Count')
    ax4.set_title('Probability Distribution by Prediction Accuracy', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'prediction_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print error analysis
    error_rate = np.mean(errors)
    print(f"\n📊 ERROR ANALYSIS:")
    print(f"Error rate: {error_rate:.4f} ({error_rate*100:.2f}%)")
    print(f"Correct predictions: {np.sum(correct)}")
    print(f"Incorrect predictions: {np.sum(errors)}")
    
    return plot_path


def generate_evaluation_report(results, save_dir='outputs'):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results (dict): Evaluation results dictionary
        save_dir (str): Directory to save the report
    """
    print("Generating evaluation report...")
    
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FAKE NEWS DETECTION MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['f1_score']:.4f}\n\n")
        
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 35 + "\n")
        f.write(classification_report(results['y_true'], results['predictions'], 
                                   target_names=['Real', 'Fake']))
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated successfully!\n")
    
    print(f"Evaluation report saved to: {report_path}")


if __name__ == "__main__":
    # Test the evaluation functions with sample data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                             n_redundant=5, n_clusters_per_class=1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a simple model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    print("Testing evaluation functions...")
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Generate report
    generate_evaluation_report(results) 