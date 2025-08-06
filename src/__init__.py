"""
Fake News Detection using NLP and Machine Learning

A comprehensive machine learning project for detecting fake news articles
using Natural Language Processing and Logistic Regression.

Author: Your Name
Email: your.email@example.com
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main functions for easy access
from .data_loader import load_data
from .preprocessing import prepare_data, clean_text, get_text_statistics
from .visualization import (
    generate_wordcloud, 
    plot_label_distribution, 
    plot_article_length_distribution
)
from .model import (
    train_model, 
    load_model, 
    predict_text, 
    get_feature_importance,
    evaluate_model_performance
)
from .evaluation import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_roc_curve,
    plot_precision_recall_curve,
    generate_evaluation_report
)

__all__ = [
    # Data loading
    'load_data',
    
    # Preprocessing
    'prepare_data',
    'clean_text', 
    'get_text_statistics',
    
    # Visualization
    'generate_wordcloud',
    'plot_label_distribution',
    'plot_article_length_distribution',
    
    # Model
    'train_model',
    'load_model',
    'predict_text',
    'get_feature_importance',
    'evaluate_model_performance',
    
    # Evaluation
    'evaluate_model',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'generate_evaluation_report'
] 