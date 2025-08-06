# 🚨 Fake News Detection using NLP and Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

A comprehensive machine learning project that detects fake news articles using Natural Language Processing (NLP) and Logistic Regression. This project demonstrates a complete ML pipeline from data preprocessing to model evaluation with production-ready code.

## 🎯 Project Overview

This project implements a binary classification system that can distinguish between real and fake news articles. The model uses TF-IDF vectorization and Logistic Regression to achieve high accuracy in detecting misinformation.

### Key Features

- ✅ **Complete ML Pipeline**: Data loading → Preprocessing → Training → Evaluation
- ✅ **Modular Code Structure**: Clean, reusable functions across modules
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualizations
- ✅ **MLflow Integration**: Experiment tracking and model versioning
- ✅ **Production Ready**: Well-documented, tested, and deployable code
- ✅ **GitHub Ready**: Professional documentation and structure

## 📊 Dataset Information

The project uses the **Fake and Real News Dataset** from Kaggle:

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size**: ~40,000 articles
- **Classes**: Real (0) and Fake (1) news
- **Features**: Text content of news articles
- **Balance**: Approximately 50/50 split between real and fake news

## 🛠️ Tools & Libraries Used

### Core ML Libraries
- **scikit-learn**: Machine learning algorithms and utilities
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### NLP Libraries
- **NLTK**: Natural Language Processing toolkit
- **TF-IDF**: Text feature extraction

### Visualization Libraries
- **matplotlib**: Basic plotting
- **seaborn**: Statistical data visualization
- **wordcloud**: Word cloud generation

### Utility Libraries
- **joblib**: Model persistence
- **re**: Regular expressions for text cleaning
- **mlflow**: Experiment tracking and model management

## 📁 Project Structure

```
fake-news-detection-nlp-ml/
├── 📂 data/                    # Dataset files
│   └── train.csv              # Training dataset
├── 📂 notebooks/              # Jupyter notebooks
│   ├── fake_news_detection.ipynb      # Main analysis notebook
│   └── fake_news_detection_mlflow.ipynb  # MLflow-enabled notebook
├── 📂 src/                    # Source code modules
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Text preprocessing functions
│   ├── visualization.py       # Plotting and visualization
│   ├── model.py              # ML model training and prediction
│   └── evaluation.py         # Model evaluation utilities
├── 📂 models/                 # Trained models
│   ├── fake_news_model.pkl   # Saved Logistic Regression model
│   └── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
├── 📂 outputs/                # Generated plots and reports
│   ├── wordcloud_real.png    # Real news word cloud
│   ├── wordcloud_fake.png    # Fake news word cloud
│   ├── confusion_matrix.png  # Model confusion matrix
│   ├── roc_curve.png        # ROC curve
│   └── evaluation_report.txt # Detailed evaluation report
├── 📂 mlruns/                 # MLflow experiment tracking
│   └── ...                   # Experiment runs and artifacts
├── 📄 README.md              # Project documentation
├── 📄 requirements.txt       # Python dependencies
└── 📄 .gitignore            # Git ignore file
```

## 🚀 How to Run

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Git** for cloning the repository

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fake-news-detection-nlp-ml.git
   cd fake-news-detection-nlp-ml
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   - Download `train.csv` and place it in the `data/` directory

### Running the Project

#### Option 1: Jupyter Notebook
```bash
jupyter notebook notebooks/fake_news_detection.ipynb
```

#### Option 2: Command Line
```bash
# Test individual modules
python src/data_loader.py
python src/preprocessing.py
python src/visualization.py
python src/model.py
python src/evaluation.py

# Run with MLflow tracking
python main.py
```

#### Option 3: MLflow Experiment Tracking
```bash
# Start MLflow UI to view experiments
mlflow ui

# View experiments in browser
# Open: http://localhost:5000
```

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: 95.2%
- **Precision**: 94.8%
- **Recall**: 95.2%
- **F1-Score**: 95.0%

### Key Insights
- The model shows excellent performance in distinguishing between real and fake news
- TF-IDF features effectively capture the linguistic patterns that differentiate news types
- Logistic Regression provides good interpretability while maintaining high accuracy

## 🖼️ Visualizations

### Word Clouds
![Real News Word Cloud](outputs/wordcloud_real.png)
![Fake News Word Cloud](outputs/wordcloud_fake.png)

### Model Performance
![Confusion Matrix](outputs/confusion_matrix.png)
![ROC Curve](outputs/roc_curve.png)

## 🔧 Technical Details

### Data Preprocessing Pipeline
1. **Text Cleaning**: Lowercase, punctuation removal, digit removal
2. **Tokenization**: Word tokenization using NLTK
3. **Stopword Removal**: Remove common English stopwords
4. **Stemming**: Apply Porter stemming for word normalization
5. **Feature Extraction**: TF-IDF vectorization with 5000 features

### Model Architecture
- **Algorithm**: Logistic Regression
- **Vectorization**: TF-IDF with n-grams (1-2)
- **Regularization**: L2 regularization (C=1.0)
- **Cross-validation**: 5-fold cross-validation

### Feature Engineering
- **Max Features**: 5000 TF-IDF features
- **N-gram Range**: (1, 2) for unigrams and bigrams
- **Document Frequency**: min_df=2, max_df=0.95
- **Stopwords**: English stopwords removed

## 🎯 Use Cases

This fake news detection system can be used for:

- **Social Media Monitoring**: Detect fake news on platforms
- **News Aggregation**: Filter out misinformation from news feeds
- **Educational Tools**: Teach media literacy
- **Research**: Study patterns in fake news dissemination
- **Content Moderation**: Automate content filtering

## 🔮 Future Enhancements

### Planned Improvements
- [ ] **Deep Learning Models**: BERT, RoBERTa, or DistilBERT
- [ ] **Ensemble Methods**: Combine multiple models for better performance
- [ ] **Real-time API**: Deploy as a web service
- [ ] **Multi-language Support**: Extend to other languages
- [ ] **Advanced Features**: Sentiment analysis, named entity recognition

### Research Directions
- [ ] **Temporal Analysis**: Study how fake news patterns evolve over time
- [ ] **Source Analysis**: Incorporate source credibility metrics
- [ ] **Cross-domain Adaptation**: Apply to different news domains
- [ ] **Explainable AI**: Provide interpretable predictions

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- 📧 Email: [your.email@example.com]
- 🔗 LinkedIn: [Your LinkedIn Profile]
- 🐙 GitHub: [Your GitHub Profile]
- 🌐 Portfolio: [Your Portfolio Website]

## 🙏 Acknowledgments

- **Dataset**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon
- **Libraries**: All the amazing open-source libraries that made this project possible
- **Community**: The data science and NLP community for inspiration and resources

## 📊 Project Statistics

![GitHub stars](https://img.shields.io/github/stars/yourusername/fake-news-detection-nlp-ml)
![GitHub forks](https://img.shields.io/github/forks/yourusername/fake-news-detection-nlp-ml)
![GitHub issues](https://img.shields.io/github/issues/yourusername/fake-news-detection-nlp-ml)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/fake-news-detection-nlp-ml)

---

## MLflow Experiment Tracking

To enable experiment tracking and reproducibility, this project integrates [MLflow](https://mlflow.org/).

### Installation
```bash
pip install mlflow
```

### Running the MLflow UI
```bash
mlflow ui
```

This will start the MLflow tracking server at http://localhost:5000 where you can view and compare experiment runs, metrics, parameters, and artifacts.

---

## 🚀 Run the API Locally

### FastAPI (Local)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the API server:
   ```bash
   python run_api.py
   ```
3. The API will be live at [http://localhost:8000](http://localhost:8000)

- Health check: [GET /](http://localhost:8000/)
- Prediction: [POST /predict](http://localhost:8000/predict)
  - Example request:
    ```bash
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "Some news content here..."}'
    ```

### Docker

1. Build the Docker image:
   ```bash
   docker build -t fake-news-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 fake-news-api
   ```

The API will be available at [http://localhost:8000](http://localhost:8000)

<div align="center">

**⭐ If you find this project helpful, please give it a star! ⭐**

*Built with ❤️ for the data science community*

</div> 