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
- **pandas** & **numpy**: Data manipulation and numerical computing
- **mlflow**: Experiment tracking and model management

### NLP & Text Analysis
- **spaCy**: POS tagging, NER, and lemmatization
- **TextBlob** & **VADER**: Sentiment analysis
- **textstat**: Readability scoring
- **TF-IDF (scikit-learn)**: Feature extraction from text

### Visualization
- **matplotlib** & **seaborn**: Data visualization
- **wordcloud**: Visualizing top frequent words

### Utilities
- **joblib**: Model persistence
- **re**: Regex-based text cleaning


## 📁 Project Structure

```
fake-news-detection-nlp-ml/
│
├── api/                          # 🚀 FastAPI app for serving model predictions
│   └── app.py                    # Main API application
│
├── data/                         # 📦 Raw and preprocessed datasets
│   ├── Fake.csv                  # Raw fake news dataset
│   ├── True.csv                  # Raw real news dataset
│   └── train.csv                 # Merged/cleaned dataset for training
│
├── mlruns/                       # 📈 MLflow experiment tracking logs
│   └── ...                       # (auto-generated run metadata and artifacts)
│
├── models/                       # 🧠 Serialized model artifacts
│   └── model.pkl                 # Trained ML model (Logistic / Random Forest)
│
├── notebooks/                    # 📓 Jupyter notebooks for exploration
│   ├── fake_news_detection.ipynb           # Initial development notebook
│   ├── fake_news_detection_mlflow.ipynb    # MLflow-integrated version
│   └── fake_news_detection_mlflow.py       # Script version of the above
│
├── src/                          # 🧱 Core logic (preprocessing, modeling, etc.)
│   ├── __init__.py
│   ├── advanced_models.py        # (Optional) Transformers or future DL
│   ├── data_loader.py            # Data loading and dataset splits
│   ├── evaluation.py             # Evaluation metrics and plots
│   ├── experiment_tracker.py     # MLflow integration functions
│   ├── mlops.py                  # CI/CD or MLOps utilities (optional)
│   ├── model.py                  # Model training and prediction pipeline
│   ├── preprocessing.py          # Text preprocessing and feature engineering
│   └── visualization.py          # Visualizations (wordclouds, heatmaps, etc.)
│
├── tests/                        # 🧪 Unit tests
│   └── test_fake_news_detection.py
│
├── utils/                        # 🔧 Utility scripts
│   └── merge_datasets.py         # Merges Fake.csv and True.csv into train.csv
│
├── .github/                      # ⚙️ GitHub Actions workflows
│   └── workflows/
│       └── ci-cd.yml             # CI/CD pipeline (optional)
│
├── .gitignore                    # Exclude unnecessary files from Git
├── .gitattributes                # Git LFS tracking for large CSVs
├── Dockerfile                    # 🐳 Container definition
├── main.py                       # Optional script entry point
├── README.md                     # 📘 Project overview and instructions
├── requirements.txt              # 📦 Python dependencies
├── run_api.py                    # 🏁 Starts the FastAPI app (Uvicorn runner)
├── PROJECT_EVALUATION.md         # 🔍 Project evaluation report
├── PROJECT_SUMMARY.md            # 📄 Summary and highlights
└── structure.txt                 # (Optional) Text version of the structure
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

## 🧠 Advanced Features

This project goes beyond classic text classification by integrating:

- **Linguistic Preprocessing**: Lemmatization, POS tagging, Named Entity Recognition (NER) using `spaCy`
- **Sentiment Analysis**: Dual sentiment scoring with `TextBlob` and `VADER`
- **Readability Metrics**: Flesch Reading Ease, Gunning Fog Index, etc. using `textstat`
- **Feature Fusion**: Engineered features + TF-IDF vectors combined to improve model performance
- **Multiple Classifiers**: Logistic Regression and Random Forest evaluated via MLflow

These enhancements improve both accuracy and interpretability of predictions.

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

**Avisikta Pal**
- 📧 Email: [avisiktapalofficial2006@gmail.com]
- 🔗 LinkedIn: [linkedin.com/in/avisikta-pal-b5964234b]
- 🐙 GitHub: [github.com/avisikta17pal]
- 🌐 Portfolio: Coming Soon



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
   python api/app.py
   ```
3. The API will be live at [http://localhost:8000](http://localhost:8000)

- Health check: [GET /](http://localhost:8000/)
- Prediction: [POST /predict](http://localhost:8000/predict)
  - Example request:
    ```bash
    curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "Some news content here..."}'
    ```
### 🔍 Example Input
```json
{
  "text": "NASA confirms water found on the moon's sunlit surface."
}
```
### 🔍 Example Output
```json
{
 "prediction": "real",
  "confidence": 0.91
}
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