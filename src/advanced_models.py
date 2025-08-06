"""
Advanced model architectures for fake news detection.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    DistilBertTokenizer, DistilBertModel,
    RobertaTokenizer, RobertaModel
)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
import os


class BERTFakeNewsClassifier:
    """
    BERT-based classifier for fake news detection.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
    def preprocess_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and encode texts for BERT."""
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: Optional[List[str]] = None, val_labels: Optional[List[int]] = None,
              epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the BERT model."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare training data
        train_encodings = self.preprocess_text(train_texts)
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(train_labels)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fake news labels and probabilities."""
        self.model.eval()
        encodings = self.preprocess_text(texts)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(outputs.logits, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()


class EnsembleFakeNewsClassifier:
    """
    Ensemble classifier combining multiple models for better performance.
    """
    
    def __init__(self, models: Optional[List] = None):
        if models is None:
            self.models = [
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('svm', SVC(probability=True, random_state=42))
            ]
        else:
            self.models = models
        
        self.ensemble = VotingClassifier(
            estimators=self.models,
            voting='soft'  # Use probability voting
        )
        self.vectorizer = None
    
    def fit(self, X_train, y_train, vectorizer):
        """Train the ensemble model."""
        self.vectorizer = vectorizer
        X_train_vectorized = vectorizer.transform(X_train)
        self.ensemble.fit(X_train_vectorized, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        X_test_vectorized = self.vectorizer.transform(X_test)
        return self.ensemble.predict(X_test_vectorized)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        X_test_vectorized = self.vectorizer.transform(X_test)
        return self.ensemble.predict_proba(X_test_vectorized)


class DeepLearningClassifier(nn.Module):
    """
    Custom deep learning classifier for fake news detection.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super(DeepLearningClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)  # Binary classification
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use the last output
        lstm_out = lstm_out[:, -1, :]
        dropped = self.dropout(lstm_out)
        output = self.fc(dropped)
        return output


def create_advanced_feature_pipeline():
    """
    Create an advanced feature engineering pipeline.
    """
    import spacy
    from textblob import TextBlob
    import readability
    
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Installing spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    def extract_advanced_features(text: str) -> Dict[str, float]:
        """Extract advanced features from text."""
        features = {}
        
        # Basic text features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        
        # Sentiment analysis
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        # Readability scores
        try:
            readability_scores = readability.getmeasures(text, lang='en')
            features['flesch_reading_ease'] = readability_scores['readability grades']['FleschReadingEase']
            features['flesch_kincaid_grade'] = readability_scores['readability grades']['FleschKincaidGrade']
        except:
            features['flesch_reading_ease'] = 0
            features['flesch_kincaid_grade'] = 0
        
        # Named entities
        doc = nlp(text)
        features['ner_count'] = len(doc.ents)
        features['ner_types'] = len(set([ent.label_ for ent in doc.ents]))
        
        # Linguistic features
        features['noun_phrases'] = len([chunk for chunk in doc.noun_chunks])
        features['verb_count'] = len([token for token in doc if token.pos_ == 'VERB'])
        features['adj_count'] = len([token for token in doc if token.pos_ == 'ADJ'])
        
        # Capitalization features (common in fake news)
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features
    
    return extract_advanced_features


def save_advanced_model(model, vectorizer, model_path: str, vectorizer_path: str):
    """Save advanced models with proper versioning."""
    import datetime
    
    # Add timestamp to model names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_path.replace('.pkl', f'_{timestamp}.pkl')
    vectorizer_path = vectorizer_path.replace('.pkl', f'_{timestamp}.pkl')
    
    # Save models
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Save metadata
    metadata = {
        'model_type': type(model).__name__,
        'vectorizer_type': type(vectorizer).__name__,
        'created_at': timestamp,
        'version': '1.0.0'
    }
    
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Advanced model saved:")
    print(f"  Model: {model_path}")
    print(f"  Vectorizer: {vectorizer_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    # Test advanced models
    print("Testing advanced model architectures...")
    
    # Test ensemble classifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    sample_texts = [
        "Scientists discover new species in the ocean.",
        "BREAKING: Aliens contact Earth government!",
        "New study shows exercise reduces heart disease risk.",
        "SHOCKING: Celebrities are actually robots!"
    ]
    sample_labels = [0, 1, 0, 1]
    
    # Create ensemble
    ensemble = EnsembleFakeNewsClassifier()
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Test training
    X_train_vectorized = vectorizer.fit_transform(sample_texts)
    ensemble.fit(sample_texts, sample_labels, vectorizer)
    
    # Test prediction
    predictions = ensemble.predict(sample_texts)
    print(f"Ensemble predictions: {predictions}")
    
    # Test advanced features
    feature_extractor = create_advanced_feature_pipeline()
    for text in sample_texts[:2]:
        features = feature_extractor(text)
        print(f"\nAdvanced features for: {text[:50]}...")
        for key, value in features.items():
            print(f"  {key}: {value}") 