"""
Text preprocessing utilities for fake news detection project.
Enhanced with Phase 1 improvements: lemmatization, NER, POS tagging, sentiment analysis, and readability scores.
"""

import re
import string
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import textstat
import warnings
warnings.filterwarnings('ignore')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️ spaCy model 'en_core_web_sm' not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    nlp = None

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()


def clean_text(text, use_advanced_features=True, preserve_entities=True):
    """
    Clean and preprocess text data with advanced NLP features.
    
    Args:
        text (str): Input text to clean
        use_advanced_features (bool): Whether to use advanced features (lemmatization, NER, etc.)
        preserve_entities (bool): Whether to preserve named entities
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    if not use_advanced_features or nlp is None:
        # Fallback to basic cleaning
        return text.strip()
    
    # Use spaCy for advanced processing
    doc = nlp(text)
    
    # Extract tokens based on criteria
    cleaned_tokens = []
    
    for token in doc:
        # Skip stopwords and very short tokens
        if token.is_stop or len(token.text) <= 2:
            continue
            
        # Preserve named entities if requested
        if preserve_entities and token.ent_type_:
            cleaned_tokens.append(token.text)
            continue
            
        # Use lemmatization instead of stemming
        if token.lemma_ != '-PRON-':  # Skip pronouns
            cleaned_tokens.append(token.lemma_)
        else:
            cleaned_tokens.append(token.text)
    
    # Join tokens back into text
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text.strip()


def extract_text_features(text, use_advanced_features=True):
    """
    Extract advanced text features including sentiment, readability, and linguistic features.
    
    Args:
        text (str): Input text
        use_advanced_features (bool): Whether to extract advanced features
        
    Returns:
        dict: Dictionary containing extracted features
    """
    features = {}
    
    if pd.isna(text) or text == '':
        return features
    
    text = str(text)
    
    # Basic features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    if not use_advanced_features:
        return features
    
    # Sentiment analysis using TextBlob
    try:
        blob = TextBlob(text)
        features['textblob_polarity'] = blob.sentiment.polarity
        features['textblob_subjectivity'] = blob.sentiment.subjectivity
    except:
        features['textblob_polarity'] = 0.0
        features['textblob_subjectivity'] = 0.0
    
    # Sentiment analysis using VADER
    try:
        vader_scores = vader_analyzer.polarity_scores(text)
        features['vader_compound'] = vader_scores['compound']
        features['vader_positive'] = vader_scores['pos']
        features['vader_negative'] = vader_scores['neg']
        features['vader_neutral'] = vader_scores['neu']
    except:
        features['vader_compound'] = 0.0
        features['vader_positive'] = 0.0
        features['vader_negative'] = 0.0
        features['vader_neutral'] = 0.0
    
    # Readability scores
    try:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['gunning_fog'] = textstat.gunning_fog(text)
        features['smog_index'] = textstat.smog_index(text)
        features['automated_readability_index'] = textstat.automated_readability_index(text)
        features['coleman_liau_index'] = textstat.coleman_liau_index(text)
        features['linsear_write_formula'] = textstat.linsear_write_formula(text)
        features['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
    except:
        features['flesch_reading_ease'] = 0.0
        features['flesch_kincaid_grade'] = 0.0
        features['gunning_fog'] = 0.0
        features['smog_index'] = 0.0
        features['automated_readability_index'] = 0.0
        features['coleman_liau_index'] = 0.0
        features['linsear_write_formula'] = 0.0
        features['dale_chall_readability_score'] = 0.0
    
    # Linguistic features using spaCy
    if nlp is not None:
        try:
            doc = nlp(text)
            
            # POS tag counts
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
            # Add POS features
            for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                features[f'pos_{pos.lower()}'] = pos_counts.get(pos, 0)
            
            # Named entity counts
            entity_counts = {}
            for ent in doc.ents:
                entity_type = ent.label_
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            # Add entity features
            for entity_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']:
                features[f'entity_{entity_type.lower()}'] = entity_counts.get(entity_type, 0)
            
            # Dependency features
            features['avg_dependency_distance'] = np.mean([token.dep_ != 'punct' for token in doc])
            
        except:
            # Fallback if spaCy processing fails
            for pos in ['noun', 'verb', 'adj', 'adv', 'propn']:
                features[f'pos_{pos}'] = 0
            for entity_type in ['person', 'org', 'gpe', 'date', 'money']:
                features[f'entity_{entity_type}'] = 0
            features['avg_dependency_distance'] = 0.0
    
    return features


def prepare_data(df, use_advanced_features=True, preserve_entities=True):
    """
    Prepare the dataset by cleaning text and extracting features and labels.
    
    Args:
        df (pd.DataFrame): Input dataframe with 'text' and 'label' columns
        use_advanced_features (bool): Whether to use advanced preprocessing features
        preserve_entities (bool): Whether to preserve named entities
        
    Returns:
        tuple: (X, y, feature_df) where X is the cleaned text features, y is the labels, and feature_df contains additional features
    """
    print("Preparing data...")
    print(f"Using advanced features: {use_advanced_features}")
    print(f"Preserving entities: {preserve_entities}")
    
    # Ensure required columns exist
    required_columns = ['text', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Clean the text column
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(
        lambda x: clean_text(x, use_advanced_features, preserve_entities)
    )
    
    # Remove rows with empty cleaned text
    initial_count = len(df)
    df = df[df['cleaned_text'].str.len() > 0]
    final_count = len(df)
    
    print(f"Removed {initial_count - final_count} rows with empty cleaned text")
    print(f"Final dataset size: {final_count}")
    
    # Extract additional features if requested
    feature_df = None
    if use_advanced_features:
        print("Extracting advanced text features...")
        feature_list = []
        
        for idx, row in df.iterrows():
            features = extract_text_features(row['text'], use_advanced_features)
            feature_list.append(features)
        
        feature_df = pd.DataFrame(feature_list, index=df.index)
        print(f"Extracted {len(feature_df.columns)} additional features")
    
    # Extract features and labels
    X = df['cleaned_text']
    y = df['label']
    
    # Display some statistics
    print(f"Number of samples: {len(X)}")
    print(f"Label distribution:")
    print(y.value_counts())
    
    if feature_df is not None:
        print(f"Additional features shape: {feature_df.shape}")
    
    return X, y, feature_df


def get_text_statistics(df):
    """
    Get statistics about the text data.
    
    Args:
        df (pd.DataFrame): Dataframe with 'text' column
        
    Returns:
        dict: Dictionary containing text statistics
    """
    stats = {}
    
    # Original text statistics
    text_lengths = df['text'].str.len()
    stats['original_mean_length'] = text_lengths.mean()
    stats['original_median_length'] = text_lengths.median()
    stats['original_min_length'] = text_lengths.min()
    stats['original_max_length'] = text_lengths.max()
    
    # Cleaned text statistics
    if 'cleaned_text' in df.columns:
        cleaned_lengths = df['cleaned_text'].str.len()
        stats['cleaned_mean_length'] = cleaned_lengths.mean()
        stats['cleaned_median_length'] = cleaned_lengths.median()
        stats['cleaned_min_length'] = cleaned_lengths.min()
        stats['cleaned_max_length'] = cleaned_lengths.max()
    
    return stats


if __name__ == "__main__":
    # Test the functions
    import numpy as np
    
    # Create sample data
    sample_texts = [
        "This is a SAMPLE text with some numbers 123 and punctuation!",
        "Another example with stopwords like 'the', 'is', 'a'",
        "Fake news detection using NLP and machine learning.",
        "BREAKING: Scientists discover new species in the Pacific Ocean. Dr. Smith from MIT says this is groundbreaking research."
    ]
    
    sample_df = pd.DataFrame({
        'text': sample_texts,
        'label': [0, 1, 0, 0]
    })
    
    print("Sample original texts:")
    for i, text in enumerate(sample_texts):
        print(f"{i+1}. {text}")
    
    print("\nCleaned texts (basic):")
    for i, text in enumerate(sample_texts):
        cleaned = clean_text(text, use_advanced_features=False)
        print(f"{i+1}. {cleaned}")
    
    print("\nCleaned texts (advanced):")
    for i, text in enumerate(sample_texts):
        cleaned = clean_text(text, use_advanced_features=True)
        print(f"{i+1}. {cleaned}")
    
    print("\nPreparing data with advanced features...")
    X, y, feature_df = prepare_data(sample_df, use_advanced_features=True)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    if feature_df is not None:
        print(f"Additional features shape: {feature_df.shape}")
        print(f"Additional features columns: {list(feature_df.columns)}") 