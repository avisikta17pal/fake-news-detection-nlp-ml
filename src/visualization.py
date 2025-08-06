"""
Visualization utilities for fake news detection project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def generate_wordcloud(df, save_dir='outputs'):
    """
    Generate wordclouds for real and fake news articles.
    
    Args:
        df (pd.DataFrame): Dataframe with 'cleaned_text' and 'label' columns
        save_dir (str): Directory to save the wordcloud images
    """
    print("Generating wordclouds...")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate real and fake news
    real_news = df[df['label'] == 0]['cleaned_text'].str.cat(sep=' ')
    fake_news = df[df['label'] == 1]['cleaned_text'].str.cat(sep=' ')
    
    # Create wordcloud for real news
    if real_news.strip():
        wordcloud_real = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='Blues'
        ).generate(real_news)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(wordcloud_real, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Real News', fontsize=16, fontweight='bold')
        
        # Save real news wordcloud
        wordcloud_real.to_file(os.path.join(save_dir, 'wordcloud_real.png'))
        print(f"Real news wordcloud saved to {save_dir}/wordcloud_real.png")
    
    # Create wordcloud for fake news
    if fake_news.strip():
        wordcloud_fake = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='Reds'
        ).generate(fake_news)
        
        plt.subplot(1, 2, 2)
        plt.imshow(wordcloud_fake, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud - Fake News', fontsize=16, fontweight='bold')
        
        # Save fake news wordcloud
        wordcloud_fake.to_file(os.path.join(save_dir, 'wordcloud_fake.png'))
        print(f"Fake news wordcloud saved to {save_dir}/wordcloud_fake.png")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'wordclouds_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_label_distribution(df, save_dir='outputs'):
    """
    Plot the distribution of fake vs real news labels.
    
    Args:
        df (pd.DataFrame): Dataframe with 'label' column
        save_dir (str): Directory to save the plot
    """
    print("Plotting label distribution...")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Create count plot
    ax = sns.countplot(data=df, x='label', palette=['#2E8B57', '#DC143C'])
    
    # Customize the plot
    plt.title('Distribution of Fake vs Real News', fontsize=16, fontweight='bold')
    plt.xlabel('Label (0=Real, 1=Fake)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():,}', 
                   (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12)
    
    # Add percentage labels
    total = len(df)
    for i, p in enumerate(ax.patches):
        percentage = (p.get_height() / total) * 100
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + total*0.01, 
               f'({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    label_counts = df['label'].value_counts()
    print(f"Label distribution:")
    print(f"Real news (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")
    print(f"Fake news (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")


def plot_article_length_distribution(df, save_dir='outputs'):
    """
    Plot the distribution of article lengths for real vs fake news.
    
    Args:
        df (pd.DataFrame): Dataframe with 'text' and 'label' columns
        save_dir (str): Directory to save the plot
    """
    print("Plotting article length distribution...")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate text lengths
    df['text_length'] = df['text'].str.len()
    
    plt.figure(figsize=(15, 5))
    
    # Create subplots
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='text_length', hue='label', bins=50, alpha=0.7, 
                palette=['#2E8B57', '#DC143C'])
    plt.title('Article Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Text Length (characters)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(['Real', 'Fake'])
    
    # Box plot
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df, x='label', y='text_length', palette=['#2E8B57', '#DC143C'])
    plt.title('Article Length by Label', fontsize=14, fontweight='bold')
    plt.xlabel('Label (0=Real, 1=Fake)', fontsize=12)
    plt.ylabel('Text Length (characters)', fontsize=12)
    
    # Violin plot
    plt.subplot(1, 3, 3)
    sns.violinplot(data=df, x='label', y='text_length', palette=['#2E8B57', '#DC143C'])
    plt.title('Article Length Distribution (Violin)', fontsize=14, fontweight='bold')
    plt.xlabel('Label (0=Real, 1=Fake)', fontsize=12)
    plt.ylabel('Text Length (characters)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'article_length_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    real_stats = df[df['label'] == 0]['text_length'].describe()
    fake_stats = df[df['label'] == 1]['text_length'].describe()
    
    print(f"\nArticle Length Statistics:")
    print(f"Real news - Mean: {real_stats['mean']:.0f}, Median: {real_stats['50%']:.0f}")
    print(f"Fake news - Mean: {fake_stats['mean']:.0f}, Median: {fake_stats['50%']:.0f}")


def plot_training_history(history, save_dir='outputs'):
    """
    Plot training history for neural network models (if applicable).
    
    Args:
        history: Training history object
        save_dir (str): Directory to save the plot
    """
    if history is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Test the visualization functions with sample data
    sample_data = {
        'text': [
            "This is a sample real news article about technology and innovation.",
            "Fake news article with misleading information and false claims.",
            "Another real news story about climate change and environmental issues.",
            "More fake news spreading conspiracy theories and misinformation."
        ],
        'label': [0, 1, 0, 1]
    }
    
    df = pd.DataFrame(sample_data)
    df['cleaned_text'] = df['text'].apply(lambda x: x.lower().replace('.', ''))
    
    print("Testing visualization functions...")
    
    # Test label distribution
    plot_label_distribution(df)
    
    # Test article length distribution
    plot_article_length_distribution(df)
    
    # Test wordcloud (will be empty due to small sample)
    print("Note: Wordclouds will be minimal due to small sample size")
    generate_wordcloud(df) 