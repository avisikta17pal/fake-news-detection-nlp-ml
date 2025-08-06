"""
Data loading utilities for fake news detection project.
"""

import pandas as pd
import os


def load_data(filepath):
    """
    Load and clean the CSV dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe with null values dropped
        
    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Display basic info
        print(f"Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            print(f"Null values found: {null_counts[null_counts > 0]}")
        
        # Drop null values
        df_cleaned = df.dropna()
        
        print(f"After cleaning - Shape: {df_cleaned.shape}")
        print(f"Rows dropped: {len(df) - len(df_cleaned)}")
        
        return df_cleaned
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found!")
        print("Please ensure the dataset is placed in the data/ directory.")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the function
    data_path = "data/train.csv"
    if os.path.exists(data_path):
        df = load_data(data_path)
        print("\nFirst few rows:")
        print(df.head())
    else:
        print(f"Test file not found at {data_path}") 