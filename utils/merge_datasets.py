# utils/merge_datasets.py (or run in notebook)

import pandas as pd
import os

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add label column
fake_df["label"] = 1  # Fake
true_df["label"] = 0  # Real

# Concatenate
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save as train.csv
os.makedirs("data", exist_ok=True)
df.to_csv("data/train.csv", index=False)

print("✅ train.csv created successfully!")
