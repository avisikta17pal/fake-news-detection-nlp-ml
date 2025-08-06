import pandas as pd
import os

# Load original large file
df = pd.read_csv("data/train.csv")

# Split into chunks of 25,000 rows (~<100MB)
chunk_size = 25000
base_path = "data"
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    file_name = os.path.join(base_path, f"train_part{i//chunk_size + 1}.csv")
    chunk.to_csv(file_name, index=False)
    print(f"✅ Saved {file_name}")
