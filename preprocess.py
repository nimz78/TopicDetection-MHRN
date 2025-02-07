import pandas as pd
import os
from glob import glob
from sklearn.model_selection import train_test_split

# Set paths
annotations_folder = "data/annotations/"
images_folder = "data/data_image/"
train_output = "data/train.csv"
test_output = "data/test.csv"

# Initialize empty dataframe
all_data = pd.DataFrame()

# Read each TSV file and extract text + image + label
for tsv_file in glob(os.path.join(annotations_folder, "*.tsv")):
    df = pd.read_csv(tsv_file, sep="\t", usecols=["tweet_text", "image_path", "text_info"])
    df = df.dropna()  # Remove empty rows

    # Convert "informative" -> 1, "not_informative" -> 0
    df["label"] = df["text_info"].apply(lambda x: 1 if x == "informative" else 0)

    # Update image paths
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(images_folder, x) if pd.notna(x) else None)
    df = df.dropna(subset=["image_path"])  # Drop rows without images

    # Append to master dataset
    all_data = pd.concat([all_data, df], ignore_index=True)

# Rename column for consistency
all_data = all_data.rename(columns={"tweet_text": "text"})

# Split into train (80%) and test (20%)
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Save train and test datasets
train_data.to_csv(train_output, index=False)
test_data.to_csv(test_output, index=False)

print(f"✅ Dataset created! Train: {len(train_data)} samples, Test: {len(test_data)} samples")
