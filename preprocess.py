import pandas as pd
import os
from glob import glob

# Set paths
annotations_folder = "data/annotations/"
images_folder = "data/data_image/"
output_file = "data/train.csv"

# Initialize empty dataframe
all_data = pd.DataFrame()

# Read each TSV file and combine
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

# Save as CSV
all_data = all_data.rename(columns={"tweet_text": "text"})
all_data.to_csv(output_file, index=False)

print(f"✅ Dataset converted! Saved as {output_file}")
