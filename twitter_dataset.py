import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

class CrisisMMDataset(Dataset):
    def __init__(self, csv_path, vocab, max_length=50):
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        if isinstance(max_length, dict):  # Check if max_length is mistakenly a dictionary
            raise TypeError(f"max_length should be an integer, but got {type(max_length)}")
        print("max_length received:", max_length)  # Debugging
        self.max_length = int(max_length)  # Ensure it's an integer

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def text_to_seq(self, text):
        if not isinstance(text, str):
            text = str(text)  # Convert to string if it's not already
        tokens = text.lower().split()[:self.max_length]  # Ensure slicing works
        seq = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        seq += [self.vocab["<PAD>"]] * (self.max_length - len(seq))  # Padding
        return torch.tensor(seq, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        image_path = row["image_path"]
        label = row["label"]

        # Debugging: Print file path
        print(f"Loading image: {image_path}")

        # Try to open the image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "text": self.text_to_seq(text),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }

