import torch

CONFIG = {
    "data_path": "data/train.csv",
    "test_path": "data/test.csv",  # (Optional: split train.csv)
    "image_folder": "data/data_image/",
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 2,
    "num_classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "models/mhrn_model.pth",
    "vocab": {"<PAD>": 0, "<UNK>": 1}  # Initialize vocabulary
}

