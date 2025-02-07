import torch
from torch.utils.data import DataLoader
from mhrn import MHRN
from twitter_dataset import CrisisMMDataset
from utils import compute_loss
from config import CONFIG

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    text_lengths = torch.tensor([len(text.nonzero()) for text in texts], dtype=torch.long)  # Compute lengths
    texts = torch.stack(texts)  # Convert to tensor
    return {"text": texts, "text_lengths": text_lengths, "image": images, "label": labels}

def train():
    dataset = CrisisMMDataset(CONFIG["data_path"], CONFIG["vocab"], max_length=50)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = MHRN(vocab_size=len(CONFIG["vocab"]), num_topics=CONFIG["num_classes"]).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    model.train()
    for epoch in range(CONFIG["num_epochs"]):
        total_loss = 0
        for batch in dataloader:
            text = batch["text"].to(CONFIG["device"])
            text_lengths = batch["text_lengths"].to(CONFIG["device"])
            images = batch["image"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(text, text_lengths, images)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Loss: {total_loss / len(dataloader)}")

if __name__ == "__main__":
    train()
