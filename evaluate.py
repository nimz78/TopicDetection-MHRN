import torch
from torch.utils.data import DataLoader
from mhrn import MHRN
from twitter_dataset import CrisisMMDataset
from utils import compute_metrics
from config import CONFIG
from torch.nn.utils.rnn import pad_sequence

# ✅ Collate function to handle variable-length sequences
def collate_fn(batch):
    texts = [item["text"] for item in batch]
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.int64)  # Compute sequence lengths
    texts = pad_sequence(texts, batch_first=True, padding_value=CONFIG["vocab"]["<PAD>"])  # Pad sequences
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return {
        "text": texts.to(CONFIG["device"]),
        "text_lengths": text_lengths.to(CONFIG["device"]),  # Ensure it's sent to the correct device
        "image": images.to(CONFIG["device"]),
        "label": labels.to(CONFIG["device"])
    }

def evaluate():
    # ✅ Load dataset with collate_fn
    dataset = CrisisMMDataset(CONFIG["test_path"], CONFIG["vocab"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)

    # ✅ Load the trained model
    model = MHRN(vocab_size=len(CONFIG["vocab"]), num_topics=CONFIG["num_classes"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            text = batch["text"]
            text_lengths = batch["text_lengths"]
            images = batch["image"]
            labels = batch["label"]

            outputs = model(text, text_lengths, images)  # ✅ Ensure text_lengths is included
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # ✅ Compute and print evaluation metrics
    metrics = compute_metrics(all_preds, all_labels)
    print("Evaluation Metrics:", metrics)

if __name__ == "__main__":
    evaluate()
