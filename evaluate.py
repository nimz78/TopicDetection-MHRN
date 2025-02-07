import torch
from torch.utils.data import DataLoader
from mhrn import MHRN
from twitter_dataset import TwitterSarcasmDataset
from utils import compute_metrics
from config import CONFIG

def evaluate():
    # Dataset and DataLoader
    dataset = TwitterSarcasmDataset(CONFIG["test_path"], CONFIG["image_folder"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Load model
    model = MHRN(num_topics=CONFIG["num_classes"]).to(CONFIG["device"])
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    model.eval()

    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(CONFIG["device"])
            attention_mask = batch["attention_mask"].to(CONFIG["device"])
            images = batch["image"].to(CONFIG["device"])
            labels = batch["label"].to(CONFIG["device"])

            outputs = model(input_ids, attention_mask, images)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels)
    print(metrics)

if __name__ == "__main__":
    evaluate()
