import torch
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification
import evaluate
import torchmetrics
from .data import CATEGORY_INDEX

def load_pretrained_vit():
    """
    Loads the pretrained TimeSFormer model and feature extractor.
    """
    extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/timesformer-base-finetuned-k400"
    )
    model = AutoModelForVideoClassification.from_pretrained(
        "facebook/timesformer-base-finetuned-k400"
    )

    # Move model to GPU if available
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # model.to(device)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.ones(1, device=device)
        print(x, "CUDA device found.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(x, "MPS device found.")
    else:
        device = torch.device("cpu")
        print("No GPU or MPS device found, using CPU.")
    return extractor, model, device

class Metrics:
    def __init__(self, device):
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.top5_metric = torchmetrics.classification.Accuracy(top_k=5, task="multiclass", num_classes=len(CATEGORY_INDEX)).to(device)
        self.device = device

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred

        logits, labels = eval_pred
        predictions = torch.tensor(logits).argmax(dim=1)

        # Compute Accuracy
        top1_acc = self.accuracy_metric.compute(predictions=predictions.numpy(), references=labels)["accuracy"]

        # Compute Top-5 Accuracy
        top5_acc = self.top5_metric(torch.tensor(logits).to(self.device), torch.tensor(labels).to(self.device)).item()

        # Compute F1-score (macro)
        f1 = self.f1_metric.compute(predictions=predictions.numpy(), references=labels, average="macro")["f1"]

        return {
            "accuracy": top1_acc,
            "top-5 accuracy": top5_acc,
            "f1-score": f1
        }