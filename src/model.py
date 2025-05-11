import torch
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification
import evaluate
import torchmetrics
from torchvision.models.video import r3d_18
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification
from transformers import Trainer
from torchvision.models.video import r3d_18
from .data import CATEGORY_INDEX
from typing import Optional, Union
from torchvision.models.video import VideoResNet


def get_device() -> torch.device:
    """
    Get the appropriate device for running the model.
    """
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
    return device


def load_model(model_type, num_classes=len(CATEGORY_INDEX)) -> tuple[
    Optional[AutoFeatureExtractor],
    Union[AutoModelForVideoClassification, VideoResNet],
    torch.device,
    Trainer,
]:
    device = get_device()

    if model_type == "timesformer":
        extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        model = AutoModelForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )

        return extractor, model.to(device), device, Trainer

    elif model_type == "r3d":
        model = r3d_18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        class CNNTrainer(Trainer):
            def compute_loss(
                self, model, inputs, return_outputs=False, num_items_in_batch=None
            ):
                videos = inputs["input"]
                labels = inputs["labels"]
                outputs = model(videos)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
                return (loss, outputs) if return_outputs else loss

            def prediction_step(
                self, model, inputs, prediction_loss_only, ignore_keys=None
            ):
                videos = inputs["input"]
                labels = inputs["labels"]

                model.eval()
                with torch.no_grad():
                    outputs = model(videos)
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(outputs, labels)

                if prediction_loss_only:
                    return (loss, None, None)
                else:
                    return (loss, outputs, labels)

        return None, model.to(device), device, CNNTrainer

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


class Metrics:
    def __init__(self, device):
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.top5_metric = torchmetrics.classification.Accuracy(
            top_k=5, task="multiclass", num_classes=len(CATEGORY_INDEX)
        ).to(device)
        self.device = device

    def compute_metrics(self, eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = torch.tensor(logits).argmax(dim=1)

        top1_acc = self.accuracy_metric.compute(
            predictions=predictions.numpy(), references=labels
        )["accuracy"]

        logits_tensor = torch.tensor(logits).to(self.device)
        labels_tensor = torch.tensor(labels).to(self.device)

        # Check if the shape is correct
        if logits_tensor.shape[1] != len(CATEGORY_INDEX):
            # Reshape if needed (in case the model outputs a different shape)
            num_classes = len(CATEGORY_INDEX)
            if logits_tensor.ndim > 1 and logits_tensor.shape[1] > num_classes:
                logits_tensor = logits_tensor[:, :num_classes]

        top5_acc = self.top5_metric(logits_tensor, labels_tensor).item()

        f1 = self.f1_metric.compute(
            predictions=predictions.numpy(), references=labels, average="macro"
        )["f1"]

        return {"accuracy": top1_acc, "top-5 accuracy": top5_acc, "f1-score": f1}
