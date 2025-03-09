import torch
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification

def load_pretrained_vit():
    """
    Loads the pretrained TimeSFormer model and feature extractor.
    """
    extractor = AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

    # Move model to GPU if available
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #model.to(device)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x, "MPS device found.")
    else:
        print ("MPS device not found.")
    return extractor, model, mps_device