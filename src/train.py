from src.model import load_pretrained_vit

# Load the model and feature extractor
extractor, model, device = load_pretrained_vit()

# Print model architecture (optional, for debugging)
print(model)
print("extractor:", extractor)
print("device:",device)