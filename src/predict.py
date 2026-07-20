from PIL import Image
import torch
from model import load_model
from gtsrb_classes import GTSRB_CLASSES
import sys
from torchvision.transforms import v2

# same transform the model saw during testing: resize, scale to 0-1, normalize
transform = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dim
    with torch.no_grad():   # skip gradient tracking, we're not training
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)      # highest score = best guess
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return predicted.item(), probs[0, predicted.item()].item()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide an image file path")
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    predicted_class, probability = predict(model, image_path)
    print(f"Predicted: {GTSRB_CLASSES[predicted_class]} with confidence {probability:.4f}")
