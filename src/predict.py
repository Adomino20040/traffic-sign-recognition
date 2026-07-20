import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model
from src.transforms import GTSRB_TRANSFORM
from src.gtsrb_classes import GTSRB_CLASSES


def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = GTSRB_TRANSFORM(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = probs.topk(1, dim=1)

    class_id = top_class.item()
    confidence = top_prob.item()
    return class_id, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = load_model()
    class_id, confidence = predict_image(model, image_path)
    print(f"Predicted: {GTSRB_CLASSES[class_id]} ({confidence:.1%})")
