import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model
from src.gtsrb_classes import GTSRB_CLASSES

transform = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(model, image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

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
