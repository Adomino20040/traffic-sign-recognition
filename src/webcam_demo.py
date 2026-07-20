import sys
from pathlib import Path

import cv2
import torch
from torchvision.transforms import v2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import load_model
from src.gtsrb_classes import GTSRB_CLASSES

transform = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0)
    return input_tensor


def main():
    model = load_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = probs.topk(1, dim=1)

        class_id = top_class.item()
        confidence = top_prob.item()
        label = f"{GTSRB_CLASSES[class_id]}: {confidence:.1%}"

        cv2.putText(
            frame, label, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        cv2.imshow("Traffic Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
