import torch
from model import load_model
from gtsrb_classes import GTSRB_CLASSES
from torchvision.transforms import v2
import cv2
from PIL import Image

transform = v2.Compose([
    v2.Resize((32, 32)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

def main():
    model = load_model()
    capture = cv2.VideoCapture(0)
    snap_count = 0
    if not capture.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            cv2.putText(frame, f"{GTSRB_CLASSES[predicted.item()]}: {probs[0, predicted.item()].item():.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
            elif key == ord('s'):
                snap_count += 1
                cv2.imwrite(f"outputs/demo_capture_{snap_count}.png", frame)
                print(f"Snapshot saved: demo_capture_{snap_count}.png")
    capture.release()
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()

    