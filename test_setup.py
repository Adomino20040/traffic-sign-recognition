# test_setup.py
# Verifies that the core project dependencies are installed and importable.
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print("All imports successful!")

# Quick sanity check
x = torch.randn(3, 224, 224)
print(f"Random tensor shape: {x.shape} - OK")
print(f"CUDA available: {torch.cuda.is_available()}")
