import torch
import torch.nn as nn

# same ConvNet from the notebook, just pulled out so other scripts can use it
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   # detects edges, arcs, colours
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)     # halves the size: 32 -> 16 -> 8
        self.relu = nn.ReLU()           # kills negative values
        self.flatten = nn.Flatten()     # unrolls 32x8x8 into 2048
        self.layer1 = nn.Linear(2048, 128)
        self.layer2 = nn.Linear(128, 43)  # one score per sign class

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# loads saved weights into a fresh ConvNet and switches to eval mode
def load_model():
    model = ConvNet()
    model.load_state_dict(torch.load("models/gtsrb_cnn.pt", map_location=torch.device("cpu")))
    model.eval()
    return model
