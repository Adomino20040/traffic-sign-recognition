import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(2048, 128)
        self.layer2 = nn.Linear(128, 43)
    
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


def load_model():
    model = ConvNet()
    model.load_state_dict(torch.load("models/gtsrb_cnn.pt", map_location = torch.device("cpu")))
    model.eval()
    return model
