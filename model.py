# model.py
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        # Layer 1: Convolutional layer with 128 filters of size 9x9
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        # Layer 2: Convolutional layer with 64 filters of size 1x1
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        # Layer 3: Convolutional layer with 3 filters of size 5x5
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
