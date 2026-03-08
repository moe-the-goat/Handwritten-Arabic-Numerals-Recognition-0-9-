# models.py -- CNN architecture for 28x28 grayscale Arabic digit classification.

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, CNN as CNN_CFG


class ArabicDigitCNN(nn.Module):
    """
    3-block CNN: 32 -> 64 -> 128 filters, BatchNorm, Global Average Pooling.
    Total ~157k parameters. Designed to be compact yet accurate on 28x28 digits.
    """

    def __init__(self, num_classes=NUM_CLASSES,
                 drop_conv=CNN_CFG["dropout_conv"],
                 drop_fc=CNN_CFG["dropout_fc"]):
        super().__init__()

        # Block 1: 28x28 -> 14x14
        self.conv1a = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1a   = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1b   = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2)
        self.drop1  = nn.Dropout2d(drop_conv)

        # Block 2: 14x14 -> 7x7
        self.conv2a = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2a   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2)
        self.drop2  = nn.Dropout2d(drop_conv)

        # Block 3: deeper features + global avg pooling -> 1x1
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3    = nn.BatchNorm2d(128)
        self.gap     = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.fc1    = nn.Linear(128, 128)
        self.drop3  = nn.Dropout(drop_fc)
        self.fc2    = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop1(self.pool1(x))

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop2(self.pool2(x))

        # Block 3 + GAP
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


def count_parameters(model):
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
