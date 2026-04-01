import torch.nn as nn
from torchvision import models
from .config import NUM_CLASSES

def get_model():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model