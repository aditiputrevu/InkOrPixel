import torch.nn as nn
from torchvision import models
from .config import NUM_CLASSES


def get_model():
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        print("Using pretrained weights")
    except Exception as e:
        print("Could not load pretrained weights, using random init")
        model = models.resnet18(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model