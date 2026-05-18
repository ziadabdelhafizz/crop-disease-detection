
import torch.nn as nn
from torchvision import models


def create_model(num_classes, freeze_backbone=True):
    """
    Creates an EfficientNet-B0 model using ImageNet pretrained weights.

    freeze_backbone=True:
        Only trains the final classifier layer first.

    freeze_backbone=False:
        Fine-tunes the whole model.
    """

    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model
