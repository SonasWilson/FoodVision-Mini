import torch
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


def create_model(num_classes):
    weights = EfficientNet_B2_Weights.DEFAULT
    model = efficientnet_b2(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return  model

def get_transform():
    weights = EfficientNet_B2_Weights.DEFAULT
    return weights.transforms()