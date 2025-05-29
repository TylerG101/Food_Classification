import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes:int=101):

    """Creates an EfficientNetB2 feature extractor model and transforms.
    Takes in:
        num_classes: number of classes we can have as outputs.
    Returns:
        model: EffNetB2 feature extractor model and transforms
    """
    #Create an EffNetb2 model (and associated transforms) we can load a state dict into
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    #Freeze base layers so we can change the classifier
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes)
    )

    return model, transforms