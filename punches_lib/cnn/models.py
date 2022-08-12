import torch
from torchvision import models

def get_model(model_class:str, pretrained:bool=False, num_classes:int=19) -> torch.nn.Module:
    '''
    Instantiate a model of the desided class.

    Parameters
    ----------
    model_class: a string indicating the model class. Must be a torchvision.models.ResNet class.
    pretrained: a boolean indicating whether to use ImageNet-pretrained weights.
    num_classes: an integer indicating the number of classes.

    Returns
    -------
    A torch.nn.Module instance.
    '''
    net = getattr(models, model_class)(pretrained=pretrained)
    net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
    return net