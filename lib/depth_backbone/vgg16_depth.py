"""Modified VGG-16 for depth feature extraction"""
from torchvision.models.vgg import VGG, vgg16, cfgs, model_urls
import torch.utils.model_zoo as model_zoo
import torch.nn as nn


def make_layers(cfg, batch_norm=False):
    layers = []
    # --  Single channel for depth images
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16_depth(pretrained=False, **kwargs):
    """VGG 16-layer depth model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))

    return model


def load_vgg(use_dropout=True, use_relu=True, use_linear=True, pretrained=True):
    """
    Modified VGG-16
    + Changed the input channel of the first conv layer (from three channels to one)
    """
    if pretrained:
        model = vgg16(pretrained=True)
    else:
        # -- load vgg16 depth version
        model = vgg16_depth()

    del model.features._modules['30']  # Get rid of the maxpool
    del model.classifier._modules['6']  # Get rid of class layer
    if not use_dropout:
        del model.classifier._modules['5']  # Get rid of dropout
        if not use_relu:
            del model.classifier._modules['4']  # Get rid of relu activation
            if not use_linear:
                del model.classifier._modules['3']  # Get rid of linear layer
    return model
