"""Modified SqzNet for depth feature extraction"""
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision.models.squeezenet import model_urls, Fire, SqueezeNet


class SqueezeNetDepth(nn.Module):
    """
    Modified SqzNet
    + Changed the input channel of the first conv layer (from three channels to one)
    - Removed the classifier
    """

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNetDepth, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                # -- Changed the input channel size
                nn.Conv2d(1, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                # -- Changed the input channel size
                nn.Conv2d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )

        # -- Initialize the layers with xavier weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    # -- Our truncated feed forward function
    def forward(self, x):
        x = self.features(x)
        return x

def squeezenet1_0_depth(pretrained=False, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = SqueezeNet(version=1.0, **kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_0']))
    else:
        model = SqueezeNetDepth(version=1.0, **kwargs)
    return model


def squeezenet1_1_depth(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = SqueezeNet(version=1.1, **kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    else:
        model = SqueezeNetDepth(version=1.1, **kwargs)
    return model
