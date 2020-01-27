"""Modified ResNet for depth feature extraction"""
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls, resnet50, resnet18

__all__ = ['ResNetDepth', 'resnet18_depth', 'resnet34_depth',
           'resnet50_depth', 'resnet101_depth', 'resnet152_depth']


class ResNetDepth(ResNet):
    """
    Modified ResNet
    + Changed the input channel of the first conv layer (from three channels to one)
    - Removed the forth residual block
    - Removed the classifier
    """

    def __init__(self, *args, three_channels=False):
        super(ResNetDepth, self).__init__(*args)

        # -- Changed the input channels of the first conv
        if not three_channels:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            # -- Initialize the first conv layer
            n = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
            self.conv1.weight.data.normal_(0, math.sqrt(2. / n))

        # -- Delete unnecessary layers
        del self.layer4
        del self.avgpool
        del self.fc

    # -- Our truncated feed forward function
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


# -- Fourth block of resnet18 (used as rel-head)
def resnet18_l4(relu_end=True, pretrained=True):
    model = resnet18(pretrained=pretrained)
    l4 = model.layer4
    if not relu_end:
        l4[-1].relu_end = False

    # -- reduce stride to have a bigger feature maps
    l4[0].conv1.stride = (1, 1)
    l4[0].downsample[0].stride = (1, 1)
    return l4


# -- Fourth block of resnet50 (used as rel-head)
def resnet50_l4(relu_end=True, pretrained=True):
    model = resnet50(pretrained=pretrained)
    l4 = model.layer4
    if not relu_end:
        l4[-1].relu_end = False

    # -- reduce stride to have a bigger feature maps
    l4[0].conv2.stride = (1, 1)
    l4[0].downsample[0].stride = (1, 1)
    return l4


def resnet18_depth(pretrained=False, **kwargs):
    """Constructs a ResNet-18 depth model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDepth(BasicBlock, [2, 2, 2, 2],
                        three_channels=pretrained, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet34_depth(pretrained=False, **kwargs):
    """Constructs a ResNet-34 depth model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDepth(BasicBlock, [3, 4, 6, 3],
                        three_channels=pretrained, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


def resnet50_depth(pretrained=False, **kwargs):
    """Constructs a ResNet-50 depth model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDepth(Bottleneck, [3, 4, 6, 3],
                        three_channels=pretrained, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101_depth(pretrained=False, **kwargs):
    """Constructs a ResNet-101 depth model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDepth(Bottleneck, [3, 4, 23, 3],
                        three_channels=pretrained, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet152_depth(pretrained=False, **kwargs):
    """Constructs a ResNet-152 depth model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDepth(Bottleneck, [3, 8, 36, 3],
                        three_channels=pretrained, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model
