"""Modified AlexNet for depth feature extraction"""
import torch.nn as nn
from torchvision.models.alexnet import AlexNet
import torch.utils.model_zoo as model_zoo
import math

__all__ = ['AlexNetDepth', 'alexnet_depth']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetDepth(nn.Module):
    """
    Modified AlexNet
    + Changed the input channel of the first conv layer (from three channels to one)
    + Added batch normalization layer
    - Removed biases from conv layers
    - Removed the classifier
    """

    def __init__(self):
        super(AlexNetDepth, self).__init__()
        self.features = nn.Sequential(
            # -- Changed the input channels of the first conv
            # -- Removed biases from convolutional layers
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2, bias=False),
            # -- Augment the layers with Batch-Normalization
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # -- Initialize the layers with xavier weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # -- Our truncated feed forward function
    def forward(self, x):
        x = self.features(x)
        return x


def alexnet_depth(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        model = AlexNet(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
        print("Warning! The pretrained alexnet doesn't have batch-norm layers.")
    else:
        model = AlexNetDepth(**kwargs)
    return model
