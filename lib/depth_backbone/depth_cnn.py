"""
The CNN backbone for depth feature extraction
"""
import torch.nn as nn
from lib.depth_backbone.resnet_depth import resnet18_depth, resnet50_depth, resnet18_l4, resnet50_l4
from lib.depth_backbone.vgg16_depth import load_vgg
from lib.depth_backbone.alexnet_depth import alexnet_depth
from lib.depth_backbone.sqznet_depth import squeezenet1_1_depth
from lib.pytorch_misc import Flattener
import math

# -- Backbone details --
DEPTH_MODELS = ('alexnet', 'resnet18', 'resnet50', 'vgg', 'sqznet')
DEPTH_DIMS = {'alexnet': 4096, 'resnet18': 512, 'resnet50': 2048, 'vgg': 4096, 'sqznet': 1024}
DEPTH_CHANNELS = {'alexnet': 256, 'resnet18': 256, 'resnet50': 1024, 'vgg': 512, 'sqznet': 512}


class DepthCNN(nn.Module):
    """
    The feature extraction model for depth images
    Depth models: AlexNet, Resnet18, Resnet50, VGG-16, SqzNet
    """

    def __init__(self, depth_model='alexnet', pretrained=False):
        """
        :param depth_model: The specified CNN's architecture
        :param pretrained: whether to use a pre-trained CNN (on ImageNet)
        """
        super(DepthCNN, self).__init__()
        # -- Check if the provided model is valid --
        assert depth_model in DEPTH_MODELS
        self.depth_model = depth_model

        # -- Initialize depth backbone --
        print(f"Initializing depth backbone (model: {self.depth_model}, pre-trained: {pretrained})...")

        if self.depth_model == 'alexnet':
            self.features_depth = alexnet_depth(pretrained=pretrained).features

        elif self.depth_model == 'resnet18':
            self.features_depth = resnet18_depth(pretrained=pretrained)

        elif self.depth_model == 'resnet50':
            self.features_depth = resnet50_depth(pretrained=pretrained)

        elif self.depth_model == 'vgg':
            self.features_depth = load_vgg(pretrained=pretrained).features

        elif self.depth_model == 'sqznet':
            self.features_depth = squeezenet1_1_depth(pretrained=pretrained).features

        print("Initialized successfully.")

    @staticmethod
    def init_weights(module):
        """
        Initialize the classifier weights (Initialize the linear modules
        with xavier weights) (PyTorch < 0.4)
        :param module: the provided nn.module
        :return: None
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal(m.weight, gain=1.0)

    def get_classifier(self, pooling_size=7):
        """
        Get the classifier network corresponding to the specified depth feature extractor.
        The classifier is used as a relation head or classification head.
        :param pooling_size: RoI pooling size
        """
        # -- AlexNet modified classifier --
        if self.depth_model == 'alexnet':
            classifier = nn.Sequential(
                nn.Dropout(),
                # -- Changed the input size ( from [6,6] to [7,7])
                nn.Linear(256 * pooling_size ** 2, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True)
                # -- Ignore the final layer
                # nn.Linear(4096, num_classes)
            )
            self.init_weights(classifier)
            return classifier

        # -- ResNet 18 modified classifier --
        elif self.depth_model == 'resnet18':
            return nn.Sequential(
                resnet18_l4(relu_end=False, pretrained=False),
                nn.AvgPool2d(pooling_size),
                Flattener(),
            )

        # -- ResNet 50 modified classifier --
        elif self.depth_model == 'resnet50':
            return nn.Sequential(
                resnet50_l4(relu_end=False, pretrained=False),
                nn.AvgPool2d(pooling_size),
                Flattener(),
            )

        # -- VGG 16 classifier part --
        elif self.depth_model == 'vgg':
            return load_vgg(pretrained=False).classifier

        # -- SqueezeNet 1.1 modified classifier --
        elif self.depth_model == 'sqznet':
            classifier = nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(512, 1024, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(pooling_size),
                Flattener()
            )

            self.init_weights(classifier)
            return classifier

    def forward(self, depth_imgs=None):
        """
        Forward pass for feature extraction
        :param depth_imgs: input depth images
        :return: extracted feature maps
        """
        return self.features_depth(depth_imgs)
