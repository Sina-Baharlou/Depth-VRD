"""
Convolutional Auto-Encoder model
"""

import torch.nn as nn
from config import IM_SCALE
from lib.depth_backbone.depth_cnn import DepthCNN, DEPTH_MODELS
from lib.object_detector import gather_res


class AEModel(nn.Module):
    """
    Convolutional Auto-Encoder for depth feature learning
    """

    def __init__(self, num_gpus=1, depth_model=None, **kwargs):
        """
        :param num_gpus: how many GPUS to use
        :param depth_model: provided architecture for depth feature extraction
        """
        super(AEModel, self).__init__()

        self.num_gpus = num_gpus

        # -- Check if the provided model is valid
        assert depth_model in DEPTH_MODELS
        self.depth_model = depth_model

        # -- Current implementation of CAE only supports alexnet
        # -- Check if the provided flags are valid for auto-encoder
        assert depth_model == "alexnet"

        # -- Initialize depth backbone (Encoder)
        self.depth_backbone = DepthCNN(depth_model="alexnet",
                                       pretrained=False)

        # -- Defined and Initialize the decoder network
        self.depth_decoder = nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(192, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 1, kernel_size=5, padding=2, bias=False),
            nn.Upsample(size=(IM_SCALE, IM_SCALE), mode='bilinear'),
        ])

        # -- Current reconstructed depth map
        self.depth_rec = None

    def forward(self, depth_imgs=None):
        """
        Forward pass for reconstructing the input
        :param depth_imgs: provided depth images
        :return depth_imgs, depth_dec
        """

        # -- Extract depth features from depth images
        depth_enc = self.depth_backbone(depth_imgs)
        # -- Reconstruct the input image using the decoder network
        depth_dec = self.depth_decoder(depth_enc)
        # -- Store the current reconstructed depth map
        self.depth_rec = depth_dec

        return depth_imgs, depth_dec

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(batch[0][9])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
