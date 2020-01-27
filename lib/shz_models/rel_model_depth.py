"""
Depth relation detection model
"""

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F

from lib.depth_backbone.depth_cnn import DepthCNN, DEPTH_DIMS, DEPTH_MODELS
from lib.shz_models.rel_model_base import RelModelBase
from torchvision.ops import RoIAlign
from lib.pytorch_misc import to_onehot, arange, xavier_init
from lib.surgery import filter_dets

MODES = ('sgdet', 'sgcls', 'predcls')


class RelModel(RelModelBase):
    """
    Depth relation detection model
    """

    # -- Depth FC layer size
    FC_SIZE_DEPTH = 4096

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True,
                 depth_model=None, pretrained_depth=False,**kwargs):

        """
        :param classes: object classes
        :param rel_classes: relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: whether two objects must intersect
        :param depth_model: provided architecture for depth feature extraction
        :param pretrained_depth: Whether the depth feature extractor should be initialized with ImageNet weights
        """
        RelModelBase.__init__(self, classes, rel_classes, mode, num_gpus, require_overlap_det)

        # -- Store depth related parameters
        assert depth_model in DEPTH_MODELS
        self.depth_model = depth_model
        self.pretrained_depth = pretrained_depth
        self.depth_pooling_dim = DEPTH_DIMS[self.depth_model]
        self.pooling_size = 7
        self.detector = nn.Module()

        # -- Initialize depth backbone
        self.depth_backbone = DepthCNN(depth_model=self.depth_model,
                                       pretrained=self.pretrained_depth)

        # -- Create a relation head which is used to carry on the feature extraction
        # from RoIs of depth features
        self.depth_rel_head = self.depth_backbone.get_classifier()

        # -- Define depth features hidden layer
        self.depth_rel_hlayer = nn.Sequential(*[
            xavier_init(nn.Linear(self.depth_pooling_dim * 2,
                                  self.FC_SIZE_DEPTH)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
        ])

        # -- Final FC layer which predicts the relations
        self.depth_rel_out = xavier_init(nn.Linear(self.FC_SIZE_DEPTH, self.num_rels, bias=True))

        # -- Freeze the backbone (Pre-trained mode)
        if self.pretrained_depth:
            self.freeze_module(self.depth_backbone)

    def get_roi_features_depth(self, features, rois):
        """
        Gets ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlign((self.pooling_size, self.pooling_size), spatial_scale=1 / 16, sampling_ratio=-1)(
            features, rois)

        # -- Flatten the layer if the model is not RESNET/SQZNET
        if self.depth_model not in ('resnet18', 'resnet50', 'sqznet'):
            feature_pool = feature_pool.view(rois.size(0), -1)

        return self.depth_rel_head(feature_pool)

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None,
                train_anchor_inds=None, return_fmap=False, depth_imgs=None):
        """
        Forward pass for relation detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: a numpy array of (h, w, scale) for each image.
        :param image_offset: oOffset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param gt_rels: [] gt relations
        :param proposals: region proposals retrieved from file
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :param return_fmap: if the object detector must return the extracted feature maps
        :param depth_imgs: depth images [batch_size, 1, IM_SIZE, IM_SIZE]
        """

        # -- Get prior `result` object (instead of calling faster-rcnn-detector)
        result = self.get_prior_results(image_offset, gt_boxes, gt_classes, gt_rels)

        # -- Get RoI and relations
        rois, rel_inds = self.get_rois_and_rels(result, image_offset, gt_boxes, gt_classes, gt_rels)

        # -- Determine subject and object indices
        subj_inds = rel_inds[:, 1]
        obj_inds = rel_inds[:, 2]

        # -- Extract features from depth backbone
        depth_features = self.depth_backbone(depth_imgs)

        # -- Prevent the gradients from flowing back to depth backbone (Pre-trained mode)
        if self.pretrained_depth:
            depth_features = depth_features.detach()

        # -- Extract RoI features for relation detection
        depth_rois_features = self.get_roi_features_depth(depth_features, rois)

        # -- Create a pairwise relation vector out of location features
        rel_depth = torch.cat((depth_rois_features[subj_inds],
                               depth_rois_features[obj_inds]), 1)
        rel_depth_fc = self.depth_rel_hlayer(rel_depth)

        # -- Predict relation distances
        result.rel_dists = self.depth_rel_out(rel_depth_fc)

        # --- *** END OF ARCHITECTURE *** ---#

        # -- Prepare object predictions vector (PredCLS)
        # Assuming its predcls
        obj_labels = result.rm_obj_labels if self.training or self.mode == 'predcls' else None
        # One hot vector of objects
        result.rm_obj_dists = Variable(to_onehot(obj_labels.data, self.num_classes))
        # Indexed vector
        result.obj_preds = obj_labels if obj_labels is not None else result.rm_obj_dists[:, 1:].max(1)[1] + 1

        if self.training:
            return result

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Boxes will get fixed by filter_dets function.
        bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)
        # Filtering: Subject_Score * Pred_score * Obj_score, sorted and ranked
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)
