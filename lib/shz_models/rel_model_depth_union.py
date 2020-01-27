"""
Depth-Union relation detection model
"""

import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from lib.depth_backbone.depth_cnn import DepthCNN, DEPTH_DIMS, DEPTH_CHANNELS, DEPTH_MODELS
from lib.shz_models.rel_model_base import RelModelBase
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.pytorch_misc import to_onehot, arange, xavier_init
from lib.surgery import filter_dets

MODES = ('sgdet', 'sgcls', 'predcls')


class RelModel(RelModelBase):
    """
    Depth-Union relation detection model
    """

    # -- Depth FC layer size
    FC_SIZE_DEPTH = 4096

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True,
                depth_model=None, pretrained_depth=False, **kwargs):

        """
        :param classes: object classes
        :param rel_classes: relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        :param depth_model: provided architecture for depth feature extraction
        :param pretrained_depth: Whether the depth feature extractor should be initialized with ImageNet weights
        """
        RelModelBase.__init__(self, classes, rel_classes, mode, num_gpus, require_overlap_det)

        # -- Store depth related parameters
        assert depth_model in DEPTH_MODELS
        self.depth_model = depth_model
        self.pretrained_depth = pretrained_depth
        self.depth_pooling_dim = DEPTH_DIMS[self.depth_model]
        self.depth_channels = DEPTH_CHANNELS[self.depth_model]
        self.pooling_size = 7
        self.detector = nn.Module()

        # -- Initialize depth backbone
        self.depth_backbone = DepthCNN(depth_model=self.depth_model,
                                       pretrained=self.pretrained_depth)

        # -- Union of Bounding boxes feature extractor
        self.depth_union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                                    dim=self.depth_channels)

        # -- Create a relation head which is used to carry on the feature extraction
        # from union features of depth features
        self.depth_rel_head_union = self.depth_backbone.get_classifier()

        # -- Final FC layer which predicts the relations
        self.depth_rel_out = xavier_init(nn.Linear(self.depth_pooling_dim, self.num_rels, bias=True))

        # -- Freeze the backbone (Pre-trained mode)
        if self.pretrained_depth:
            self.freeze_module(self.depth_backbone)

    def get_union_features_depth(self, features, rois, pair_inds):
        """
        Gets features of Union of Bounding Boxes
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        assert pair_inds.size(1) == 2
        uboxes = self.depth_union_boxes(features, rois, pair_inds)

        if self.depth_model not in ('resnet18', 'resnet50', 'sqznet'):
            uboxes = uboxes.view(pair_inds.size(0), -1)

        return self.depth_rel_head_union(uboxes)

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None,
                train_anchor_inds=None, return_fmap=False, depth_imgs=None):
        """
        Forward pass for relation detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: a numpy array of (h, w, scale) for each image.
        :param image_offset: offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param gt_rels: [] gt relations
        :param proposals: region proposals retrieved from file
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :param return_fmap: If the object detector must return the extracted feature maps
        :param depth_imgs: depth images [batch_size, 1, IM_SIZE, IM_SIZE]
        """

        # -- Get prior `result` object (instead of calling faster-rcnn-detector)
        result = self.get_prior_results(image_offset, gt_boxes, gt_classes, gt_rels)

        # -- Get RoI and relations
        rois, rel_inds = self.get_rois_and_rels(result, image_offset, gt_boxes, gt_classes, gt_rels)

        # -- Extract features from depth backbone
        depth_features = self.depth_backbone(depth_imgs)

        # -- Prevent the gradients from flowing back to depth backbone (Pre-trained mode)
        if self.pretrained_depth:
            depth_features = depth_features.detach()

        # -- Extract UoBB features --
        union_features = self.get_union_features_depth(depth_features, rois, rel_inds[:, 1:])

        # -- Get the final rel distances
        result.rel_dists = self.depth_rel_out(union_features)

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
