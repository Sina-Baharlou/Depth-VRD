"""
Depth-Fusion relation detection model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.ops import RoIAlign

from lib.depth_backbone.depth_cnn import DepthCNN, DEPTH_DIMS, DEPTH_MODELS
from lib.shz_models.rel_model_base import RelModelBase
from lib.fpn.box_utils import center_size
from lib.get_union_boxes import UnionBoxesAndFeats
from lib.object_detector import ObjectDetector, load_vgg
from lib.pytorch_misc import arange, xavier_init, ScaleLayer, Flattener
from lib.sparse_targets import FrequencyBias
from lib.surgery import filter_dets
from lib.word_vectors import obj_edge_vectors

MODES = ('sgdet', 'sgcls', 'predcls')


class RelModel(RelModelBase):
    """
    Depth-Fusion relation detection model
    """

    # -- Different components' FC layer size
    FC_SIZE_VISUAL = 512
    FC_SIZE_CLASS = 64
    FC_SIZE_LOC = 20
    FC_SIZE_DEPTH = 4096
    LOC_INPUT_SIZE = 8

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, use_vision=False,
                 require_overlap_det=True, embed_dim=200, hidden_dim=4096,
                 use_resnet=False, thresh=0.01, use_proposals=False, use_bias=True,
                 limit_vision=True, depth_model=None, pretrained_depth=False,
                 active_features=None, frozen_features=None, use_embed=False, **kwargs):

        """
        :param classes: object classes
        :param rel_classes: relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param use_vision: enable the contribution of union of bounding boxes
        :param require_overlap_det: whether two objects must intersect
        :param embed_dim: word2vec embeddings dimension
        :param hidden_dim: dimension of the fusion hidden layer
        :param use_resnet: use resnet as faster-rcnn's backbone
        :param thresh: faster-rcnn related threshold (Threshold for calling it a good box)
        :param use_proposals: whether to use region proposal candidates
        :param use_bias: enable frequency bias
        :param limit_vision: use truncated version of UoBB features
        :param depth_model: provided architecture for depth feature extraction
        :param pretrained_depth: whether the depth feature extractor should be initialized with ImageNet weights
        :param active_features: what set of features should be enabled (e.g. 'vdl' : visual, depth, and location features)
        :param frozen_features: what set of features should be frozen (e.g. 'd' : depth)
        :param use_embed: use word2vec embeddings
        """
        RelModelBase.__init__(self, classes, rel_classes, mode, num_gpus,
                              require_overlap_det, active_features, frozen_features)
        self.pooling_size = 7
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = 2048 if use_resnet else 4096
        self.use_vision = use_vision
        self.use_bias = use_bias
        self.limit_vision = limit_vision

        # -- Store depth related parameters
        assert depth_model in DEPTH_MODELS
        self.depth_model = depth_model
        self.pretrained_depth = pretrained_depth
        self.depth_pooling_dim = DEPTH_DIMS[self.depth_model]
        self.use_embed = use_embed
        self.detector = nn.Module()
        features_size = 0

        # -- Check whether ResNet is selected as faster-rcnn's backbone
        if use_resnet:
            raise ValueError("The current model does not support ResNet as the Faster-RCNN's backbone.")

        """ *** DIFFERENT COMPONENTS OF THE PROPOSED ARCHITECTURE *** 
        This is the part where the different components of the proposed relation detection 
        architecture are defined. In the case of RGB images, we have class probability distribution
        features, visual features, and the location ones. If we are considering depth images as well,
        we augment depth features too. """

        # -- Visual features
        if self.has_visual:
            # -- Define faster R-CNN network and it's related feature extractors
            self.detector = ObjectDetector(
                classes=classes,
                mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
                use_resnet=use_resnet,
                thresh=thresh,
                max_per_img=64,
            )
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier

            # -- Define union features
            if self.use_vision:
                # -- UoBB pooling module
                self.union_boxes = UnionBoxesAndFeats(pooling_size=self.pooling_size, stride=16,
                                                      dim=1024 if use_resnet else 512)

                # -- UoBB feature extractor
                roi_fmap = [
                    Flattener(),
                    load_vgg(use_dropout=False, use_relu=False, use_linear=self.hidden_dim == 4096,
                             pretrained=False).classifier,
                ]
                if self.hidden_dim != 4096:
                    roi_fmap.append(nn.Linear(4096, self.hidden_dim))
                self.roi_fmap = nn.Sequential(*roi_fmap)

            # -- Define visual features hidden layer
            self.visual_hlayer = nn.Sequential(*[
                xavier_init(nn.Linear(self.obj_dim * 2, self.FC_SIZE_VISUAL)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.8)
            ])
            self.visual_scale = ScaleLayer(1.0)
            features_size += self.FC_SIZE_VISUAL

        # -- Location features
        if self.has_loc:
            # -- Define location features hidden layer
            self.location_hlayer = nn.Sequential(*[
                xavier_init(nn.Linear(self.LOC_INPUT_SIZE, self.FC_SIZE_LOC)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            self.location_scale = ScaleLayer(1.0)
            features_size += self.FC_SIZE_LOC

        # -- Class features
        if self.has_class:
            if self.use_embed:
                # -- Define class embeddings
                embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.embed_dim)
                self.obj_embed = nn.Embedding(self.num_classes, self.embed_dim)
                self.obj_embed.weight.data = embed_vecs.clone()

            classme_input_dim = self.embed_dim if self.use_embed else self.num_classes
            # -- Define Class features hidden layer
            self.classme_hlayer = nn.Sequential(*[
                xavier_init(nn.Linear(classme_input_dim * 2, self.FC_SIZE_CLASS)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            self.classme_scale = ScaleLayer(1.0)
            features_size += self.FC_SIZE_CLASS

        # -- Depth features
        if self.has_depth:
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
            self.depth_scale = ScaleLayer(1.0)
            features_size += self.FC_SIZE_DEPTH

        # -- Initialize frequency bias if needed
        if self.use_bias:
            self.freq_bias = FrequencyBias()

        # -- *** Fusion layer *** --
        # -- A hidden layer for concatenated features (fusion features)
        self.fusion_hlayer = nn.Sequential(*[
            xavier_init(nn.Linear(features_size, self.hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)])

        # -- Final FC layer which predicts the relations
        self.rel_out = xavier_init(nn.Linear(self.hidden_dim, self.num_rels, bias=True))

        # -- Freeze the user specified features
        if self.frz_visual:
            self.freeze_module(self.detector)
            self.freeze_module(self.roi_fmap_obj)
            self.freeze_module(self.visual_hlayer)
            if self.use_vision:
                self.freeze_module(self.roi_fmap)
                self.freeze_module(self.union_boxes.conv)

        if self.frz_class:
            self.freeze_module(self.classme_hlayer)

        if self.frz_loc:
            self.freeze_module(self.location_hlayer)

        if self.frz_depth:
            self.freeze_module(self.depth_backbone)
            self.freeze_module(self.depth_rel_head)
            self.freeze_module(self.depth_rel_hlayer)

    def get_roi_features(self, features, rois):
        """
        Gets ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlign((self.pooling_size, self.pooling_size), spatial_scale=1 / 16, sampling_ratio=-1)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def get_union_features(self, features, rois, pair_inds):
        """
        Gets UoBB features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4]
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :param pair_inds: inds to use when predicting
        :return: UoBB features
        """
        assert pair_inds.size(1) == 2
        uboxes = self.union_boxes(features, rois, pair_inds)
        return self.roi_fmap(uboxes)

    def get_roi_features_depth(self, features, rois):
        """
        Gets ROI features (depth)
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

    @staticmethod
    def get_loc_features(boxes, subj_inds, obj_inds):
        """
        Calculate the scale-invariant location feature
        :param boxes: ground-truth/detected boxes
        :param subj_inds: subject indices
        :param obj_inds: object indices
        :return: location_feature
        """
        boxes_centered = center_size(boxes.data)

        # -- Determine box's center and size (subj's box)
        center_subj = boxes_centered[subj_inds][:, 0:2]
        size_subj = boxes_centered[subj_inds][:, 2:4]

        # -- Determine box's center and size (obj's box)
        center_obj = boxes_centered[obj_inds][:, 0:2]
        size_obj = boxes_centered[obj_inds][:, 2:4]

        # -- Calculate the scale-invariant location features of the subject
        t_coord_subj = (center_subj - center_obj) / size_obj
        t_size_subj = torch.log(size_subj / size_obj)

        # -- Calculate the scale-invariant location features of the object
        t_coord_obj = (center_obj - center_subj) / size_subj
        t_size_obj = torch.log(size_obj / size_subj)

        # -- Put everything together
        location_feature = Variable(torch.cat((t_coord_subj, t_size_subj,
                                               t_coord_obj, t_size_obj), 1))
        return location_feature

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
        :param return_fmap: if the object detector must return the extracted feature maps
        :param depth_imgs: depth images [batch_size, 1, IM_SIZE, IM_SIZE]
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """

        if self.has_visual:
            # -- Feed forward the rgb images to Faster-RCNN
            result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                                   train_anchor_inds, return_fmap=True)
        else:
            # -- Get prior `result` object (instead of calling faster-rcnn's detector)
            result = self.get_prior_results(image_offset, gt_boxes, gt_classes, gt_rels)

        # -- Get RoI and relations
        rois, rel_inds = self.get_rois_and_rels(result, image_offset, gt_boxes, gt_classes, gt_rels)
        boxes = result.rm_box_priors

        # -- Determine subject and object indices
        subj_inds = rel_inds[:, 1]
        obj_inds = rel_inds[:, 2]

        # -- Prepare object predictions vector (PredCLS)
        # replace with ground truth labels
        result.obj_preds = result.rm_obj_labels
        # replace with one-hot distribution of ground truth labels
        result.rm_obj_dists = F.one_hot(result.rm_obj_labels.data, self.num_classes).float()
        obj_cls = result.rm_obj_dists
        result.rm_obj_dists = result.rm_obj_dists * 1000 + (1 - result.rm_obj_dists) * (-1000)

        rel_features = []
        # -- Extract RGB features
        if self.has_visual:
            # Feed the extracted features from first conv layers to the last 'classifier' layers (VGG)
            # Here, only the last 3 layers of VGG are being trained. Everything else (in self.detector)
            # is frozen.
            result.obj_fmap = self.get_roi_features(result.fmap.detach(), rois)

            # -- Create a pairwise relation vector out of visual features
            rel_visual = torch.cat((result.obj_fmap[subj_inds], result.obj_fmap[obj_inds]), 1)
            rel_visual_fc = self.visual_hlayer(rel_visual)
            rel_visual_scale = self.visual_scale(rel_visual_fc)
            rel_features.append(rel_visual_scale)

        # -- Extract Location features
        if self.has_loc:
            # -- Create a pairwise relation vector out of location features
            rel_location = self.get_loc_features(boxes, subj_inds, obj_inds)
            rel_location_fc = self.location_hlayer(rel_location)
            rel_location_scale = self.location_scale(rel_location_fc)
            rel_features.append(rel_location_scale)

        # -- Extract Class features
        if self.has_class:
            if self.use_embed:
                obj_cls = obj_cls @ self.obj_embed.weight
            # -- Create a pairwise relation vector out of class features
            rel_classme = torch.cat((obj_cls[subj_inds], obj_cls[obj_inds]), 1)
            rel_classme_fc = self.classme_hlayer(rel_classme)
            rel_classme_scale = self.classme_scale(rel_classme_fc)
            rel_features.append(rel_classme_scale)

        # -- Extract Depth features
        if self.has_depth:
            # -- Extract features from depth backbone
            depth_features = self.depth_backbone(depth_imgs)
            depth_rois_features = self.get_roi_features_depth(depth_features, rois)

            # -- Create a pairwise relation vector out of location features
            rel_depth = torch.cat((depth_rois_features[subj_inds], depth_rois_features[obj_inds]), 1)
            rel_depth_fc = self.depth_rel_hlayer(rel_depth)
            rel_depth_scale = self.depth_scale(rel_depth_fc)
            rel_features.append(rel_depth_scale)

        # -- Create concatenated feature vector
        rel_fusion = torch.cat(rel_features, 1)

        # -- Extract relation embeddings (penultimate layer)
        rel_embeddings = self.fusion_hlayer(rel_fusion)

        # -- Mix relation embeddings with UoBB features
        if self.has_visual and self.use_vision:
            uobb_features = self.get_union_features(result.fmap.detach(), rois, rel_inds[:, 1:])
            if self.limit_vision:
                # exact value TBD
                uobb_limit = int(self.hidden_dim / 2)
                rel_embeddings = torch.cat((rel_embeddings[:, :uobb_limit] * uobb_features[:, :uobb_limit],
                                            rel_embeddings[:, uobb_limit:]), 1)
            else:
                rel_embeddings = rel_embeddings * uobb_features

        # -- Predict relation distances
        result.rel_dists = self.rel_out(rel_embeddings)

        # -- Frequency bias
        if self.use_bias:
            result.rel_dists = result.rel_dists + self.freq_bias.index_with_labels(torch.stack((
                result.obj_preds[rel_inds[:, 1]],
                result.obj_preds[rel_inds[:, 2]],
            ), 1))

        if self.training:
            return result

        # --- *** END OF ARCHITECTURE *** ---#

        twod_inds = arange(result.obj_preds.data) * self.num_classes + result.obj_preds.data
        result.obj_scores = F.softmax(result.rm_obj_dists, dim=1).view(-1)[twod_inds]

        # Bbox regression
        if self.mode == 'sgdet':
            bboxes = result.boxes_all.view(-1, 4)[twod_inds].view(result.boxes_all.size(0), 4)
        else:
            # Boxes will get fixed by filter_dets function.
            bboxes = result.rm_box_priors

        rel_rep = F.softmax(result.rel_dists, dim=1)
        # Filtering: Subject_Score * Pred_score * Obj_score, sorted and ranked
        return filter_dets(bboxes, result.obj_scores,
                           result.obj_preds, rel_inds[:, 1:], rel_rep)
