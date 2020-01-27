"""
Relation detection base model
"""

import torch
import torch.nn as nn
import torch.nn.parallel

from lib.fpn.box_utils import bbox_overlaps
from lib.fpn.proposal_assignments.proposal_assignments_gtbox import proposal_assignments_gtbox
from lib.object_detector import Result
from lib.object_detector import gather_res
from lib.pytorch_misc import diagonal_inds
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments

MODES = ('sgdet', 'sgcls', 'predcls')
MODEL_FEATURES = {'v',  # visual features
                  'l',  # location features
                  'c',  # class features
                  'd'}  # depth features


class RelModelBase(nn.Module):
    """
    Base model for performing predicate prediction with depth-maps
    """

    def __init__(self, classes, rel_classes, mode='sgdet', num_gpus=1, require_overlap_det=True,
                 active_features=None, frozen_features=None):
        """
        :param classes: object classes
        :param rel_classes: relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: whether two objects must intersect
        :param active_features: string containing the active features
        :param frozen_features: string containing the frozen features
        """
        super(RelModelBase, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        # -- Current implementation only supports predicate classification mode
        assert mode in MODES
        assert mode == 'predcls'
        self.mode = mode
        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        # -- Determine active features from provided flags
        features_set = self.get_flags(active_features, MODEL_FEATURES, 'vcl')

        self.has_visual = 'v' in features_set
        self.has_loc = 'l' in features_set
        self.has_class = 'c' in features_set
        self.has_depth = 'd' in features_set

        # -- Determine frozen features from provided flags
        frozen_set = self.get_flags(frozen_features, MODEL_FEATURES)
        self.frz_visual = 'v' in frozen_set
        self.frz_loc = 'l' in frozen_set
        self.frz_class = 'c' in frozen_set
        self.frz_depth = 'd' in frozen_set

        # -- Frozen features must be a subset of active features
        assert frozen_set.issubset(features_set)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    @staticmethod
    def freeze_module(module):
        """
        Freeze a module (turn off the gradients)
        :param module: provided nn.Module
        """
        for p_name, param in module.named_parameters():
            param.requires_grad = False

    @staticmethod
    def get_flags(input_str, superset=None, default_str=''):
        """
        Determine a flag set from input string
        a `Result` object
        :param input_str: provided string containing the flags (e.g. "vcl")
        :param superset: determined flags should be a subset of this item
        :param default_str: default flags
        :return: flags_set
        """
        flags_str = default_str if input_str in ['', None] else input_str
        flags_set = set(flags_str.strip().lower())

        # -- Check if the provided flags are valid
        if superset and not flags_set.issubset(superset):
            raise ValueError("Invalid flags: valid set is {}".format(superset))

        return flags_set

    def gt_boxes(self, image_offset, gt_boxes=None, gt_classes=None, gt_rels=None):
        """
        Gets Ground-Truth boxes.
        :param image_offset: offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param gt_rels: [] gt relations
        :return rois, labels, rel_labels
        """
        assert gt_boxes is not None
        im_inds = gt_classes[:, 0] - image_offset
        rois = torch.cat((im_inds.float()[:, None], gt_boxes), 1)
        if gt_rels is not None and self.training:
            rois, labels, rel_labels = proposal_assignments_gtbox(
                rois.data, gt_boxes.data, gt_classes.data, gt_rels.data, image_offset,
                fg_thresh=0.5)
        else:
            labels = gt_classes[:, 1]
            rel_labels = None

        return rois, labels, rel_labels

    def get_prior_results(self, image_offset, gt_boxes, gt_classes, gt_rels):
        """
        Get prior `results` object (predcls only)
        We can call this instead of r-cnn detector to
        get the necessary items to carry on the rel detection
        :param image_offset: offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param gt_rels: [] gt relations
        :return object_detector.Result()
        """
        rois, obj_labels, rel_labels = \
            self.gt_boxes(image_offset, gt_boxes, gt_classes, gt_rels)

        im_inds = rois[:, 0].long().contiguous() + image_offset
        box_priors = rois[:, 1:]

        return Result(
            od_box_priors=box_priors,
            rm_box_priors=box_priors,
            od_obj_labels=obj_labels,
            rm_obj_labels=obj_labels,
            rel_labels=rel_labels,
            im_inds=im_inds)

    def get_rois_and_rels(self, result, image_offset, gt_boxes, gt_classes, gt_rels):
        """
        Get region of interests and their corresponding relationships given
        a `Result` object
        :param result: provided `Result` object
        :param image_offset: offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes: Ground-Truth boxes over the batch.
        :param gt_classes: Ground-Truth classes where each one is (img_id, class)
        :param gt_rels: Ground-Truth relations
        :return: rois, rel_inds
        """
        if result.is_none():
            return ValueError("The provided result is empty")

        # -- Get image indices and image boxes
        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        # -- Assign the relations
        if self.training and result.rel_labels is None:
            assert self.mode == 'sgdet'
            result.rel_labels = rel_assignments(im_inds.data, boxes.data, result.rm_obj_labels.data,
                                                gt_boxes.data, gt_classes.data, gt_rels.data,
                                                image_offset, filter_non_overlap=True,
                                                num_sample_per_gt=1)

        # -- Determine the relation indices and region of interests
        rel_inds = self.get_rel_inds(result.rel_labels, im_inds, boxes)
        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        return rois, rel_inds

    def get_rel_inds(self, rel_labels, im_inds, box_priors):
        """
        Get the relationship candidates
        :param rel_labels: array of relation labels
        :param im_inds:  image indices
        :param box_priors: RoI bounding boxes
        :return rel_inds
        """
        if self.training:
            rel_inds = rel_labels[:, :3].data.clone()
        else:
            rel_cands = im_inds.data[:, None] == im_inds.data[None]
            rel_cands.view(-1)[diagonal_inds(rel_cands)] = 0

            # Require overlap for detection
            if self.require_overlap:
                rel_cands = rel_cands & (bbox_overlaps(box_priors.data,
                                                       box_priors.data) > 0)

                # if there are fewer then 100 things then we might as well add some?
                amt_to_add = 100 - rel_cands.long().sum()

            rel_cands = rel_cands.nonzero()
            if rel_cands.dim() == 0:
                rel_cands = im_inds.data.new(1, 2).fill_(0)

            rel_inds = torch.cat((im_inds.data[rel_cands[:, 0]][:, None], rel_cands), 1)
        return rel_inds

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None,
                proposals=None, train_anchor_inds=None,
                return_fmap=False, depth_imgs=None):
        raise NotImplementedError

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])

        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])

        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
