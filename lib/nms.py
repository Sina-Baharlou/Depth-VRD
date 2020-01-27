# Le code for doing NMS
import torch
import numpy as np
# from .._ext import nms
from torchvision.ops import nms


def apply_nms(scores, boxes,  pre_nms_topn=12000, post_nms_topn=2000, boxes_per_im=None,
              nms_thresh=0.7):
    """
    Note - this function is non-differentiable so everything is assumed to be a tensor, not
    a variable.
        """
    just_inds = boxes_per_im is None
    if boxes_per_im is None:
        boxes_per_im = [boxes.size(0)]


    s = 0
    keep = []
    im_per = []
    for bpi in boxes_per_im:
        e = s + int(bpi)
        keep_im = _nms_single_im(scores[s:e], boxes[s:e], pre_nms_topn, post_nms_topn, nms_thresh)
        keep.append(keep_im + s)
        im_per.append(keep_im.size(0))

        s = e

    inds = torch.cat(keep, 0)
    if just_inds:
        return inds
    return inds, im_per


def _nms_single_im(scores, boxes,  pre_nms_topn=12000, post_nms_topn=2000, nms_thresh=0.7):

    '''Replaced the usage of compiled "nms" function with torchvision nms and moved this
    whole file into lib/nms so that the "excluded folders" rule doesn't apply to it.
    '''
    vs, idx = torch.sort(scores, dim=0, descending=True)
    if idx.size(0) > pre_nms_topn:
        idx = idx[:pre_nms_topn]
    boxes_sorted = boxes[idx].contiguous()
    scores = scores[idx].contiguous()
    # keep = torch.cuda.IntTensor(boxes_sorted.size(0))
    # num_out = nms.nms_apply(keep, boxes_sorted, nms_thresh)

    # num_out = nms(boxes_sorted, scores, nms_thresh)
    # keep = scores[:num_out].long()
    keep = nms(boxes_sorted, scores, nms_thresh)
    num_out = min(keep.shape[0], post_nms_topn)
    keep = keep[:num_out]
    # keep = keep[:num_out].long()
    # keep = idx[keep.cuda(scores.get_device())]
    keep = idx[keep]
    #keep = keep.cpu()
    return keep
