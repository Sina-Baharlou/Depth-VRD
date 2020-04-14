"""
Training script for scene graph detection. Integrated with my faster rcnn setup
"""

from dataloaders.visual_genome import VGDataLoader, VG
import numpy as np
from torch import optim
import torch
import pandas as pd
import time
import os
from tensorboardX import SummaryWriter

from config import ModelConfig, BOX_SCALE, IM_SCALE
from torch.nn import functional as F
from lib.pytorch_misc import optimistic_restore, de_chunkize, clip_grad_norm
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator
from lib.pytorch_misc import print_para
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.pytorch_misc import set_random_seed, log_depth_details, \
    add_module_summary, remove_params

# -- Get model configuration
conf = ModelConfig()

# -- Set random seed
if conf.rnd_seed is not None:
    set_random_seed(conf.rnd_seed)

# -- Import the specified model
if conf.model == 'motifnet':
    from lib.rel_model import RelModel
elif conf.model == 'stanford':
    from lib.rel_model_stanford import RelModelStanford as RelModel
# -- Depth-Fusion models --
elif conf.model == 'shz_depth':
    from lib.shz_models.rel_model_depth import RelModel
elif conf.model == 'shz_depth_union':
    from lib.shz_models.rel_model_depth_union import RelModel
elif conf.model == 'shz_fusion':
    from lib.shz_models.rel_model_fusion import RelModel
elif conf.model == 'shz_fusion_beta':
    from lib.shz_models.rel_model_fusion_beta import RelModel
# --
else:
    raise ValueError()

# -- Create Tensorboard summary writer
writer = SummaryWriter(comment='_run#'+ conf.save_dir.split('/')[-1])

# -- Create dataset splits and dataset loader
train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=conf.use_proposals,
                          filter_non_overlap=conf.mode == 'sgdet',
                          # -- Depth dataset parameters
                          use_depth=conf.load_depth,
                          three_channels_depth=conf.pretrained_depth)

train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               # -- Depth dataset parameters
                                               use_depth=conf.load_depth)

# -- Create the specified Relation-Detection model
detector = RelModel(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, order=conf.order,
                    nl_edge=conf.nl_edge, nl_obj=conf.nl_obj, hidden_dim=conf.hidden_dim,
                    use_proposals=conf.use_proposals,
                    pass_in_obj_feats_to_decoder=conf.pass_in_obj_feats_to_decoder,
                    pass_in_obj_feats_to_edge=conf.pass_in_obj_feats_to_edge,
                    pooling_dim=conf.pooling_dim,
                    rec_dropout=conf.rec_dropout,
                    use_bias=conf.use_bias,
                    use_tanh=conf.use_tanh,
                    use_vision=conf.use_vision,
                    # -- The proposed model parameters
                    depth_model=conf.depth_model,
                    pretrained_depth=conf.pretrained_depth,
                    active_features=conf.active_features,
                    frozen_features=conf.frozen_features,
                    use_embed=conf.use_embed)

# -- Freeze the detector (Faster-RCNN)
for n, param in detector.detector.named_parameters():
    param.requires_grad = False

# -- Print model parameters
print(print_para(detector), flush=True)

# -- Define training related functions
def is_conv_param_depth(name):
    """
    Checks if the provided parameter is in the convolutional parameters list
    :param name: parameter name
    :return: `True` if the parameter is in the list
    """
    if conf.depth_model in ["resnet18", "resnet50"]:
        depth_conv_params = ['depth_backbone',
                             'depth_rel_head',
                             'depth_rel_head_union'
                             'depth_union_boxes']
    else:
        depth_conv_params = ['depth_backbone',
                             'depth_union_boxes']
    for param in depth_conv_params:
        if name.startswith(param):
            return True
    return False


def get_optim(lr):
    # Lower the learning rate on the VGG fully connected layers by 1/10th. It's a hack, but it helps
    # stabilize the models.

    if conf.model.startswith('shz_depth'):
        fc_params = [p for n,p in detector.named_parameters()
                     if not is_conv_param_depth(n) and p.requires_grad]
        non_fc_params = [p for n, p in detector.named_parameters()
                         if is_conv_param_depth(n) and p.requires_grad]
    else:
        fc_params = [p for n, p in detector.named_parameters() if n.startswith('roi_fmap') and p.requires_grad]
        non_fc_params = [p for n, p in detector.named_parameters() if not n.startswith('roi_fmap') and p.requires_grad]

    # -- Show the number of FC/non-FC parameters
    print("#FC params:{}, #non-FC params:{}".format(len(fc_params),
                                                        len(non_fc_params)))

    params = [{'params': fc_params, 'lr': lr / 10.0}, {'params': non_fc_params}]

    if conf.adam:
        optimizer = optim.Adam(params, lr=lr)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=6, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler


# -- The parameters to be removed from the provided checkpoint
rm_params = ['rel_out.bias',
             'rel_out.weight',
             'fusion_hlayer.bias',
             'fusion_hlayer.weight']

# -- Load the checkpoint if it's provided
start_epoch = -1
if conf.ckpt is not None:
    ckpt = torch.load(conf.ckpt)

    # -- If the provided checkpoint is `vg-faster-rcnn`
    if conf.ckpt.endswith("vg-faster-rcnn.tar"):
        print("Loading Faster-RCNN checkpoint...")
        start_epoch = -1
        optimistic_restore(detector.detector, ckpt['state_dict'])

        # -- Load different heads' weights from faster r-cnn
        if hasattr(detector, "roi_fmap") and detector.roi_fmap is not None:
            detector.roi_fmap[1][0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
            detector.roi_fmap[1][3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
            detector.roi_fmap[1][0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
            detector.roi_fmap[1][3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])
        if hasattr(detector, "roi_fmap_obj") and detector.roi_fmap_obj is not None:
            detector.roi_fmap_obj[0].weight.data.copy_(ckpt['state_dict']['roi_fmap.0.weight'])
            detector.roi_fmap_obj[3].weight.data.copy_(ckpt['state_dict']['roi_fmap.3.weight'])
            detector.roi_fmap_obj[0].bias.data.copy_(ckpt['state_dict']['roi_fmap.0.bias'])
            detector.roi_fmap_obj[3].bias.data.copy_(ckpt['state_dict']['roi_fmap.3.bias'])

    # -- Otherwise
    else:
        print("Loading everything...")
        start_epoch = ckpt['epoch']

        # -- Attach the extra checkpoint if it's provided
        if conf.extra_ckpt is not None:
            print("Attaching the extra checkpoint to the main one!")
            extra_ckpt_state_dict = torch.load(conf.extra_ckpt)
            ckpt['state_dict'].update(extra_ckpt_state_dict['state_dict'])

        # -- Remove unwanted weights from state_dict (last two layers)
        if not conf.keep_weights:
            remove_params(ckpt['state_dict'], rm_params)

        # -- Load the checkpoint
        if not optimistic_restore(detector, ckpt['state_dict']):
            start_epoch = -1

detector.cuda()


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()

    # -- Early logging to the tensorboard
    if conf.tensorboard_ex:
        log_depth_details(detector, None, writer)
        # -- *** ADD OTHER MODULES HERE ***
        if hasattr(detector, "fusion_hlayer") and detector.fusion_hlayer is not None:
            add_module_summary(detector.fusion_hlayer, writer, "fusion_hlayer")
        if hasattr(detector, "rel_out") and detector.rel_out is not None:
            add_module_summary(detector.rel_out, writer, "rel_out")

    for b, batch in enumerate(train_loader):
        tr.append(train_batch(batch, verbose=b % (conf.print_interval*10) == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval
            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)

            writer.add_scalar('data/class_loss', mn.class_loss, (epoch_num * len(train_loader) + b))
            writer.add_scalar('data/rel_loss', mn.rel_loss, (epoch_num * len(train_loader) + b))
            writer.add_scalar('data/total_loss', mn.total, (epoch_num * len(train_loader) + b))

            # -- Store additional information about depth maps and depth cnn
            if conf.tensorboard_ex:
                depth_batch = batch[0][9]
                log_depth_details(detector, depth_batch, writer)
                # -- *** ADD OTHER MODULES HERE ***
                if hasattr(detector, "fusion_hlayer") and detector.fusion_hlayer is not None:
                    add_module_summary(detector.fusion_hlayer, writer, "fusion_hlayer")
                if hasattr(detector, "rel_out") and detector.rel_out is not None:
                    add_module_summary(detector.rel_out, writer, "rel_out")

            print('-----------', flush=True)
            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b, verbose=False):
    """
    :param b: contains:
          :param imgs: the image, [batch_size, 3, IM_SIZE, IM_SIZE]
          :param all_anchors: [num_anchors, 4] the boxes of all anchors that we'll be using
          :param all_anchor_inds: [num_anchors, 2] array of the indices into the concatenated
                                  RPN feature vector that give us all_anchors,
                                  each one (img_ind, fpn_idx)
          :param im_sizes: a [batch_size, 4] numpy array of (h, w, scale, num_good_anchors) for each image.

          :param num_anchors_per_img: int, number of anchors in total over the feature pyramid per img

          Training parameters:
          :param train_anchor_inds: a [num_train, 5] array of indices for the anchors that will
                                    be used to compute the training loss (img_ind, fpn_idx)
          :param gt_boxes: [num_gt, 4] GT boxes over the batch.
          :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
    :return:
    """
    result = detector[b]

    losses = {}
    losses['class_loss'] = F.cross_entropy(result.rm_obj_dists, result.rm_obj_labels)
    losses['rel_loss'] = F.cross_entropy(result.rel_dists, result.rel_labels[:, -1])
    loss = sum(losses.values())

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm(
        [(n, p) for n, p in detector.named_parameters() if p.grad is not None],
        max_norm=conf.clip, verbose=verbose, clip=True)
    losses['total'] = loss
    optimizer.step()
    res = pd.Series({x: y.item() for x, y in losses.items()})
    return res


def val_epoch():
    detector.eval()
    evaluator = BasicSceneGraphEvaluator.all_modes()
    for val_b, batch in enumerate(val_loader):
        val_batch(conf.num_gpus * val_b, batch, evaluator)
    evaluator[conf.mode].print_stats(epoch, writer)
    return np.mean(evaluator[conf.mode].result_dict[conf.mode + '_recall'][100])


def val_batch(batch_num, b, evaluator):
    det_res = detector[b]
    if conf.num_gpus == 1:
        det_res = [det_res]

    for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
        gt_entry = {
            'gt_classes': val.gt_classes[batch_num + i].copy(),
            'gt_relations': val.relationships[batch_num + i].copy(),
            'gt_boxes': val.gt_boxes[batch_num + i].copy(),
        }
        assert np.all(objs_i[rels_i[:, 0]] > 0) and np.all(objs_i[rels_i[:, 1]] > 0)

        pred_entry = {
            'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
            'pred_classes': objs_i,
            'pred_rel_inds': rels_i,
            'obj_scores': obj_scores_i,
            'rel_scores': pred_scores_i,  # hack for now.
        }

        evaluator[conf.mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )


print("Training starts now!")

# -- Create optimizer and scheduler
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):

    # -- Perform a training epoch
    rez = train_epoch(epoch)

    # -- Show overall losses
    print("overall{:2d}: ({:.3f})\n{}".format(epoch, rez.mean(1)['total'], rez.mean(1)), flush=True)

    # -- Save the model
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgrel', epoch)))

    # -- Perform a validation epoch
    mAp = val_epoch()

    # -- Step the scheduler
    scheduler.step(mAp)

    # -- (DISABLED) Stopping early stopping!
    # if any([pg['lr'] <= (conf.lr * conf.num_gpus * conf.batch_size)/99.0 for pg in optimizer.param_groups]):
    #     print("exiting training early", flush=True)
    #     break
