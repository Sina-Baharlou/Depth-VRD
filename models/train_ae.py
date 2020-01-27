"""
Training script for Convolutional Auto-Encoder
"""
import os
import time
import pandas as pd
import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from config import ModelConfig
from dataloaders.visual_genome import VGDataLoader, VG
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from lib.pytorch_misc import print_para
from lib.shz_models.ae_model_depth import AEModel
from lib.pytorch_misc import set_random_seed, log_depth_details

conf = ModelConfig()

# -- Set random seed
if conf.rnd_seed is not None:
    set_random_seed(conf.rnd_seed)

# -- Get model configuration
conf = ModelConfig()

# -- Create Tensorboard summary writer
writer = SummaryWriter(comment='_run#' + conf.save_dir.split('/')[-1])

# -- Create dataset splits and dataset loader
train, val, _ = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                          use_proposals=False,
                          filter_non_overlap=False,
                          # -- (ADDED) add depth related parameters
                          use_depth=True,
                          three_channels_depth=False)

train_loader, val_loader = VGDataLoader.splits(train, val, mode='det',
                                               batch_size=conf.batch_size,
                                               num_workers=conf.num_workers,
                                               num_gpus=conf.num_gpus,
                                               use_depth=True)
# -- Create Auto-Encoder model
detector = AEModel(num_gpus=conf.num_gpus, depth_model=conf.depth_model)

# -- Print model parameters
print(print_para(detector), flush=True)

# -- Load the specified checkpoint
if conf.ckpt is not None:
    print("Loading the checkpoint: ", conf.ckpt)
    ckpt = torch.load(conf.ckpt)
    start_epoch = ckpt['epoch']

    if not optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = -1
else:
    print("Checkpoint is not provided.")
    start_epoch = -1

detector.cuda()


# -- Define training related functions
def get_optim(lr):
    params = [p for n, p in detector.named_parameters() if p.requires_grad]

    if conf.adam:
        optimizer = optim.Adam(params, lr=lr)
    else:
        optimizer = optim.SGD(params, weight_decay=conf.l2, lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
                                  verbose=True, threshold=0.0001, threshold_mode='abs', cooldown=1)
    return optimizer, scheduler


def train_epoch(epoch_num):
    detector.train()
    tr = []
    start = time.time()

    # -- Early logging to the tensorboard
    if conf.tensorboard_ex:
        log_depth_details(detector, None, writer)
        # -- *** ADD OTHER MODULES HERE AS WELL***

    for b, batch in enumerate(train_loader):

        tr.append(train_batch(batch, verbose=b % (conf.print_interval * 10) == 0))  # b == 0))

        if b % conf.print_interval == 0 and b >= conf.print_interval:
            mn = pd.concat(tr[-conf.print_interval:], axis=1).mean(1)
            time_per_batch = (time.time() - start) / conf.print_interval

            print("\ne{:2d}b{:5d}/{:5d} {:.3f}s/batch, {:.1f}m/epoch".format(
                epoch_num, b, len(train_loader), time_per_batch, len(train_loader) * time_per_batch / 60))
            print(mn)
            print('-----------', flush=True)

            writer.add_scalar('data/ae_loss', mn.ae_loss, (epoch_num * len(train_loader) + b))

            # -- Store additional information about depth maps and depth cnn
            if conf.tensorboard_ex:
                depth_batch = batch[0][9]
                log_depth_details(detector, depth_batch, writer)

            start = time.time()
    return pd.concat(tr, axis=1)


def train_batch(b, verbose=False):
    depth_imgs, dec_fmaps = detector[b]

    losses = {}
    losses['ae_loss'] = F.mse_loss(dec_fmaps, depth_imgs)
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


def val_epoch(epoch_num):
    detector.eval()
    vl_arr = []

    for val_b, batch in enumerate(val_loader):
        vl_arr.append(val_batch(batch))

    mn = pd.concat(vl_arr, axis=1).mean(1)
    writer.add_scalar('data/val_ae_loss', mn.ae_loss, epoch_num)

    print("========================"
          "\nValidation epoch loss:")
    print(mn)
    print("========================\n")

    return mn.total


def val_batch(b):
    depth_imgs, dec_fmaps = detector[b]

    losses = {}
    losses['ae_loss'] = F.mse_loss(depth_imgs, dec_fmaps)
    loss = sum(losses.values())
    losses['total'] = loss

    res = pd.Series({x: y.item() for x, y in losses.items()})
    return res


# -- Create optimizer and scheduler
optimizer, scheduler = get_optim(conf.lr * conf.num_gpus * conf.batch_size)

print("Training starts now!")
for epoch in range(start_epoch + 1, start_epoch + 1 + conf.num_epochs):

    # -- Perform a training epoch
    rez = train_epoch(epoch)

    # -- Show overall losses
    print("overall{:2d}: ({:.3f})\n{}".format(
        epoch, rez.mean(1)['total'], rez.mean(1)),
        flush=True)

    # -- Save the model
    if conf.save_dir is not None:
        torch.save({
            'epoch': epoch,
            'state_dict': detector.state_dict(),
        }, os.path.join(conf.save_dir, '{}-{}.tar'.format('vgae', epoch)))

    # -- Perform a validation epoch
    mAp = val_epoch(epoch)

    # -- Step the scheduler
    scheduler.step(mAp)
