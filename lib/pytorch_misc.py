"""
Miscellaneous functions that might be useful for pytorch
"""

import h5py
import numpy as np
import random
import torch
from torch.autograd import Variable
import os
import dill as pkl
from itertools import tee
from torch import nn
from torch.nn import functional as F

def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()

    # -- (ADDED) added a visual separator for better readability
    print("\n==================================\n"
          "Loading checkpoint parameters...\n")

    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
            print("Successfully loaded {} with size {}".format(name, param.size()))
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("\n*** We couldn't find {}".format(','.join(missing)))
        mismatch = True

    # -- (ADDED) added a visual separator for better readability
    print("==================================\n")
    return not mismatch


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_ranking(predictions, labels, num_guesses=5):
    """
    Given a matrix of predictions and labels for the correct ones, get the number of guesses
    required to get the prediction right per example.
    :param predictions: [batch_size, range_size] predictions
    :param labels: [batch_size] array of labels
    :param num_guesses: Number of guesses to return
    :return:
    """
    assert labels.size(0) == predictions.size(0)
    assert labels.dim() == 1
    assert predictions.dim() == 2

    values, full_guesses = predictions.topk(predictions.size(1), dim=1)
    _, ranking = full_guesses.topk(full_guesses.size(1), dim=1, largest=False)
    gt_ranks = torch.gather(ranking.data, 1, labels.data[:, None]).squeeze()

    guesses = full_guesses[:, :num_guesses]
    return gt_ranks, guesses


def cache(f):
    """
    Caches a computation
    """

    def cache_wrapper(fn, *args, **kwargs):
        if os.path.exists(fn):
            with open(fn, 'rb') as file:
                data = pkl.load(file)
        else:
            print("file {} not found, so rebuilding".format(fn))
            data = f(*args, **kwargs)
            with open(fn, 'wb') as file:
                pkl.dump(data, file)
        return data

    return cache_wrapper


class Flattener(nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def to_variable(f):
    """
    Decorator that pushes all the outputs to a variable
    :param f: 
    :return: 
    """

    def variable_wrapper(*args, **kwargs):
        rez = f(*args, **kwargs)
        if isinstance(rez, tuple):
            return tuple([Variable(x) for x in rez])
        return Variable(rez)

    return variable_wrapper


def arange(base_tensor, n=None):
    new_size = base_tensor.size(0) if n is None else n
    new_vec = base_tensor.new(new_size).long()
    torch.arange(0, new_size, out=new_vec)
    return new_vec


def to_onehot(vec, num_classes, fill=1000):
    """
    Creates a [size, num_classes] torch FloatTensor where
    one_hot[i, vec[i]] = fill
    
    :param vec: 1d torch tensor
    :param num_classes: int
    :param fill: value that we want + and - things to be.
    :return: 
    """
    onehot_result = vec.new(vec.size(0), num_classes).float().fill_(-fill)
    arange_inds = vec.new(vec.size(0)).long()
    torch.arange(0, vec.size(0), out=arange_inds)

    onehot_result.view(-1)[vec + num_classes * arange_inds] = fill
    return onehot_result


def save_net(fname, net):
    h5f = h5py.File(fname, mode='w')
    for k, v in list(net.state_dict().items()):
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    h5f = h5py.File(fname, mode='r')
    for k, v in list(net.state_dict().items()):
        param = torch.from_numpy(np.asarray(h5f[k]))

        if v.size() != param.size():
            print("On k={} desired size is {} but supplied {}".format(k, v.size(), param.size()))
        else:
            v.copy_(param)


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


def batch_map(f, a, batch_size):
    """
    Maps f over the array a in chunks of batch_size.
    :param f: function to be applied. Must take in a block of
            (batch_size, dim_a) and map it to (batch_size, something).
    :param a: Array to be applied over of shape (num_rows, dim_a).
    :param batch_size: size of each array
    :return: Array of size (num_rows, something).
    """
    rez = []
    for s, e in batch_index_iterator(a.size(0), batch_size, skip_end=False):
        print("Calling on {}".format(a[s:e].size()))
        rez.append(f(a[s:e]))

    return torch.cat(rez)


def const_row(fill, l, volatile=False):
    input_tok = Variable(torch.LongTensor([fill] * l), volatile=volatile)
    if torch.cuda.is_available():
        input_tok = input_tok.cuda()
    return input_tok


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    # -- (ADDED) add total trainable parameters --
    total_train_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        # -- (ADDED) Add to trainable parameters
        total_train_params += np.prod(p.size()) if p.requires_grad else 0
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.3f}M total parameters \n {:.3f}M total trainable parameters \n ----- \n \n{}\n'.format(
        total_params / 1000000.0,
        total_train_params / 1000000.0,
        '\n'.join(strings))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def nonintersecting_2d_inds(x):
    """
    Returns np.array([(a,b) for a in range(x) for b in range(x) if a != b]) efficiently
    :param x: Size
    :return: a x*(x-1) array that is [(0,1), (0,2)... (0, x-1), (1,0), (1,2), ..., (x-1, x-2)]
    """
    rs = 1 - np.diag(np.ones(x, dtype=np.int32))
    relations = np.column_stack(np.where(rs))
    return relations


def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def gather_nd(x, index):
    """

    :param x: n dimensional tensor [x0, x1, x2, ... x{n-1}, dim]
    :param index: [num, n-1] where each row contains the indices we'll use
    :return: [num, dim]
    """
    nd = x.dim() - 1
    assert nd > 0
    assert index.dim() == 2
    assert index.size(1) == nd
    dim = x.size(-1)

    sel_inds = index[:, nd - 1].clone()
    mult_factor = x.size(nd - 1)
    for col in range(nd - 2, -1, -1):  # [n-2, n-3, ..., 1, 0]
        sel_inds += index[:, col] * mult_factor
        mult_factor *= x.size(col)

    grouped = x.view(-1, dim)[sel_inds]
    return grouped


def enumerate_by_image(im_inds):
    im_inds_np = im_inds.cpu().numpy()
    initial_ind = int(im_inds_np[0])
    s = 0
    for i, val in enumerate(im_inds_np):
        if val != initial_ind:
            yield initial_ind, s, i
            initial_ind = int(val)
            s = i
    yield initial_ind, s, len(im_inds_np)
    # num_im = im_inds[-1] + 1
    # # print("Num im is {}".format(num_im))
    # for i in range(num_im):
    #     # print("On i={}".format(i))
    #     inds_i = (im_inds == i).nonzero()
    #     if inds_i.dim() == 0:
    #         continue
    #     inds_i = inds_i.squeeze(1)
    #     s = inds_i[0]
    #     e = inds_i[-1] + 1
    #     # print("On i={} we have s={} e={}".format(i, s, e))
    #     yield i, s, e


def diagonal_inds(tensor):
    """
    Returns the indices required to go along first 2 dims of tensor in diag fashion
    :param tensor: thing
    :return: 
    """
    assert tensor.dim() >= 2
    assert tensor.size(0) == tensor.size(1)
    size = tensor.size(0)
    arange_inds = tensor.new(size).long()
    torch.arange(0, tensor.size(0), out=arange_inds)
    return (size + 1) * arange_inds


def enumerate_imsize(im_sizes):
    s = 0
    for i, (h, w, scale, num_anchors) in enumerate(im_sizes):
        na = int(num_anchors)
        e = s + na
        yield i, s, e, h, w, scale, na

        s = e


def argsort_desc(scores):
    """
    Returns the indices that sort scores descending in a smart way
    :param scores: Numpy array of arbitrary size
    :return: an array of size [numel(scores), dim(scores)] where each row is the index you'd
             need to get the score.
    """
    return np.column_stack(np.unravel_index(np.argsort(-scores.ravel()), scores.shape))


def unravel_index(index, dims):
    unraveled = []
    index_cp = index.clone()
    for d in dims[::-1]:
        unraveled.append(index_cp % d)
        index_cp /= d
    return torch.cat([x[:, None] for x in unraveled[::-1]], 1)


def de_chunkize(tensor, chunks):
    s = 0
    for c in chunks:
        yield tensor[s:(s + c)]
        s = s + c


def random_choose(tensor, num):
    "randomly choose indices"
    num_choose = min(tensor.size(0), num)
    if num_choose == tensor.size(0):
        return tensor

    # Gotta do this in numpy because of https://github.com/pytorch/pytorch/issues/1868
    rand_idx = np.random.choice(tensor.size(0), size=num, replace=False)
    rand_idx = torch.LongTensor(rand_idx).cuda(tensor.get_device())
    chosen = tensor[rand_idx].contiguous()

    # rand_values = tensor.new(tensor.size(0)).float().normal_()
    # _, idx = torch.sort(rand_values)
    #
    # chosen = tensor[idx[:num]].contiguous()
    return chosen


def transpose_packed_sequence_inds(lengths):
    """
    Goes from a TxB packed sequence to a BxT or vice versa. Assumes that nothing is a variable
    :param ps: PackedSequence
    :return:
    """

    new_inds = []
    new_lens = []
    cum_add = np.cumsum([0] + lengths)
    max_len = lengths[0]
    length_pointer = len(lengths) - 1
    for i in range(max_len):
        while length_pointer > 0 and lengths[length_pointer] <= i:
            length_pointer -= 1
        new_inds.append(cum_add[:(length_pointer + 1)].copy())
        cum_add[:(length_pointer + 1)] += 1
        new_lens.append(length_pointer + 1)
    new_inds = np.concatenate(new_inds, 0)
    return new_inds, new_lens


def right_shift_packed_sequence_inds(lengths):
    """
    :param lengths: e.g. [2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    :return: perm indices for the old stuff (TxB) to shift it right 1 slot so as to accomodate
             BOS toks
             
             visual example: of lengths = [4,3,1,1]
    before:
    
        a (0)  b (4)  c (7) d (8)
        a (1)  b (5)
        a (2)  b (6)
        a (3)
        
    after:
    
        bos a (0)  b (4)  c (7)
        bos a (1)
        bos a (2)
        bos              
    """
    cur_ind = 0
    inds = []
    for (l1, l2) in zip(lengths[:-1], lengths[1:]):
        for i in range(l2):
            inds.append(cur_ind + i)
        cur_ind += l1
    return inds


def clip_grad_norm(named_parameters, max_norm, clip=False, verbose=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    if verbose:
        print('\n---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
        print('-------------------------------\n', flush=True)

    return total_norm


def update_lr(optimizer, lr=1e-4):
    print("------ Learning rate -> {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


####################################
# -- Our helper functions --
####################################
def normalize_batch(input_batch):
    """
    Normalizes each images in the batch between 0 and 1
    :param input_batch: Tensor[N,C,H,W]
    :return: Tensor[N,C,H,W]
    """
    N, C, H, W = input_batch.size()
    # -- Reshape the batch to be able to calculate the min and max
    # per image in the batch
    reshaped_batch = input_batch.view(N, -1)
    # -- Calculate minimum and maximum per image
    min_per_img = reshaped_batch.min(dim=-1)[0][:, None, None, None]
    max_per_img = reshaped_batch.max(dim=-1)[0][:, None, None, None]
    # -- Normalize the images between 0 and 1
    return (input_batch - min_per_img) / (max_per_img - min_per_img)


def add_module_summary(module, writer, namespace):
    """
    Adds a histogram summary of the provided weights to the tensorboardX
    :param module: The provided module
    :param writer: tensorboardX summary writer
    :param namespace: the provided namespace which is used to divide the histogram sections
    :return: None
    """
    for module_name, module in module.named_modules():
        if isinstance(module, nn.Conv2d):
            writer.add_histogram(f"{namespace}/{module_name}/conv_weights", module.weight)
            if module.bias is not None:
                writer.add_histogram(f"{namespace}/{module_name}/conv_bias", module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            writer.add_histogram(f"{namespace}/{module_name}/bn_weights", module.weight)
            writer.add_histogram(f"{namespace}/{module_name}/bn_bias", module.bias)
        elif isinstance(module, nn.Linear):
            writer.add_histogram(f"{namespace}/{module_name}/dense_weights", module.weight)
            writer.add_histogram(f"{namespace}/{module_name}/dense_bias", module.bias)


def log_depth_details(detector, depth_batch, writer):
    """
    Log extra visual details about the depth data
    :param detector: The provided model (nn.Module)
    :param depth_batch: Provided depth batch (Tensor)
    :param writer: tensorboardX summary writer
    :return: None
    """
    # -- Add a sample depth batch to the TensorBoard
    if depth_batch is not None:
        scaled_depth = F.upsample(depth_batch, size=(64, 64), mode='bilinear')
        writer.add_images("sample_depth_batch", normalize_batch(scaled_depth))

    # -- Add the corresponding reconstructed depth batch to the TensorBoard (AE mode only)
    if hasattr(detector, "depth_rec") and detector.depth_rec is not None:
        scaled_depth_rec = F.upsample(detector.depth_rec, size=(64, 64), mode='bilinear')
        writer.add_images("sample_depth_batch_rec", normalize_batch(scaled_depth_rec))

    # -- Add details about the depth backbone
    if hasattr(detector, "depth_backbone") and detector.depth_backbone is not None:
        # -- Add the first convolutional layer of depth_backbone to the TensorBoard
        if detector.depth_backbone.depth_model in ["resnet18", "resnet50"]:
            conv_filters = detector.depth_backbone.features_depth.conv1.weight
        else:
            conv_filters = detector.depth_backbone.features_depth[0].weight
        writer.add_images("layer0/conv_weights", normalize_batch(conv_filters))

        # -- Add histogram of weights to the TensorBoard
        add_module_summary(detector.depth_backbone.features_depth, writer, "depth_backbone")


def set_random_seed(rnd_seed):
    """
    Fix the random seed among different libraries
    :param rnd_seed: random seed (int)
    :return: None
    """
    # -- Print random seed
    print(f"Set random seed to: {rnd_seed}")

    # -- Set different libraries random seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    # -- Set CUDA's random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rnd_seed)
        torch.cuda.manual_seed_all(rnd_seed)

    # -- Disable CUDNN
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def remove_params(state_dict, parameters):
    """
    Remove the provided paramteres' weights from the model
    :param state_dict: Checkpoints' state dictionary
    :param parameters: Parameters to be removed
    :return:
    """
    if len(parameters)==0:
        return

    # -- added a visual separator for better readability
    print("\n==================================\n"
          "Removing unnecessary parameters...\n")

    for param in parameters:
        if param in state_dict:
            state_dict.pop(param)
            print("Successfully removed parameter:", param)
        else:
            print("Couldn't remove parameter:", param)

    # -- added a visual separator for better readability
    print("==================================\n")

def xavier_init(module):
    """
    Takes a nn.Linear module and initializes it's weights
    with Xavier normal function (torch =<0.4)
    :param module: The provided nn.Linear module
    :return: Initialized nn.Linear module
    """
    assert isinstance(module, nn.Linear)
    module.weight = torch.nn.init.xavier_normal_(module.weight, gain=1.0)
    return module


class ScaleLayer(nn.Module):
    """
    Scaling layer
    Simply multiply the input by a float tensor
    """

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
