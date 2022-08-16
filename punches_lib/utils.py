from typing import Collection
import torch
import torchvision
from copy import deepcopy
import numpy as np

def use_cuda_if_possible():
    '''
    Returns the GPU torch device if CUDA is available.
    Defaults to the GPU with ID #0.
    '''
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

class AverageMeter(object):
    '''
    a generic class to keep track of performance metrics during training or testing of models
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(nn_output:torch.Tensor, ground_truth:torch.Tensor, k: int=1) -> float:
    '''
    Get accuracy@k for the given model output and ground truth

    Parameters
    ----------
    nn_output: a tensor of shape (num_datapoints x num_classes) which may 
       or may not be the output of a softmax or logsoftmax layer
    ground_truth: a tensor of longs or ints of shape (num_datapoints)
    k: the 'k' in 'accuracy@k'
    '''
    # get classes of assignment for the top-k nn_outputs row-wise
    nn_out_classes = nn_output.topk(k).indices
    # make ground_truth a column vector
    ground_truth_vec = ground_truth.unsqueeze(-1)
    # and repeat the column k times (= reproduce nn_out_classes shape)
    ground_truth_vec = ground_truth_vec.expand_as(nn_out_classes)
    # produce tensor of booleans - at which position of the nn output is the correct class located?
    correct_items = (nn_out_classes == ground_truth_vec)
    # now getting the accuracy is easy, we just operate the sum of the tensor and divide it by the number of examples
    acc = correct_items.sum().item() / nn_output.shape[0]
    return acc

def bucket_mean(embeddings:torch.Tensor, labels:torch.Tensor, num_classes:int) -> torch.Tensor:
    '''
    A helper function to calculate the mean of the embeddings separately for each category.

    Parameters
    ----------
    embeddings: a tensor of shape (num_datapoints x embedding_dim)
    labels: a tensor of longs or ints of shape (num_datapoints)
    num_classes: the number of categories

    Returns
    -------
    a tensor of shape (num_classes x embedding_dim).
    '''
    device = embeddings.device
    tot = torch.zeros(num_classes, embeddings.shape[1], device=device).index_add(0, labels, embeddings)
    count = torch.zeros(num_classes, embeddings.shape[1], device=device).index_add(0, labels, torch.ones_like(embeddings))
    return tot/count

def subset_imagefolder(imagefolder:torchvision.datasets.ImageFolder, indices:Collection[bool]):
    indices = indices.numpy()

    d = deepcopy(imagefolder)
    d.imgs = (np.array(d.imgs)[indices]).tolist()
    d.samples = (np.array(d.samples)[indices]).tolist()
    d.targets = (np.array(d.targets[indices])).tolist()
    return d