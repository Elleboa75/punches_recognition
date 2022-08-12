import torch
from typing import Union
from tqdm import tqdm
from .. import utils
from .ii_loss import IILoss

def train_model(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    ii_loss_fn:IILoss,
    ce_loss_fn:torch.nn.Module,
    num_epochs:int,
    optimizer:torch.optim.Optimizer,
    lr_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
    device:Union[torch.device, str]=None
):
    '''
    Trains a model with the given parameters using a dual loss composed of IILoss and CELoss.

    Parameters
    ----------
    model: a torch.nn.Module instance.
    dataloader: a torch.utils.data.DataLoader instance.
    ii_loss_fn: an instance of IILoss.
    ce_loss_fn: a torch.nn.Module instance (possibly an instance of CELoss).
    optimizer: a torch.optim.Optimizer instance.
    num_epochs: an integer indicating the number of epochs to train.
    lr_scheduler: a learning rate scheduler - torch.optim.lr_scheduler._LRScheduler instance.
    device: a torch.device instance or a string indicating the device to use. If None, will use CUDA if available.
    '''

    if device is None:
        device = utils.use_cuda_if_possible()
    
    num_classes = len(dataloader.dataset.classes)

    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        ii_loss_meter = utils.AverageMeter()
        ce_loss_meter = utils.AverageMeter()
        performance_meter = utils.AverageMeter()

        # added print for LR
        print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")

        for X, y in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. reset the gradients previously accumulated by the optimizer
            optimizer.zero_grad() 
            # 2. get the predictions from the current state of the model
            embeddings, y_hat = model(X)
            # 3. calculate ii_loss and backpropagate
            ii_loss = ii_loss_fn(embeddings, y, num_classes)
            ii_loss.backward(retain_graph=True)
            # 4. calculate ce_loss and backpropagate
            ce_loss = ce_loss_fn(y_hat, y)
            ce_loss.backward()
            # 5. update the value of the params
            optimizer.step()
            # 6. calculate the accuracy for this mini-batch
            acc = utils.accuracy(y_hat, y)
            # 7. update the losses and accuracy AverageMeters
            ii_loss_meter.update(val=ii_loss.item(), n=X.shape[0])
            ce_loss_meter.update(val=ce_loss.item(), n=X.shape[0])
            performance_meter.update(val=acc, n=X.shape[0])

        print(f"Epoch {epoch+1} completed. Average II loss: {ii_loss_meter.avg}; CE loss: {ce_loss_meter.avg:.4f}; Performance: {performance_meter.avg:.4f}")

        # update the state of the lr scheduler if provided
        if lr_scheduler is not None:
            lr_scheduler.step()