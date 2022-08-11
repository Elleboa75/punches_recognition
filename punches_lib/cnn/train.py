from typing import Union
import torch

from .. import utils

def train_model(
    model:torch.nn.Module,
    dataloader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    num_epochs:int,
    lr_scheduler:torch.optim.lr_scheduler._LRScheduler=None,
    device:Union[torch.device, str]=None
):
    '''
    Trains a model with the given parameters.

    Parameters
    ----------
    model: a torch.nn.Module instance.
    dataloader: a torch.utils.data.DataLoader instance.
    loss_fn: a torch.nn.Module instance.
    optimizer: a torch.optim.Optimizer instance.
    num_epochs: an integer indicating the number of epochs to train.
    lr_scheduler: a learning rate scheduler - torch.optim.lr_scheduler._LRScheduler instance.
    device: a torch.device instance or a string indicating the device to use. If None, will use CUDA if available.
    '''

    if device is None:
        device = utils.use_cuda_if_possible()
    
    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        loss_meter = utils.AverageMeter()
        performance_meter = utils.AverageMeter()

        # added print for LR
        print(f"Epoch {epoch+1} --- learning rate {optimizer.param_groups[0]['lr']:.5f}")

        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            # 1. reset the gradients previously accumulated by the optimizer
            optimizer.zero_grad() 
            # 2. get the predictions from the current state of the model
            y_hat = model(X)
            # 3. calculate the loss on the current mini-batch
            loss = loss_fn(y_hat, y)
            # 4. execute the backward pass given the current loss
            loss.backward()
            # 5. update the value of the params
            optimizer.step()
            # 6. calculate the accuracy for this mini-batch
            acc = utils.accuracy(y_hat, y)
            # 7. update the loss and accuracy AverageMeter
            loss_meter.update(val=loss.item(), n=X.shape[0])
            performance_meter.update(val=acc, n=X.shape[0])

        print(f"Epoch {epoch+1} completed. Average loss: {loss_meter.avg:.4f}; Performance: {performance_meter.avg:.4f}")

        # update the state of the lr scheduler if provided
        if lr_scheduler is not None:
            lr_scheduler.step()


