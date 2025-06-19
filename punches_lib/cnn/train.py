from typing import Union
import torch
from tqdm import tqdm
from collections import Counter

from .. import utils


def compute_class_weights(dataset):
    """
    Computes class weights based on the dataset label distribution.
    """
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    num_classes = len(class_counts)
    total_samples = sum(class_counts.values())

    weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
    weight_tensor = torch.tensor([weights[i] for i in range(num_classes)], dtype=torch.float)

    return weight_tensor


def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        loss_fn: Union[torch.nn.Module, None],
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: Union[torch.device, str] = None,
        use_class_weights: bool = False
):
    """
    Trains a model with the given parameters.

    Parameters
    ----------
    model: a torch.nn.Module instance.
    train_loader: a torch.utils.data.DataLoader instance for training data.
    loss_fn: a torch.nn.Module instance for loss calculation, or None if using class weights.
    optimizer: a torch.optim.Optimizer instance for model optimization.
    num_epochs: an integer indicating the number of epochs to train.
    lr_scheduler: a learning rate scheduler - torch.optim.lr_scheduler._LRScheduler instance.
    device: a torch.device instance or a string indicating the device to use. If None, will use CUDA if available.
    use_class_weights: a boolean indicating whether to use class weights in the loss function.
    """
    if device is None:
        device = utils.use_cuda_if_possible()

    model = model.to(device)

    # Compute class weights if needed
    if use_class_weights and loss_fn is None:
        class_weights = compute_class_weights(train_loader.dataset).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    elif loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss_meter = utils.AverageMeter()
        train_perf_meter = utils.AverageMeter()

        print(f"Epoch {epoch + 1} --- Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        for X, y in tqdm(train_loader, desc="Training"):
            X, y = X.to(device), y.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            y_hat = model(X)

            # Compute loss
            loss = loss_fn(y_hat, y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update performance metrics
            acc = utils.accuracy(y_hat, y)
            train_loss_meter.update(val=loss.item(), n=X.shape[0])
            train_perf_meter.update(val=acc, n=X.shape[0])

        print(
            f"Training - Epoch {epoch + 1}: Average loss: {train_loss_meter.avg:.4f}, "
            f"Performance: {train_perf_meter.avg:.4f}"
        )

        # Learning rate adjustment
        if lr_scheduler is not None:
            lr_scheduler.step()
