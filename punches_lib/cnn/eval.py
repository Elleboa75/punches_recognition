import torch
from collections import OrderedDict
from .. import utils

def test_model(model:torch.nn.Module, dataloader:torch.utils.data.DataLoader, loss_fn=None, device=None):
    '''
    Evalates a model on a given dataloader. Will also run a per-class accuracy check.

    Parameters
    ----------
    model: a torch.nn.Module instance.
    dataloader: a torch.utils.data.DataLoader instance.
    loss_fn: a torch.nn.Module instance. If None, will eval only on accuracy.
    device: a torch.device instance or a string indicating the device to use. If None, will use CUDA if available.
    '''
    # create an AverageMeter for the loss if passed
    if loss_fn is not None:
        loss_meter = utils.AverageMeter()
    performance_meter = utils.AverageMeter()
    
    if device is None:
        device = utils.use_cuda_if_possible()

    model = model.to(device)

    num_classes = len(dataloader.dataset.classes)
    num_correct_classes = OrderedDict({cl:0 for cl in range(num_classes)})
    num_items_classes = OrderedDict({cl:0 for cl in range(num_classes)})

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            y_hat = model(X)
            loss = loss_fn(y_hat, y) if loss_fn is not None else None
            acc = utils.accuracy(y_hat, y)
            for cl in range(num_classes):
                num_correct_classes[cl] += (y[y==cl].eq(y_hat.argmax(1)[y==cl])).sum().item()
                num_items_classes[cl] += (y==cl).sum().item()
            
            if loss_fn is not None:
                loss_meter.update(loss.item(), X.shape[0])
            performance_meter.update(acc, X.shape[0])
    # get final performances
    fin_loss = loss_meter.sum if loss_fn is not None else None
    fin_perf = performance_meter.avg
    print(f"TESTING - loss {fin_loss if fin_loss is not None else '--'} - performance {fin_perf:.4f}")
    print("---------------")
    print("PER CLASS PERFORMANCE")
    for (cl, corr), (_, num_items) in zip(num_correct_classes.items(), num_items_classes.items()):
        perf_per_class = corr / num_items
        punch_id = dataloader.dataset.classes[cl]
        print(f"Class: {cl} [ID: {punch_id}] - correct: {corr} - num items: {num_items} - accuracy: {perf_per_class:.4f}")