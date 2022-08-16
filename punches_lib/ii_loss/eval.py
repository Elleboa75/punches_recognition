from collections import OrderedDict
from typing import Collection, List, Union
import torch
from tqdm import tqdm
from .ii_loss import outlier_score
from .. import utils

def get_mean_embeddings(dataloader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device) -> torch.Tensor:
    '''
    Computes the mean embeddings for a model on a dataloader.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        full_embeddings = []
        labels = []
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            embeddings, _ = model(X)
            full_embeddings.append(embeddings)
            labels.append(y)
    full_embeddings = torch.cat(full_embeddings)
    labels = torch.cat(labels)
    return utils.bucket_mean(full_embeddings, labels, num_classes=labels.max().item()+1)

def eval_outlier_scores(dataloader:torch.utils.data.DataLoader, model:torch.nn.Module, traindata_means:torch.Tensor, device:torch.device) -> torch.Tensor:
    '''
    Evaluates the outlier scores for a model on a dataloader.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        outlier_scores = torch.zeros(len(dataloader.dataset))
        for i, (X, y) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            embeddings, y_hat = model(X)
            outlier_scores_batch = outlier_score(embeddings, traindata_means)
            outlier_scores[i*X.shape[0]:(i+1)*X.shape[0]] = outlier_scores_batch
    return outlier_scores

def eval_on_threshold(outlier_scores:torch.Tensor, threshold:float, comparison_fn=torch.gt) -> float:
    '''
    Thresholds the outlier scores on the given threshold with respect to the specified comparison operator.
    
    Parameters
    ----------
    outlier_scores: a torch.Tensor containing the outlier scores for a data set (e.g., train, validation...)
    threshold: a float specifying the threshold to use
    comparison_fn: a torch.Tensor comparison function (e.g., torch.gt, torch.le). Defaults to torch.gt. It operates the comparison between the outlier scores and the threshold. E.g., if comparison_fn is torch.gt, then it evals on outlier scores being greater than the threshold.
    '''
    return (comparison_fn(outlier_scores, threshold)).sum().item()

def eval_multiple_outlier_scores_series_on_thresholds(outlier_scores:Collection[torch.Tensor], comparison_fns:Collection, thresholds:Union[float,Collection[float]], series_names:Collection[str]=None) -> List[float]:
    '''
    Evaluates multiple outlier scores on multiple thresholds with respect to the specified comparison operators.
    
    Parameters
    ----------
    outlier_scores: a collection of torch.Tensor containing the outlier scores for a data set (e.g., train, validation...)
    comparison_fns: a collection of torch.Tensor comparison functions (e.g., torch.gt, torch.le). It operates the comparison between the outlier scores and the threshold. Must have same size as outlier_scores.
    thresholds: a collection of floats specifying the thresholds to use. Can be a float or a collection of floats.
    series_names: a collection of strings specifying the names of the series. Defaults to None. If specified, must have same size as outlier_scores.
    '''
    assert len(outlier_scores) == len(comparison_fns), f"Expected len(outlier_scores) ({len(outlier_scores)}) to be equal to len(comparison_fns) ({len(comparison_fns)})"
    if series_names is not None:
        assert len(outlier_scores) == len(series_names), f"Expected len(outlier_scores) ({len(outlier_scores)}) to be equal to len(series_names) ({len(series_names)})"
    else:
        series_names = range(len(outlier_scores))
    
    if isinstance(thresholds, float):
        thresholds = [thresholds]

    results = {"threshold": []}
    for name in series_names:
        results[name + "_N"] = []
        results[name + "_N_corr"] = []
        results[name + "_pct"] = []
    for th in thresholds:
        results["threshold"].append(th)
        for scores, fn, name in zip(outlier_scores, comparison_fns, series_names):
            results[name + "_N"].append(len(scores))
            N_corr = eval_on_threshold(scores, th, fn)
            results[name + "_N_corr"].append(N_corr)
            results[name + "_pct"].append(N_corr / len(scores))
    return results

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
            
            _, y_hat = model(X)
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