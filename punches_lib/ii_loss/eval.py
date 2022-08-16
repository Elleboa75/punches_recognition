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
    return (comparison_fn(outlier_scores, threshold)).float().mean().item()

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

    results = {}
    for th in thresholds:
        results[th] = {}
        for scores, fn, name in zip(outlier_scores, comparison_fns, series_names):
            results[th][name] = eval_on_threshold(scores, th, fn)
    return results