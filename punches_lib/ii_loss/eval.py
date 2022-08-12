import torch
from tqdm import tqdm
from .ii_loss import outlier_score

def get_mean_embeddings(dataloader:torch.utils.data.DataLoader, model:torch.nn.Module, device:torch.device) -> torch.Tensor:
    '''
    Computes the mean embeddings for a model on a dataloader.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        full_embeddings = []
        for i, (X, y) in tqdm(dataloader):
            X = X.to(device)
            y = y.to(device)
            embeddings, _ = model(X)
            full_embeddings.append(embeddings)
    full_embeddings = torch.cat(full_embeddings)
    return full_embeddings.mean(0)

def eval_outlier_scores(dataloader:torch.utils.data.DataLoader, model:torch.nn.Module, traindata_means:torch.Tensor, device:torch.device) -> torch.Tensor:
    '''
    Evaluates the outlier scores for a model on a dataloader.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        outlier_scores = torch.zeros(len(dataloader.dataset))
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)
            embeddings, y_hat = model(X)
            outlier_scores_batch = outlier_score(embeddings, traindata_means)
            outlier_scores[i*X.shape[0]:(i+1)*X.shape[0]] = outlier_scores_batch
    return outlier_scores