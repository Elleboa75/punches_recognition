import torch
from .. import utils

class IILoss(torch.nn.Module):
    '''
    Loss defined as in "Learning a Neural-network-based Representation for Open Set Recognition" (2018), Hassen & Chan.
    Intra_spread is sum of the squared distances between each data point and its class mean.
    Inter_separation is the minimum of the squared distances between each class mean and the mean of the other classes.
    The loss is defined as intra_spread - inter_separation.

    Attributes
    ----------
    delta: a float representing the maximum inter_separation between classes. It is used to prevent the inter_separation term from dominating the intra_spread term and other losses such as cross-entropy.
    '''
    def __init__(self, delta:float=float("inf")):
        self.delta = torch.Tensor([delta])

    def forward(self, embeddings:torch.Tensor, labels:torch.Tensor, num_classes:int) -> torch.Tensor:
        '''
        Compute the loss for the given embeddings and labels.

        Parameters
        ----------
        embeddings: a torch.Tensor of shape (N, D) where N is the number of data points and D is the embedding dimension.
        labels: a torch.Tensor of longs or ints of shape (N)
        num_classes: the number of classes. Needed in case some classes are not represented within the current mini-batch.

        Returns
        -------
        a singleton torch.Tensor representing the loss.
        '''
        n_datapoints = len(embeddings)
        device = embeddings.device
        intra_spread = torch.Tensor([0]).to(device)
        inter_separation = torch.Tensor([float("inf")]).to(device)
        class_mean = utils.bucket_mean(embeddings, labels, num_classes)
        empty_classes = []

        for j in range(num_classes):
            # update intra_spread
            data_class = embeddings[labels == j]
            if len(data_class) == 0:
                empty_classes.append(j)
                continue
            difference_from_mean = data_class - class_mean[j]
            norm_from_mean = difference_from_mean.norm()**2
            intra_spread += norm_from_mean
            # update inter_separation
            class_mean_previous = class_mean[list(set(range(j)).difference(empty_classes))]
            if class_mean_previous.shape[0] > 0:
                norm_from_previous_means = (class_mean_previous - class_mean[j]).norm(dim=1)**2
                inter_separation = min(inter_separation, norm_from_previous_means.min())
        
        return intra_spread/n_datapoints - torch.min(self.delta, inter_separation)

def outlier_score(embeddings:torch.Tensor, train_class_means:torch.Tensor):
    '''
    Compute the outlier score for the given batch of embeddings and class means obtained from the training set.
    The outlier score for a single datapoint is defined as min_j(||z - m_j||^2), where j is a category and m_j is the mean embedding of this class.

    Parameters
    ----------
    embeddings: a torch.Tensor of shape (N, D) where N is the number of data points and D is the embedding dimension.
    train_class_means: a torch.Tensor of shape (K, D) where K is the number of classes.

    Returns
    -------
    a torch.Tensor of shape (N), representing the outlier score for each of the data points.
    '''
    assert len(embeddings.shape) == 2, f"Expected 2D tensor of shape N ⨉ D (N=datapoints, D=embedding dimension), got {embeddings.shape}"
    assert len(train_class_means.shape) == 2, f"Expected 2D tensor of shape K ⨉ D (K=num_classes, D=embedding dimension), got {train_class_means.shape}"
    # create an expanded version of the embeddings of dimension N ⨉ K ⨉ D, useful for subtracting means
    embeddings_repeated = embeddings.unsqueeze(1).repeat((1, train_class_means.shape[0], 1))
    # compute the difference between the embeddings and the class means
    difference_from_mean = embeddings_repeated - train_class_means
    # compute the squared norm of the difference (N ⨉ K matrix)
    norm_from_mean = difference_from_mean.norm(dim=2)**2
    # get the min for each datapoint
    return norm_from_mean.min(dim=1).values


# def compute_ii_loss(out_z, labels, num_classes):
#     intra_spread = torch.Tensor([0]).cuda()
#     inter_separation = torch.tensor(float('inf')).cuda()# torch.inf.cuda()
#     class_mean = utils.bucket_mean(out_z, labels, num_classes) 
#     for j in range(num_classes):
#         data_class = out_z[labels == j]
#         difference_from_mean = data_class - class_mean[j]
#         norm_from_mean = difference_from_mean.norm()**2
#         intra_spread += norm_from_mean
#         class_mean_previous = class_mean[:j]
#         norm_form_previous_means = (class_mean_previous - class_mean[j]).norm()**2
#         inter_separation = min(inter_separation, norm_form_previous_means.min())

#     return intra_spread - inter_separation

# def compute_threshold(embeddings, mean):
#     outlier_score = []
#     for j in range(embeddings.shape[0]):
#         a=(mean - embeddings[j])
#         b=a.norm(dim=1)**2
#         c=b.min()
#         print(f"X-M:\n{a}\n\n||X-M||^2:\n{b}\n\nmin(||X-M||^2):\n{c}")
#         outlier_score.append(c) 
    
#     return outlier_score