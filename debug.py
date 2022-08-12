import torch
from punches_lib.ii_loss import ii_loss as II

if __name__ == "__main__":
    # Z = torch.Tensor([[2,2],[3,3],[2,3], [-1,1],[-1,2],[-1,3], [-1,-1],[2,-1],[0,-3]])
    # y = torch.Tensor([0,0,0,1,1,1,2,2,2]).long()
    # print(dir(II))
    # ii = II.IILoss()
    # ii(Z,y,3)
    # II.compute_ii_loss(Z,y,3)
    Z = torch.load("embeddings.pt")
    y = torch.load("labels.pt")
    i = II.IILoss()(Z, y, 19)