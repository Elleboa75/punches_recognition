import torch
import torchvision

def get_bare_transforms(size:int=256):
    '''
    Returns a bare minimum transform for the dataset of punches.
    This includes a resizing to a common size (default=256x256 px) and a normalization.

    Parameters
    ----------
    size: an integer indicating the size of the resized images.

    Returns
    -------
    A pipeline of torchvision transforms.
    '''
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_dataloader(root, batch_size:int=32, num_workers:int=4, transforms=None, shuffle=True) -> torch.utils.data.DataLoader:
    dataset = torchvision.datasets.ImageFolder(root, transform=transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)