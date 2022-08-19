import torch
import torchvision
from typing import Collection
from torchvision import transforms as T

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
        torchvision.transforms.Resize((size, size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def get_dataset(root, transforms=None) -> torch.utils.data.Dataset:
    return torchvision.datasets.ImageFolder(root, transform=transforms)

def get_dataloader(root, batch_size:int=32, num_workers:int=4, transforms=None, shuffle=True) -> torch.utils.data.DataLoader:
    dataset = get_dataset(root, transforms)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

class BasicDataset(torch.utils.data.Dataset):
    '''
    A simple dataset wrapping a container of images without labels.
    '''
    def __init__(self, data, transform=None):
        '''
        Parameters:
        -----------
        data: a generic container of tensors
        transform: a pipeline of transformations to apply to the data
        '''
        self.data = data
        self.current_set_len = len(data)
        self.transform = transform
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curdata = self.data[idx]
        if self.transform is not None:
            return self.transform(curdata)
        return curdata

class BasicDatasetLabels(torch.utils.data.Dataset):
    '''
    A simple dataset wrapping a container of images with fake labels.
    '''
    def __init__(self, data, transform=None, label=0):
        '''
        Parameters:
        -----------
        data: a generic container of tensors
        transform: a pipeline of transformations to apply to the data
        '''
        self.data = data
        self.current_set_len = len(data)
        self.transform = transform
        self.label = label
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        curdata = self.data[idx]
        if self.transform is not None:
            return self.transform(curdata)
        return curdata, self.label