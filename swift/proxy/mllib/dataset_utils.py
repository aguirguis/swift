import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

def read_cifar(data_bytes, params, train=False):
    """
    read cifar (10 or 100) dataset from data_bytes and return the result in the form of dataloader
    :param data_bytes:		the data source encoded in bytes format
    :param params:		dict of parameters passed to the ML task
    :param train:		flag to mark is it the training or the test set
    """
    data = pickle.loads(data_bytes, encoding='bytes')
    imgs = data[b'data']
    #get start and end of images needed to be inferenced
    start = int(params['Start']) if 'Start' in params.keys() else 0
    end = int(params['End']) if 'End' in params.keys() else len(imgs)
    assert start >= 0 and end <= len(imgs)
    #the next two lines mimic the official source code of PyTorch:
    #https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10
    imgs = np.vstack(imgs[start:end]).reshape(-1, 3, 32, 32).transpose((0,2,3,1))
    #do the necessary transformation
    if not train:
        transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    imgs = InMemoryDataset(imgs, transform)
    batch_size = int(params['Batch-Size']) if 'Batch-Size' in params.keys() else 100
    assert batch_size > 0 and batch_size <= len(imgs)
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
    return dataloader

#Wrapper to datasets already loaded in memory
class InMemoryDataset(Dataset):
    """In memory dataset wrapper."""

    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset (ndarray): array of dataset samples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.dataset[idx]
        if self.transform:
            image = self.transform(image)

        return image
