import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import codecs
from io import BytesIO

SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}

def read_imagenet(data_bytes_arr, labels, params, train=False, logFile=None):
    """
    read Imagenet dataset from data_bytes and return the result in the form of dataloader
    :param data_bytes_arr:      array of images encoded in bytes format
    :param labels_bytes:        labels in txt format
    :param params:             	dict of parameters passed to the ML task
    :param train:              	flag to mark is it the training or the test set
    :param logFile:             file handle to log whatever in it (for debugging purposes)
    """
    images = [np.array(Image.open(BytesIO(data_bytes)).convert('RGB')) for data_bytes in data_bytes_arr]
    imgs = np.array(images)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = InMemoryDataset(imgs, labels=labels, transform=transform)
    batch_size = int(params['Batch-Size']) if 'Batch-Size' in params.keys() else 100
    assert batch_size > 0 and batch_size <= len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def read_mnist(data_bytes, labels_bytes, params, train=False, logFile=None):
    """
    read MNIST dataset from data_bytes and return the result in the form of dataloader
    :param data_bytes:          the data source encoded in bytes format
    :param labels_bytes:	labels in byte format
    :param params:              dict of parameters passed to the ML task
    :param train:               flag to mark is it the training or the test set
    :param logFile:             file handle to log whatever in it (for debugging purposes)
    """
    imgs = read_sn3_pascalvincent_tensor(data_bytes, strict=False)
    labels = None
    if labels_bytes:
        labels = read_sn3_pascalvincent_tensor(labels_bytes, strict=False).long()
    #get start and end of images needed to be inferenced
    start = int(params['Start']) if 'Start' in params.keys() else 0
    end = int(params['End']) if 'End' in params.keys() else len(imgs)
    assert start >= 0 and end <= len(imgs)
    #do the necessary transformation
    transform=transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset = InMemoryDataset(imgs, labels=labels, transform=transform, logFile=logFile)
    batch_size = int(params['Batch-Size']) if 'Batch-Size' in params.keys() else 100
    assert batch_size > 0 and batch_size <= len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def read_cifar(data_bytes, params, train=False, logFile=None):
    """
    read cifar (10 or 100) dataset from data_bytes and return the result in the form of dataloader
    :param data_bytes:		the data source encoded in bytes format
    :param params:		dict of parameters passed to the ML task
    :param train:		flag to mark is it the training or the test set
    :param logFile:		file handle to log whatever in it (for debugging purposes)
    """
    data = pickle.loads(data_bytes, encoding='bytes')
    imgs = data[b'data']
    labels = None
    if train:
        labels = data[b'labels'] if b'labels' in data else data[b'fine_labels']
    #get start and end of images needed to be inferenced
    start = int(params['Start']) if 'Start' in params.keys() else 0
    end = int(params['End']) if 'End' in params.keys() else len(imgs)
    assert start >= 0 and end <= len(imgs)
    #the next two lines mimic the official source code of PyTorch:
    #https://pytorch.org/vision/0.8/_modules/torchvision/datasets/cifar.html#CIFAR10
    imgs = np.vstack(imgs[start:end]).reshape(-1, 3, 32, 32).transpose((0,2,3,1))
    #do the necessary transformation
    if train:
        transform = transforms.Compose([
   		transforms.RandomCrop(32, padding=4),
    		transforms.RandomHorizontalFlip(),
   		transforms.ToTensor(),
   		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
    else:
        transform = transforms.Compose([
               transforms.ToTensor(),
               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    dataset = InMemoryDataset(imgs, labels=labels, transform=transform)
    batch_size = int(params['Batch-Size']) if 'Batch-Size' in params.keys() else 100
    assert batch_size > 0 and batch_size <= len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

#Wrapper to datasets already loaded in memory
class InMemoryDataset(Dataset):
    """In memory dataset wrapper."""

    def __init__(self, dataset, labels=None, transform=None, logFile=None):
        """
        Args:
            dataset (ndarray): array of dataset samples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.dataset = dataset
        self.labels = labels
        self.transform = transform
        self.logFile = logFile

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.dataset[idx]
        try:
            image = Image.fromarray(image)
        except:
            image = Image.fromarray(image.numpy(), mode='L')
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            return image, int(self.labels[idx])
        return image

#copied from https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

#copied from https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html
def read_sn3_pascalvincent_tensor(data: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert nd >= 1 and nd <= 3
    assert ty >= 8 and ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)
