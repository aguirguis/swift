import psutil
import numpy as np
import torch
import torchvision
from swift.proxy.mllib.models import *

def get_model(model_str, dataset):
    """
    Returns a model of choice from the library, adapted also to the passed dataset
    :param model_str: the name of the required model
    :param dataset: the name of the dataset to be used
    :raises: ValueError
    :returns: Model object
    """
    num_class_dict = {'mnist':10, 'cifar10':10, 'cifar100':100, 'imagenet':1000}
    if dataset not in num_class_dict.keys():
        raise ValueError("Provided dataset ({}) is not known!".format(dataset))
    num_classes = num_class_dict[dataset]
    if model_str == 'cifarnet':
        model = Cifarnet(num_classes=num_classes)
    elif model_str == "resnet50":
        model = torchvision.models.resnet50(num_classes=num_classes)
    return model

def get_mem_usage():
    #returns a dict with memory usage values (in GBs)
    mem_dict = psutil.virtual_memory()
    all_mem = np.array([v for v in mem_dict])/(1024*1024*1024)	#convert to GB
    return {"available":all_mem[1],	"used":all_mem[3],
	"free":all_mem[4], "active":all_mem[5],
	"buffers":all_mem[7],"cached":all_mem[8]}
