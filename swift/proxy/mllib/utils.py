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
    models = {'convnet':Net,
		'cifarnet':Cifarnet,
		'cnn': CNNet,
		'resnet18':torchvision.models.resnet18,
                'resnet34':torchvision.models.resnet34,
                'resnet50':torchvision.models.resnet50,
                'resnet152':torchvision.models.resnet152,
		'inception':torchvision.models.inception_v3,
		'vgg16':torchvision.models.vgg16,
		'vgg19':torchvision.models.vgg19,
		'preactresnet18': PreActResNet18,
		'googlenet': GoogLeNet,
		'densenet121': DenseNet121,
		'resnext29': ResNeXt29_2x64d,
		'mobilenet': MobileNet,
		'mobilenetv2': MobileNetV2,
		'dpn92': DPN92,
		'shufflenetg2': ShuffleNetG2,
		'senet18': SENet18,
		'efficientnetb0': EfficientNetB0,
		'regnetx200': RegNetX_200MF}
    num_class_dict = {'mnist':10, 'cifar10':10, 'cifar100':100, 'imagenet':1000}
    if dataset not in num_class_dict.keys():
        raise ValueError("Provided dataset ({}) is not known!".format(dataset))
    num_classes = num_class_dict[dataset]
    if model_str not in models.keys():
        raise ValueError("Provided model ({}) is not known!".format(model_str))
    return models[model_str](num_classes=num_classes)

def get_mem_usage():
    #returns a dict with memory usage values (in GBs)
    mem_dict = psutil.virtual_memory()
    all_mem = np.array([v for v in mem_dict])/(1024*1024*1024)	#convert to GB
    return {"available":all_mem[1],	"used":all_mem[3],
	"free":all_mem[4], "active":all_mem[5],
	"buffers":all_mem[7],"cached":all_mem[8]}
