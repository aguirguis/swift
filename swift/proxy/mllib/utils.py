import psutil
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from swift.proxy.mllib.models import *

def get_optimizer(model, optimizer, *args, **kwargs):
    """ Select optimizer to use
    Args
    model        the model to optimize
    optimizer    optimizer name required to be initialized
    device       device to put model on (cuda or cpu)
    """
    optimizers = {'sgd': optim.SGD,
		'adam': optim.Adam,
		'adamw':optim.AdamW,
		'rmsprop': optim.RMSprop,
		'adagrad': optim.Adagrad}
    if optimizer in optimizers.keys():
        return optimizers[optimizer](model.parameters(),  *args, **kwargs)
    else:
        print("The selected optimizer is undefined, the available optimizers are: ", optimizers.keys())
        raise

def get_loss(loss_fn):
    """ Select loss function to optimize with
    Args
    loss_fn        Name of the loss function to optimize against
    """
    losses = {'nll': nn.NLLLoss, 'cross-entropy':nn.CrossEntropyLoss}
    if loss_fn in losses.keys():
        return losses[loss_fn]()
    else:
        print("The selected loss function is undefined, available losses are: ", losses.keys())
        raise

def get_model(model_str, dataset, device):
    """
    Returns a model of choice from the library, adapted also to the passed dataset
    :param model_str: the name of the required model
    :param dataset: the name of the dataset to be used
    :param device: CPU or GPU
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
    model = models[model_str](num_classes=num_classes)
    model = model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)
        cudnn.benchmark = True
    return model

def get_mem_usage():
    #returns a dict with memory usage values (in GBs)
    mem_dict = psutil.virtual_memory()
    all_mem = np.array([v for v in mem_dict])/(1024*1024*1024)	#convert to GB
    return {"available":all_mem[1],	"used":all_mem[3],
	"free":all_mem[4], "active":all_mem[5],
	"buffers":all_mem[7],"cached":all_mem[8]}
