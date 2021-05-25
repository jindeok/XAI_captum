import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms

from utils import *
from models import Net, XAI

from captum.attr import visualization as viz
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    LRP,
)


def parse_args():
    '''
    Parses the arguments.
    '''
    parser = argparse.ArgumentParser(description="Run XAI captum framework (CIFAR 10 currently)")
    parser.add_argument('--batch_size', nargs='?', default= 12, help = "batchsize for CIFAR 10 training")  
    parser.add_argument('--epoch', nargs='?', default= 1, help = "# of epochs for training")  
    parser.add_argument('--lr', nargs='?', default= 0.003, help = "learning rate for optimizer")  
    parser.add_argument('--img_idx', nargs='?', default= 0, help = "target img to explain")  
    parser.add_argument('--XAI_method', nargs='?', default = IntegratedGradients, help = "IntegratedGradients, DeepLift, ...")

    return parser.parse_args()

args = parse_args()


if __name__ == "__main__": 
    
    data = DataLoader(batch_size = args.batch_size)    
    trainloader = data.trainloader
    testloader = data.testloader
    
    model = Net()
    
    prune_xai = XAI(model, xai_model=args.XAI_method(model))
    prune_xai.initial_training(trainloader, epochs = args.epoch, learning_rate = args.lr)
    prune_xai.show_heatmap(trainloader, target_img_idx = args.img_idx)



