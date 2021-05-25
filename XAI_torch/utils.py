import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms



class DataLoader:
    
    def __init__(self, batch_size = 4):
        '''
        num_workers should be 0 in window env (otherwise pipe_error occurs)
        '''
        print("---- downloading dataset from online... ----")        
        trainset, testset = self.download_data()        
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0) 
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=0)
        
        
    def download_data(self):
        '''
        download CIFAR-10 data which can be replaced by MNIST or etc later on.
        '''
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        return trainset, testset
    
    def show_image(self):
        '''
        Just for the testing.
        '''
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()
        
        # show images
        imshow(torchvision.utils.make_grid(images))
      
        
      
        
      
        
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
        
        

    

    