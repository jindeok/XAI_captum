import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Define image classifier

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        # about the network specifications
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Define XAI framework

class XAI:
    
    def __init__(self, model, xai_model):
        
        self.model = model
        self.xai_model = xai_model
        
    def explain(self, input, target_class):
        
        attributions, delta = self.xai_model.attribute(input, target = target_class, return_convergence_delta=True)
        # print('IG Attributions:', attributions)
        # print('Convergence Delta:', delta)
        
        return attributions

    
    def initial_training(self, trainloader, epochs = 2, learning_rate = 0.001):
        
        # initialize optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr= learning_rate, momentum=0.9)
        
        #train model
        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches, replace to tqdm later on.
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        
        print('Finished Training')
        # If you want to save model
        # PATH = './cifar_net.pth'
        # torch.save(self.model.state_dict(), PATH)
        
    def show_heatmap(self, trainloader, target_img_idx = 1):
        
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        outputs = self.model(images)
        # show original image
        imshow(torchvision.utils.make_grid(images[target_img_idx]))
        # perform XAI and its heatmap
        target_class_idx = outputs[target_img_idx].argmax(0)
        heatmap = self.explain(images, target_class = target_class_idx)
        # visualize heatmap
        # aggregate 3 channel by summing up.
        heatmap = heatmap.sum(axis=np.argmax(np.asarray(heatmap.shape) == 3))
        plt.imshow(heatmap[0], cmap="seismic", clim=(-0.25, 0.25))


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    