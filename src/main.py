import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import torchattacks

import matplotlib.pyplot as plt  
import numpy as np
from tqdm import tqdm
import math
import cv2
import time
import seaborn as sns
import itertools
import os
import scipy as sp
from scipy.fftpack import dct, idct

# Downloading and loading the data into train, test and adversarial sets."""

transformations = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = datasets.CIFAR10(root='data/',
                               train=True,
                              transform = transformations,
                              download=True)

test_dataset = datasets.CIFAR10(root='data/',
                               train=False,
                              transform = transformations,
                              download=True)

# This set loads the images in 32x32 (original resolution) instead of 224x224 unlike the above two.
adv_set = datasets.CIFAR10(root='data/',
                               train=False,
                              transform = transforms.ToTensor(),
                              download=True)

trainset = DataLoader(dataset=train_dataset, batch_size=500, shuffle=True)
testset = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)
advset = DataLoader(dataset=adv_set, batch_size=200, shuffle=True)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""# Load ResNet 18 pre-trained on Imagenet."""

net = models.resnet18(pretrained=True)

"""# Carrying out simple Transfer Learning."""

# freeze all the layers
for param in net.parameters():
    param.requires_grad = False
    
# set the final FC layer to 10 classes.
net.fc = nn.Linear(512, len(classes))

# unfreeze the last layer so it learns on CIFAR-10.
for param in net.fc.parameters():
    param.requires_grad = True

print(net)

"""# Training Regimen. """

# I lowered the learning rate for the last layer but it didn’t perform better. 
# If I unfreeze the layers and set a very low learning rate the model performance is improved. 

# However, I did'nt focus much on model metrics as reproducing the maps were important and time was very short.

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)

correct = total = 0

for i, epoch in enumerate(range(5)):
    for data in tqdm(trainset): 
        X, y = data 
        net.zero_grad()  
        output = net(X) 
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
        loss = F.nll_loss(output, y) 
        loss.backward()  
        optimizer.step()  
    print("\nTrain Accuracy: ", round((correct/total)*100, 4))

"""# Testing Regimen"""

correct = total = 0

with torch.no_grad():
    for data in tqdm(testset):
        X, y = data
        output = net(X)
        print(output.shape, type(output))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
        break

print("\nTest Accuracy: ", round((correct/total)*100, 4))

# Save model
# torch.save(net, '....../latest_model.pt')

# Restore model
net = torch.load('...../latest_model.pt')

# The model checkpoint above will be given to you in mail.

"""# Now replicate the RCT Maps"""

# Here I have used ResNet 18 to generate the RCT maps instead of VGG 19 as it would'nt fit into the RAM of Google colab due to
# it's excessive number of parameters. Hence, the generated maps may not be exactly similar to the ones shown in the paper.

# The maps are almost the same however.

# Function for 2D DCT.

def dct2d(img):

    gray_img = cv2.cvtColor(np.float32(transforms.ToPILImage()(img)), cv2.COLOR_BGR2GRAY)
    dct_img = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')
    return dct_img

# reconst_img = idct(idct(dct_img.T, norm='ortho').T, norm='ortho')

# PGD Attack

for batch_idx, data in enumerate(advset):
    imgs, labels = data
    # max. perturbation 0.15 and in Linf ball by default.
    attack1 = torchattacks.PGD(net, eps=0.15)
    adv_images = attack1(imgs, labels)
    break # We only need 200 images or a single batch here.

# Define RCT Map
rct = np.zeros((32, 32)) # Resnet has input 224x224 but in paper it is shown for 32x32. 

for adv_img, img in zip(adv_images, imgs):
    num = dct2d(adv_img)-dct2d(img)
    deno = dct2d(img)
    pre_rct = num/deno
    rct+=pre_rct

rct = rct/200 # Testing was done with 200 images in paper so N=200.
print(rct.shape)

ax = sns.heatmap(rct, vmin=0, vmax=1, cmap="YlGnBu")

# FGSM Attack

for batch_idx, data in enumerate(advset):
    imgs1, labels = data
    # max. perturbation 0.15 and in Linf ball by default.
    attack2 = torchattacks.FGSM(net, eps=0.15)
    adv_images1 = attack2(imgs1, labels)
    break

# Define RCT Map
rct1 = np.zeros((32, 32)) 

for adv_img, img in zip(adv_images1, imgs1):
    num = dct2d(adv_img)-dct2d(img)
    deno = dct2d(img)
    pre_rct = num/deno
    rct1+=pre_rct

rct1 = rct1/200
print(rct1.shape)

ax1 = sns.heatmap(rct1, vmin=0, vmax=1, cmap="YlGnBu")

# CW Attack

for batch_idx, data in enumerate(advset):
    imgs2, labels = data
    # L2 ball by default.
    attack3 = torchattacks.CW(net, c = 1)
    adv_images2 = attack3(imgs2, labels)
    break

# Define RCT Map
rct2 = np.zeros((32, 32)) 

for adv_img, img in zip(adv_images2, imgs2):
    num = dct2d(adv_img)-dct2d(img)
    deno = dct2d(img)
    pre_rct = num/deno
    rct2+=pre_rct

rct2 = rct2/200
print(rct2.shape)

ax2 = sns.heatmap(rct2, vmin=0, vmax=1, cmap="YlGnBu")

if __name__ == '__main__':
    main()