# GoogLeNet on mammals dataset

from utils.Training import train
from utils.Datasets import Mammals
from utils.ClassificationModel import GoogLeNet, initialize_he

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# load data
# load mammals dataset
path = "../data/mammals"
labels = os.listdir(path)

images = np.array([])
targets = np.array([])
for n, label in enumerate(labels):
    sub = os.listdir(f'{path}/{label}')
    sub = [f'{path}/{label}/' + x for x in sub]
    images = np.concatenate((images,sub))
    targets = np.concatenate((targets,[n for x in range(len(sub))]))

data = pd.DataFrame({'path': images, 'labels': targets})
data.labels = data.labels.astype(int)

# train test split
trainidx, testidx = train_test_split(data.index,test_size=0.3)

# initialize dataset
resize = (224,224)
trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
trainset = Mammals(data.iloc[trainidx],trans)
testset = Mammals(data.iloc[testidx],trans)

batch = 128
trainloader = DataLoader(trainset,batch_size=batch,shuffle=True)
testloader = DataLoader(testset,batch_size=batch,shuffle=False)

# initialize model
cuda = torch.device('cuda')
model = GoogLeNet(45,lr=0.01,momentum=0.9,weight_decay=0.0002) # suggested from literature
model.initialize_weights(next(iter(trainloader))[0],initialize_he)
model.to(cuda)

# training, reduce lr 3 times after val acc stops improving
train(model,trainloader,testloader,epochs=50,delta=0.005,top_3=True)

model.lr /= 2
print(f'New LR: {model.lr}')
train(model,trainloader,testloader,epochs=50,delta=0.003,top_3=True)

model.lr /= 2
print(f'New LR: {model.lr}')
train(model,trainloader,testloader,epochs=50,delta=0,top_3=True)

# checkpoint
torch.save(model.state_dict(),'MammalsGoogLeNet.params')

# 0.589 top-1 val accuracy, prev models couldnt even train at all
# 0.627 top-3 val accuracy
# mammals dataset is very small, on average 300 labeled images for each category, only about 200 used for training
# some categories are even smaller, only about 200 labeled images
# image augmentations to expand training data

# wont evaluate aux classifiers on mammals dataset since train/test split changes each iteration
# model accuracy and training time could be influenced by random variations in data split