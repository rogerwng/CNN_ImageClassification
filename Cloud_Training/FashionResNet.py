# Residual Nets on FashionMNIST

from utils.Training import train
from utils.Datasets import FashionMNIST
from utils.ClassificationModel import ResNet, initialize_xa

import torch

# loading data
data = FashionMNIST(batch_size=128,resize=(96,96))
trainloader = data.train_dataloader()
valloader = data.val_dataloader()

# initialize model
cuda = torch.device('cuda')
arch_resnet18 = [(2,64),(2,128),(2,256),(2,512)]
model = ResNet(10,arch_resnet18,lr=0.01,momentum=0.9,weight_decay=0.005)
# initialize weights
model.initialize_weights(next(iter(trainloader))[0],initialize_xa)
model.to(cuda)

# train
train(model,trainloader,valloader,epochs=10)

model.lr *= 0.8
print(f'New LR: {model.lr}')
train(model,trainloader,valloader,epochs=10)

# checkpoint
torch.save(model.state_dict(),'FashionResNet.params')

# very fast training, 0.87 after first epoch
# val acc 0.935 after 18 epochs, no further progress bc training loss goes to zero here