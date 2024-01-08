# branch networks on FashionMNIST

from utils.Training import train
from utils.Datasets import FashionMNIST
from utils.ClassificationModel import GoogLeNet, initialize_xa

import torch

# loading data
data = FashionMNIST(batch_size=128,resize=(96,96))
trainloader = data.train_dataloader()
valloader = data.val_dataloader()

# initialize model
cuda = torch.device('cuda')
model = GoogLeNet(10,lr=0.01)
# initialize weights
model.initialize_weights(next(iter(trainloader))[0],initialize_xa)
model.to(cuda)

# train
train(model,trainloader,valloader,epochs=10)

model.lr *= 0.8
print(f'New LR: {model.lr}')
train(model,trainloader,valloader,epochs=10)

# checkpoint
torch.save(model.state_dict(),'FashionGoogLeNet.params')

# val accuracy 0.835
# trains very fast and converges quickly