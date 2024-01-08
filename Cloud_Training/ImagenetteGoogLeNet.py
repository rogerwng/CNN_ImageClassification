# GoogLeNet on Imagenette

from utils.Training import train
from utils.Datasets import Imagenette
from utils.ClassificationModel import GoogLeNet, initialize_he

import torch
from torchvision import transforms
from torch.utils.data import DataLoader


# loading data
path = "../data/imagenette2-160"
resize = (224,224)
trans = transforms.Compose([transforms.Resize(resize),transforms.ToTensor()])
trainset = Imagenette(True, root=path, transform=trans)
valset = Imagenette(False, root=path, transform=trans)

batch = 128
trainloader = DataLoader(trainset,batch,True)
valloader = DataLoader(valset,batch,False)

# initialize model
cuda = torch.device('cuda')
model = GoogLeNet(10,lr=0.01,momentum=0.9,weight_decay=0.0002) # suggested from literature
model.initialize_weights(next(iter(trainloader))[0],initialize_he)
model.to(cuda)

# training, reduce lr 3 times after val acc stops improving
train(model,trainloader,valloader,epochs=50,delta=0.005)

model.lr /= 2
print(f'New LR: {model.lr}')
train(model,trainloader,valloader,epochs=50,delta=0.003)

model.lr /= 2
print(f'New LR: {model.lr}')
train(model,trainloader,valloader,epochs=50,delta=0)

# checkpoint
torch.save(model.state_dict(),'ImagenetteGoogLeNet.params')

# val accuracy 0.8 initial training after 51 epochs