# use Amazon AWS EC2 to train VGG on Imagenette dataset (subset of ImageNet)
# g4dn.xlarge instance

from utils.Training import train_schedule
from utils.Datasets import Imagenette
from utils.ClassificationModel import VGGNet, initialize

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
blocks_VGG_A = [(1,64),(1,128),(2,256),(2,512),(2,512)] # VGG A config, baseline for pretraining larger VGG
model = VGGNet(10,blocks_VGG_A,lr=0.01,momentum=0.9, weight_decay=0.0005)
# initialize weights w dummy input
model.initialize_weights(next(iter(trainloader))[0],initialize)
model.to(cuda)

# train
train_schedule(model,trainloader,valloader)

# save params
torch.save(model.state_dict(),'ImagenetteVGG_A.params')

# initial training, 0.695 val accuracy
# training loss 0.009 while val score constant in 0.69 range, suggests overfitting