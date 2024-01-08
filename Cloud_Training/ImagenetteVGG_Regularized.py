# use Amazon AWS EC2 to train VGG on Imagenette dataset (subset of ImageNet)
# g4dn.xlarge instance

# VGG pretrained from model w/o batch norm in VGG block

from utils.Training import train
from utils.Datasets import Imagenette
from utils.ClassificationModel import VGGNet_Norm, initialize_xa

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
model = VGGNet_Norm(10,blocks_VGG_A,lr=0.01,momentum=0.9, weight_decay=0.0005)
# initialize weights w dummy input
model.initialize_weights(next(iter(trainloader))[0],initialize_xa)

# load model params from pretrained VGG w/o batch norm
pretrained = torch.load('ImagenetteVGG_A.params')
state_dict = model.state_dict()

# key match from notebook
match_keys = list(zip(['net.0.0.weight', 'net.0.0.bias', 'net.1.0.weight', 'net.1.0.bias', 'net.2.0.weight', 'net.2.0.bias', 'net.2.2.weight', 'net.2.2.bias', 'net.3.0.weight', 'net.3.0.bias', 'net.3.2.weight', 'net.3.2.bias', 'net.4.0.weight', 'net.4.0.bias', 'net.4.2.weight', 'net.4.2.bias', 'net.6.weight', 'net.6.bias', 'net.9.weight', 'net.9.bias', 'net.12.weight', 'net.12.bias'],
              ['net.0.0.weight', 'net.0.0.bias', 'net.1.0.weight', 'net.1.0.bias', 'net.2.0.weight', 'net.2.0.bias', 'net.2.3.weight', 'net.2.3.bias', 'net.3.0.weight', 'net.3.0.bias', 'net.3.3.weight', 'net.3.3.bias', 'net.4.0.weight', 'net.4.0.bias', 'net.4.3.weight', 'net.4.3.bias', 'net.6.weight', 'net.6.bias', 'net.9.weight', 'net.9.bias', 'net.12.weight', 'net.12.bias']))

for (pre,post) in match_keys:
    state_dict[post] = pretrained[pre]

model.load_state_dict(state_dict)
model.to(cuda)

# train, decrease lr after 25 epochs or early stoppage
for i in range(3):
    train(model,trainloader,valloader,epochs=25,delta=0)
    model.lr *= 0.9
    print(f'New Learning Rate: {model.lr}')

# save model params
torch.save(model.state_dict(), 'ImagenetteVGG_A_Reg.params')

# VGG w batch norm val accuracy 0.8, 45 epochs w pretraining
# add image augmentation to increase training datset, training loss is 0.037
# more data w image augmentation will allow model to learn more and increase scores