# Datasets.py contains all dataset classes used for training

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

import os
from PIL import Image

# FashionMNIST
class FashionMNIST():
    def __init__(self, root='../data', batch_size=64, resize=(28,28)):
        self.batch_size = batch_size
        self.resize = resize

        # image transforms, resize and convert to tensor
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

        # initialize train (60k) and val (10k) sets
        self.train = torchvision.datasets.FashionMNIST(root=root,train=True,transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(root=root,train=False,transform=trans,download=True)

    # data loader for dataset
    def get_dataloader(self, train):
        data = self.train if train else self.val
        return DataLoader(data, self.batch_size, shuffle=train)
    
    # getting training data
    def train_dataloader(self):
        return self.get_dataloader(True)
    
    # getting val data
    def val_dataloader(self):
        return self.get_dataloader(False)

# Imagenette
class Imagenette(Dataset):
    def __init__(self, train, root="../data/imagenette2-160", transform=None):
        self.trans = transform
        # if train or val dataset
        if train:
            root+="/train"
        else:
            root+="/val"
        
        # collet data in df, original dataset has folders renamed to corresponding class names
        self.labels = os.listdir(root)
        self.images = np.array([])
        self.targets = np.array([])

        for n, label in enumerate(self.labels):
            sub = os.listdir(f'{root}/{label}')
            sub = [f'{root}/{label}/' + x for x in sub]
            self.images = np.concatenate((self.images,sub))
            self.targets = np.concatenate((self.targets,[n for x in range(len(sub))]))

    def __getitem__(self,idx):
        img = self.trans(Image.open(self.images[idx]).convert('RGB'))
        return img,self.targets[idx].astype('int')
    
    def __len__(self):
        return len(self.images)
    
# Mammals
class Mammals(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.images = data.path.to_list()
        self.labels = data.labels.to_list()
        self.trans = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.trans(Image.open(self.images[idx]))
        return img, self.labels[idx]