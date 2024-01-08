# classes for image classification models

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD

# base classification model
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    # initialize weights with dummy input
    def initialize_weights(self, input, init=None):
        self.forward(input)
        if init is not None:
            self.net.apply(init)

    def forward(self,x):
        return self.net(x)
    
    # cross-entropy loss
    def loss(self, batch):
        logits, labels = batch
        return F.cross_entropy(logits, labels) # mean reduction default

    def accuracy(self, batch, averaged=True):
        logits, labels = batch
        predict = self.predict(logits).type(labels.dtype)
        compare = (predict == labels).type(torch.float32)
        return compare.mean() if averaged else compare
    
    # return top_3 accuracy
    def accuracy_3(self, batch, averaged=True):
        logits, labels = batch
        predict = self.predict_3(logits)
        compare = (predict == labels.reshape(logits.shape[0],1).expand(logits.shape[0],3).to(predict.device)).type(torch.float32)
        return compare.sum()/len(compare) if averaged else compare

    def predict(self, logits):
        return logits.argmax(axis=1) 
    
    # return top_3 pred
    def predict_3(self, logits):
        # logits dim (batch size, # classes)
        ans = torch.zeros(logits.shape[0],3)

        for i in range(3):
            pred = torch.argmax(logits,dim=1)
            ans[:,i] = pred
            logits[:,pred] = 0

        return ans

    def initOptimizer(self):
        self.optim = SGD(self.parameters(), self.lr, self.momentum, self.weight_decay)

# weight initialization functions passed to ClassifierModel's initialize_weights
# model does not converge without initialization
def initialize_xa(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight) 
def initialize_he(module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.kaiming_uniform_(module.weight) 

# conv block used in LeNet
class ConvBlock(nn.Module):
    def __init__(self, channels, c_kernel, c_stride, c_padding, p_kernel, p_stride):
        super().__init__()
        
        self.net = nn.Sequential(nn.LazyConv2d(channels,c_kernel,c_stride,c_padding),
                                 nn.ReLU(),
                                 nn.MaxPool2d(p_kernel,p_stride))

    def forward(self,x):
        return self.net(x)

# LeNet
class LeNet(ClassificationModel):
    def __init__(self, outputs, lr, momentum, weight_decay):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.net = nn.Sequential(ConvBlock(6,5,1,2,2,2), # conv: 6 channels, kernel=5 w 1 stride and 2 padding, pool: kernel = 2, stride 2
                                 ConvBlock(16,5,1,0,2,2), # conv: 16 channels no padding
                                 nn.Flatten(),
                                 nn.LazyLinear(120), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.LazyLinear(84), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.LazyLinear(outputs),
                                 #nn.Dropout(0.5), # dropout on output layer as shown in MLP notebook increases acc in FashionMNIST due to noisy labels, remove for other datasets
                                 )
        
# AlexNet
class AlexNet(ClassificationModel):
    def __init__(self, outputs, lr, momentum, weight_decay):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.net = nn.Sequential(nn.LazyConv2d(out_channels=96,kernel_size=11,stride=4),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
                                 nn.LazyConv2d(out_channels=256,kernel_size=5,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
                                 nn.LazyConv2d(out_channels=384,kernel_size=3,padding=1),nn.ReLU(),
                                 nn.LazyConv2d(out_channels=384,kernel_size=3,padding=1),nn.ReLU(),
                                 nn.LazyConv2d(out_channels=256,kernel_size=3,padding=1),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2),
                                 nn.Flatten(),
                                 nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
                                 nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
                                 nn.LazyLinear(outputs))
        
# VGG block, sequence of convolutions followed by 2x2 max pooling
# VGG block func takes # of conv and # of output channels for the block
def VGGBlock(nconv, noutputs):
    block = []
    for i in range(nconv):
        block.append(nn.LazyConv2d(noutputs,kernel_size=3,padding=1))
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*block)

# VGG net A
# blocks_list parameter is list of tuples for each block: [(num conv, num output channels), ....]
class VGGNet(ClassificationModel):
    def __init__(self, num_classes, blocks_list, lr=0.1, momentum=0, weight_decay=0):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # create VGG blocks
        blocks = []
        for block_nconv, block_noutputs in blocks_list:
            blocks.append(VGGBlock(block_nconv,block_noutputs))
        
        # flatten and FC layers after VGG blocks
        self.net = nn.Sequential(*blocks,nn.Flatten(),
                                 nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
                                 nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
                                 nn.LazyLinear(num_classes))
        
# VGG block w batch norm
def VGGBlock_Norm(nconv, noutputs):
    block = []
    for i in range(nconv):
        block.append(nn.LazyConv2d(noutputs,kernel_size=3,padding=1))
        block.append(nn.BatchNorm2d(noutputs))
        block.append(nn.ReLU())
    block.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*block)
        
# VGG net A, add batch norm and increase dropout for added regularization
class VGGNet_Norm(ClassificationModel):
    def __init__(self, num_classes, blocks_list, lr=0.1, momentum=0, weight_decay=0):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # create VGG blocks
        blocks = []
        for block_nconv, block_noutputs in blocks_list:
            blocks.append(VGGBlock_Norm(block_nconv,block_noutputs))
        
        # flatten and FC layers after VGG blocks
        self.net = nn.Sequential(*blocks,nn.Flatten(),
                                 nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.7),
                                 nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.7),
                                 nn.LazyLinear(num_classes))
        
# NiN Block, normal conv layer followed by 2 1x1 conv 
def NiNBlock(nchannels, kernel, padding, stride):
    return nn.Sequential(nn.LazyConv2d(nchannels,kernel_size=kernel,stride=stride,padding=padding), nn.ReLU(),
                         nn.LazyConv2d(nchannels,kernel_size=1), nn.ReLU(),
                         nn.LazyConv2d(nchannels,kernel_size=1), nn.ReLU())
# NiN
class NiN(ClassificationModel):
    def __init__(self,noutputs,lr=0.1,momentum=0,weight_decay=0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # same conv sizes as AlexNet: 11x11, 5x5, 3x3. each followed by max pooling stride 2 3x3
        self.net = nn.Sequential(NiNBlock(nchannels=96, kernel=11, padding=0, stride=4),nn.MaxPool2d(kernel_size=3, stride=2),
                                 NiNBlock(nchannels=256, kernel=5, padding=2, stride=1),nn.MaxPool2d(kernel_size=3, stride=2),
                                 NiNBlock(nchannels=384, kernel=3, padding=1, stride=1),nn.MaxPool2d(kernel_size=3,stride=2),
                                 nn.Dropout(0.5),
                                 NiNBlock(nchannels=noutputs, kernel=3, padding=1, stride=1),
                                 nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        
# inception block
class InceptionBlock(nn.Module):
    def __init__(self, b1_nchannels, b2_nchannels, b3_nchannels, b4_nchannels):
        super().__init__()
        self.nchannels = [b1_nchannels,b2_nchannels,b3_nchannels,b4_nchannels]

        # branch 1, 1x1 conv
        self.branch_1 = nn.Sequential(nn.LazyConv2d(self.nchannels[0],kernel_size=1),nn.ReLU())
        # branch 2, 1x1 conv for dimensionality reduction and then 3x3 conv, padding 1
        self.branch_2 = nn.Sequential(nn.LazyConv2d(self.nchannels[1][0],kernel_size=1),nn.ReLU(),
                                      nn.LazyConv2d(self.nchannels[1][1],kernel_size=3,padding=1),nn.ReLU())
        # branch 3, 1x1 conv, 5x5 conv padding 2
        self.branch_3 = nn.Sequential(nn.LazyConv2d(self.nchannels[2][0],kernel_size=1),nn.ReLU(),
                                      nn.LazyConv2d(self.nchannels[2][1],kernel_size=5,padding=2),nn.ReLU())
        # branch 4, 3x3 max pooling and 1x1 conv
        self.branch_4 = nn.Sequential(nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
                                      nn.LazyConv2d(self.nchannels[3],kernel_size=1),nn.ReLU())
        
    def forward(self,x):
        return torch.cat((self.branch_1(x),self.branch_2(x),self.branch_3(x),self.branch_4(x)), dim=1)
    
# GoogLeNet class    
class GoogLeNet(ClassificationModel):
    def __init__(self,nclasses,lr=0.1,momentum=0,weight_decay=0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # divide architecture in 3 blocks to make it easier for intermediate predictors implementation
        self.block_1 = nn.Sequential(nn.LazyConv2d(64,kernel_size=7,stride=2,padding=3),nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                                     nn.LazyConv2d(64,kernel_size=1),nn.ReLU(),
                                     nn.LazyConv2d(192,kernel_size=3,padding=1),nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                                     InceptionBlock(64,(96,128),(16,32),(32)),
                                     InceptionBlock(128,(128,192),(32,96),(64)),
                                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                                     InceptionBlock(192,(96,208),(16,48),64))
        self.block_2 = nn.Sequential(InceptionBlock(160,(112,224),(24,64),64),
                                     InceptionBlock(128,(128,256),(24,64),64),
                                     InceptionBlock(112,(144,288),(32,64),64))
        self.block_3 = nn.Sequential(InceptionBlock(256,(160,320),(32,128),128),
                                     nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                                     InceptionBlock(256,(160,320),(32,128),128),
                                     InceptionBlock(384,(192,384),(48,128),128),
                                     nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),
                                     nn.LazyLinear(nclasses))
        
        self.net = nn.Sequential(self.block_1,self.block_2,self.block_3)

# aux predictors class
class AuxPredictor(nn.Module):
    def __init__(self,nclasses):
        super().__init__()

        self.net = nn.Sequential(nn.AvgPool2d(kernel_size=5,stride=3),
                                 nn.LazyConv2d(128,kernel_size=1),nn.ReLU(),nn.Flatten(),
                                 nn.LazyLinear(1024),nn.ReLU(),nn.Dropout(0.7),
                                 nn.LazyLinear(nclasses))
        
    def forward(self,x):
        return self.net(x)
    
    # cross-entropy loss
    def loss(self, batch):
        logits, labels = batch
        return F.cross_entropy(logits, labels) # mean reduction default
    
# GoogLeNet with aux classifiers at intermediate stages to help with gradient propogation
class GoogLeNet_Aux(GoogLeNet):
    def __init__(self,nclasses,lr=0.1,momentum=0,weight_decay=0):
        super().__init__(nclasses,lr,momentum,weight_decay)

        # aux classifiers
        self.aux_1 = AuxPredictor(nclasses)
        self.aux_2 = AuxPredictor(nclasses)

    # save intermediate logits of aux classifiers during training
    def forward(self,x):
        if self.training:
            int_logits = self.block_1(x)
            self.logit_1 = self.aux_1(int_logits)
            int_logits = self.block_2(int_logits)
            self.logit_2 = self.aux_2(int_logits)

            return self.block_3(int_logits)
        else:
            return self.block_3(self.block_2(self.block_1(x)))
        
    # loss of aux classifiers added to total loss during training
    def loss(self, batch):
        logit_3, labels = batch
        if self.training:
            return F.cross_entropy(logit_3,labels) + F.cross_entropy(self.logit_2,labels) + F.cross_entropy(self.logit_1,labels)
        else:
            return F.cross_entropy(logit_3,labels)