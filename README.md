# Exploring CNN Models on Image Classification Tasks
PyTorch implementation of CNN architectures including LeNet, AlexNet, VGG, NiN, GoogLeNet, etc. 

Models evaluated on FashionMNIST and Imagenette (subset of Imagenet).

Many models use same boilerplate code for training loops, forward and loss methods, etc.
Larger models trained on AWS EC2 Instance (g4dn.xlarge).

Notebooks contains Jupyter Notebooks which some models were trained in. Larger models are trained on the EC2 instance and then parameters loaded into notebooks.
Notebooks also include some statistics on model performance such as precision and recall.

Cloud Training contains files uploaded to EC2 instance to train large models.
