# Python Implementations of Various Classification Models, Datasets, and Training Loops
## Classification Models
Base module containing methods universal for all image classification models. These including initialization, forward, loss methods, etc. Models inherit base class.  
Also includes block classes for common blocks found in CNNs, such as VGG and Inception blocks.  
  
## Datasets
PyTorch Dataset classes for each corresponding dataset. These are passed into Data Loaders in training scripts.  
  
## Training
Utility functions for training loops. This includes an early stopping model for ending training once validation accuracy stops improving to prevent overfitting and save time. All training scripts use the basic train function. Training schedules implemented by calling this function successively.