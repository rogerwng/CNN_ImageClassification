# Jupyter Notebooks for Image Classification
All notebooks in Visual Studio Code environment and using virtual conda environments. Since these models are trained on my host machine (Macbook Air M2), they include the preview version of PyTorch that supports M2 GPU acceleration.  
  
I noticed during experimentation an issue with calling argmax on PyTorch tensors on the MPS device. Although the function can be called without exceptions, it doesn't return a proper argmax on its logits. I found no solution online, but found one forum that blamed the issue with MPS implementations of PyTorch. For these models, argmax is performed only on the CPU, which involves moving the model and logits to and from the CPU and GPU repeatedly, which unfortunately increases the training time due to overhead costs.  
  
Imagenette is a subset of ImageNet.   
  
Models implemented and best scores:  
MLP on FashionMNIST - 0.860  
LeNet on FashionMNIST - 0.860  
AlexNet on FashionMNIST - 0.841  
AlexNet on Imagenette - 0.671  
VGG on FashionMNIST - 0.913  
  
Model accuracies can likely be improved with hyperparameter tuning (hyperparameters selected from paper), image augmentation techniques (not yet implemented), better optimizers/learning rate schedulers (Adam etc.), and longer training.