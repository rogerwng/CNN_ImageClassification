# Cloud Training on AWS EC2 Instance
Instance used: g4dn.xlarge  
  
This folder contains .py files for classification models, training utility functions, datasets, and main training scripts for various models and datasets.  
  
EC2 instance was used since these models are too large to fit on my host machine (Macbook Air M2). Using cloud GPUs also allowed me to work more in the Terminal using ssh tools and keys, as well as understanding the wide variety of instances and services provided by AWS that can help enhance machine learning training and evaluation.
  
## Models Implemented and Best Scores (validation accuracy):  
VGG on Imagenette                           - 0.695  
VGG on Imagenette (pretrained + batch norm) - 0.800  
GoogLeNet (Inception) on Fashion            - 0.835  
GoogLeNet on Imagenette                     - 0.800  
GoogLeNet on Mammals                        - 0.589 (top-1), 0.627 (top-3)  
ResNet on Fashion                           - 0.935  
ResNet on Imagenette                        - 0.764