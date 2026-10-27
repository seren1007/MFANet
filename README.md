# MFANet
## Introduction
Official implementation of "Boundary-Aware and Trans-Dimensional Multi-Scale Attention for Semi-Supervised Prostate Zonal Segmentation". 
## Requirements
This repository is based on PyTorch 1.12.1, CUDA 11.3 and Python 3.10.0. All experiments in our paper were conducted on NVIDIA A100 GPU with an identical experimental setting.

To train a model,
```
python train_Prostate.py  #for Prostate Datasets training
python train_ACDC.py  #for ACDC training
``` 

To test a model,
```
python test_Prostate.py  #for Prostate Datasets testing
python test_ACDC.py  #for ACDC testing
```

## Acknowledgements
Our code is largely based on [BCP] and [SSL4MIS]. Thanks for these authors for their valuable work, hope our work can also contribute to related research.






