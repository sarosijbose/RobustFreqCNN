# RobustFreqCNN

## About

This repository contains the [PyTorch] implementation of the paper "Towards Frequency-Based Explanation for Robust CNN" [arxiv](https://arxiv.org/pdf/2005.03141.pdf). It primarly deals the extent to which image features are robust in the frequency domain. 

## Steps

1. It is recommended to setup a fresh virtual environment first.
```bash
python -m venv env
source activate env/bin/activate
```
2. Install the torchattacks package

```bash
pip install torchattacks
```
3. Run the ```main.py``` file. 
 
The original paper implemented the attacks using a VGG 19 model. However, due to memory constraints I did it using ResNet 18. Here I have provided a fine-tuned (on CIFAR 10) version of ResNet 18 which is pre-trained on ImageNet. The checkpoint can be downloaded using [this](https://drive.google.com/file/d/1bG5G-fgTahyuD8QdDd6yU-HWPMI_VXE7/view?usp=sharing) link. 

## RCT Maps

CW             |  FGSM       |  PGD
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/sarosijbose/RobustFreqCNN/blob/main/imgs/cw.png)  |  ![](https://github.com/sarosijbose/RobustFreqCNN/blob/main/imgs/fgsm.png)|  ![](https://github.com/sarosijbose/RobustFreqCNN/blob/main/imgs/pgd.png)

## Disclaimer

This is not an official implementation of the paper. I am not associated with the authors of the paper or Lab in any manner whatsoever and I don't claim credit for any of the algorithms proposed in the paper. 
