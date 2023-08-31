# FCDDN

This repository is a segmentation of thyroid nodule ultrasound images using FCDDN.

The dataset contains 3794 ultrasound images. The training set, validation set and test set containing 3034, 380 and 380 images respectively.

# DSMA-Net Architecture
<div align="center">
  <img src="./picture/framework.png" width="600" height="350">
</div>
The framework of Fully Convolutional Dense Dilated Net (FCDDN). 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Usage
`FCDDN.py` is our proposed network, `train.py` is used for training, `test.py` is used for testing, `predict.py` is used to display the segmented image, and `parameters.py` is used to calculate the number of parameters.
