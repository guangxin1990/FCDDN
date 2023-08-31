# FCDDN

This repository is FCDDN segmentation of thyroid nodules ultrasound images.

The Thyroid dataset contains 3794 ultrasound images. The training set, validation set and test set containing 3034, 380 and 380 images respectively.
The External Thyroid dataset contains 124 ultrasound images, for testing only.

# DSMA-Net Architecture
<div align="center">
  <img src="./picture/framework.png" width="450" height="200">
</div>
The framework of Fully Convolutional Dense Dilated Net (FCDDN). 

<div align="center">
  <img src="./picture/dilated1.png" width="80" height="80" />
  <img src="./picture/dilated2.png" width="80" height="80" />
  <img src="./picture/dilated3.png" width="80" height="80" />

  <img src="./picture/DDB.png" width="250" height="80" />

  <img src="./picture/layer1.png" width="80" height="200" />
  <img src="./picture/layer2.png" width="80" height="200" />
  <img src="./picture/layer3.png" width="80" height="200" />
</div>

Dense Layer, Dense Dilated Block and Dilated convolution.

<div align="center">
  <img src="./picture/b17170509141815.png" width="100" height="100" />
  <img src="./picture/fcddn_b17170509141815.png" width="100" height="100" />
  <img src="./picture/fcddn_cropb17170509141815.png" width="80" height="50" />
  
  <img src="./picture/b7120820135000.png" width="100" height="100" />
  <img src="./picture/fcddn_b7120820135000.png" width="100" height="100" />
  <img src="./picture/fcddn_cropb7120820135000.png" width="80" height="50" />
</div>
Segmentation results

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Usage
`FCDDN.py` is our proposed network, `train.py` is used for training, `test.py` is used for testing, `predict.py` is used to display the segmented image, and `parameters.py` is used to calculate the number of parameters.
