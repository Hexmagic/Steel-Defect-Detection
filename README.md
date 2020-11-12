# Steel-Defect-Detection
Code for Kaggle Steel Defect Detection
# Install 

Install segmentation_models.pytorch
```
pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
Install pytorch-lightning
```
pip install git+https://github.com/PytorchLightning/pytorch-lightning.git
```
# Result
Kaggle Steel defect detection

|        | model | lr_method | lr   | criterion | mIOU |
| ------ | ----- | --------- | ---- | --------- | ---- |
| target | unet  | radam     | 7e-5 | bce       | 0.6  |
| 2      | fpn   | adamW     | 7e-5 | bce+dice  | 0. 62|


failed:
1. ColorJitter(0.4, 0.3, 0.3), Erasing