# FB-SSEM-dataset

The FB-SSEM dataset is a synthetic dataset consisting of surround-view fisheye camera images and BEV maps from simulated sequences of ego car motion

## About
We use the Unity game engine to simulate a parking lot environment for our dataset. The parking lot consists of parked cars/trucks, buses, electric vehicle (EV) charging stations of varying dimensions, and large containers of varying heights (on the boundaries). All the vehicles in the parking lot, except the ego car, are static. For the ego car, we use a forward-looking wide camera to simulate its four surround-view fisheye cameras. Our dataset consists of 20 sequences of ego car motion through the parking lot environment. Each sequence represents a different parking lot setup, i.e., different placement of all the vehicles in the lot and ground textures. Each sequence consists of 1000 samples; each sample consists of RGB images from the four car-mounted fisheye cameras (i.e., front, left, rear, and right cameras) and the BEV camera. Corresponding semantic segmentation maps for all five views and normalized height maps for the BEV are also generated. In addition, ego-motion information (3D rotation and translation) corresponding to every sample is obtained. We consider five semantic classes for the BEV segmentation map: car (ego car and parked cars/trucks), bus, EV charger, ground, and a non-driveable area.

[F2BEV: Bird's Eye View Generation from Surround-View Fisheye Camera Images for Automated Driving](https://arxiv.org/abs/2303.03651)

## Dataset
Links to download FB-SSEM dataset are below. 12000 files per image sequence as described [here](https://fb-ssem.s3.us-west-2.amazonaws.com/README.pdf)  

* [Link to download data](https://fb-ssem.s3.us-west-2.amazonaws.com/index.html)

## Camera calibration parameters
* [Camera intrinsics](https://fb-ssem.s3.us-west-2.amazonaws.com/CameraCalibrationParameters/camera_intrinsics.yml)  
* [Camera positions for extrinsics](https://fb-ssem.s3.us-west-2.amazonaws.com/CameraCalibrationParameters/camera_positions_for_extrinsics.txt)
## Legal notice
* Volvo Cars Technology USA LLC is the sole and exclusive owner of this dataset.
* The datset is licensed under [CC BY-SA 4.0
](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
* Any public use, distribution, display of this data set must contain this notice in its entirety.

## Privacy
Volvo Cars takes reasonable care to remove or hide personal data.

## Public Distribution
When using the FB-SSEM dataset for public distribution, we would be glad if you cite our [paper](https://arxiv.org/abs/2303.03651). Please cite the following:

``` 
@article{samani2023f2bev, 
title={F2BEV: Bird's Eye View Generation from Surround-View Fisheye Camera Images for Automated Driving},
author={Samani, Ekta U and Tao, Feng and Dasari, Harshavardhan R and Ding, Sihao and Banerjee, Ashis G}, 
journal={arXiv preprint arXiv:2303.03651},
year={2023}}
```

## Contact
Harshavardhan R. Dasari  
mail    : harshavardhan.reddy.dasari@volvocars.com  
Ekta Samani  
mail    : eusamani@gmail.com
