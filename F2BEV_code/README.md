# F2BEV: Bird's Eye View Generation from Surround-View Fisheye Camera Images for Automated Driving


## Requirements

Package requirements are included in the ```f2bev_conda_env.yml``` file. A conda virtual environment can be created using this file as follows:

```bash
conda env create --file f2bev_conda_env.yml
```


## Data

Download the FB-SSEM dataset from [here](https://github.com/volvo-cars/FB-SSEM-dataset). In particular, download ```.zip``` files corresponding to all twenty sequences, unzip them, and place them in a single folder named ```data```. This folder muster be placed inside the ```F2BEV``` folder generated from cloning this repository. 

## Compute Reference Points

A part of the reference point computation for the distortion-aware spatial cross attention in the network can be done offline to save on training time. 

Run the following commands from within the ```F2BEV``` folder.

```bash
cd pre_computation
python3 computeNormalizedReferencePoints.py
cd ../
```
The outputs of this code are already placed in the ```unity_data``` fold inside the ```pre_computation``` folder for convenience.

## Training and Testing F2BEV

Use ```train_<model_type>.py``` to train an F2BEV network and ```test_<model_type>.py``` to test a trained F2BEV network.

For e.g., to train an F2BEV network to generate (only) a discretized BEV height map using an attention-based task-specific head, where the height of every pixel is classified into one of three classes (below car bumper, above car height, or car height) run the following. The traininglog will be saved in ```traininglog_f2bev_attn_st_height.out```.

```bash
nohup python3 -u train_f2bev_attn_st_height.py > traininglog_f2bev_attn_st_height.out &
```

To test a trained F2BEv network, run ```test_<model_type>.py```.
For e.g., to test the above trained model, run the following

```bash
python3 test_f2bev_attn_st_height.py
```

Training and test scripts for the all model types discussed in the [F2BEV paper](https://arxiv.org/abs/2303.03651) are included in this repository. They are as follows


| <model_type> | Description |
| ------ | ------ |
| f2bev_attn_st_height | To generate discretized BEV height maps (alone) using an attention-based task-specific head, where the height of every pixel in classified into one of three classes |
| f2bev_attn_st_seg | To generate BEV semantic segmentation maps (alone) using an attention-based task-specific head |
| f2bev_attn_mt | To generate discretized BEV height maps and BEV semantic segmentation maps simultaneously using attention-based task-specific heads |
| f2bev_conv_st_height | To generate discretized BEV height maps (alone) using a convolution-based task-specific head, where the height of every pixel in classified into one of three classes|
| f2bev_conv_st_seg | To generate BEV semantic segmentation maps (alone) using a convolution-based task-specific head|
| f2bev_conv_mt | To generate discretized BEV height maps and BEV semantic segmentation maps simultaneously using convolution-based task-specific heads |

## Citation
If you find our code beneficial, please cite the [F2BEV paper](https://arxiv.org/abs/2303.03651)

```bash
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

