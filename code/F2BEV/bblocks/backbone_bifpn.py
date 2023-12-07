"""
Created on Wed Dec 28 10:51:01 2022

@author: Ekta
"""


import torch
import torch.nn as nn
from collections import OrderedDict
import timm
from typing import Callable
from .bifpn import BiFpn
#from functools import partial
#from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

def get_feature_info(backbone):
    if isinstance(backbone.feature_info, Callable):
        # old accessor for timm versions <= 0.1.30, efficientnet and mobilenetv3 and related nets only
        feature_info = [dict(num_chs=f['num_chs'], reduction=f['reduction'])
                        for i, f in enumerate(backbone.feature_info())]
    else:
        # new feature info accessor, timm >= 0.2, all models supported
        feature_info = backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
    return feature_info


    
class ResNet34BiFPN(nn.Module):
    def __init__(self):
        super(ResNet34BiFPN,self).__init__()
        self.backbone = timm.create_model(
            'resnet34', features_only=True,
            out_indices= (1, 2, 3, 4),
            pretrained=True)
        feature_info = get_feature_info(self.backbone)
        self.fpn = BiFpn(feature_info)

        
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        return x
