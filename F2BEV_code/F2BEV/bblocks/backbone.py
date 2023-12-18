#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:19:36 2022

@author: Ekta
"""
## From: https://pytorch.org/vision/stable/feature_extraction.html
import torch
from torchvision.models import resnet50,resnet34 ,ResNet50_Weights, resnet34, ResNet34_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork



class Resnet34WithFPN(torch.nn.Module):
    def __init__(self):
        super(Resnet34WithFPN, self).__init__()
        # Get a resnet50 backbone
        #m = resnet50()
        #m = resnet50(weights=ResNet50_Weights.DEFAULT)
        m = resnet34(weights=ResNet34_Weights.DEFAULT)
        #m = resnet34()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        # print(resnet34)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                              for v, k in enumerate([1, 2, 3, 4])})
        inp = torch.randn(1, 3, 540, 640)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        #print(in_channels_list)
        # # Build FPN
        self.out_channels = 256
        # self.fpn = FeaturePyramidNetwork(
        #     in_channels_list, out_channels=self.out_channels,
        #     extra_blocks=LastLevelMaxPool())
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels)
    def forward(self, x):
        x = self.body(x)
        #print(x.keys(),x['0'].shape)
        x = self.fpn(x)
        return x
