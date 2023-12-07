#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:10:39 2022

@author: Ekta

"""

import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features = 256, out_features = 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2,inplace=False),
            nn.Linear(in_features = 512, out_features = 256, bias=True),
            nn.Dropout(p=0.2,inplace=False))
        
    def forward(self,x):
        x = self.layers(x)
        return x
