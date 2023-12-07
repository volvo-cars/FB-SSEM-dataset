#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 16:45:22 2022

@author: Ekta
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class MonodepthLoss(nn.modules.Module):
    def __init__(self):
        super(MonodepthLoss, self).__init__()

    def gradient_x(self,img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
        #print(gx)
        return gx
    
    def gradient_y(self,img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
        return gy
    
    def disp_smoothness_fn(self,disp, img):
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        #print(torch.unique(torch.isnan(disp_gradients_x)))    
        #print(torch.unique(torch.isnan(disp_gradients_y)))        
        
        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img) 

        weight_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True)) 
        weight_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))
        
        smoothness_x = disp_gradients_x * weight_x
        smoothness_y = disp_gradients_y * weight_y
                        

        return torch.abs(smoothness_x) + torch.abs(smoothness_y)


    def forward(self, height, seg):
        disp = 1/height
        #print(torch.unique(torch.isnan(disp)))
        disp_smoothness = self.disp_smoothness_fn(disp, seg)

        loss = torch.mean(torch.abs(disp_smoothness))
                          
        return loss
        