#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:12:33 2022

@author: Ekta
"""

import torch
import torch.nn as nn
import numpy as np

class HeightMulticlassHead(nn.Sequential):

    def __init__(self, in_channels=16, out_channels=3, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        #softmax = nn.Softmax()  
        super().__init__(conv2d,upsampling)


class HeightHead(nn.Sequential):

    def __init__(self, in_channels=16, out_channels=1, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        relu = nn.ReLU()  
        super().__init__(conv2d,relu,upsampling)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels=16, out_channels=5, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        sigmoid = nn.Sigmoid()
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d,upsampling)
        #super().__init__(conv2d, sigmoid,upsampling)


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dropout = nn.Dropout(p=0.2) #new addition by me

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x
    

class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 256
        self.decoder_channels = (128,64,16)
        self.head_channels = 512
        self.n_skip = 0
        self.skip_channels = [256,64,16] ##dummy
        self.conv_more = Conv2dReLU(
            self.hidden_size,
            self.head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = self.decoder_channels
        head_channels = self.head_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.n_skip != 0:
            skip_channels = self.skip_channels
            for i in range(4-self.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #print(B, n_patch,hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        #print(x.shape)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            #print(x.shape)
        return x

class UpSampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return x

class BEVUpSample(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = [256,128,64]
        out_channels = [128,64,16]        
        blocks = [
            UpSampleBlock(in_ch, out_ch) for in_ch, out_ch in zip(in_channels, out_channels)
        ]
        
        self.blocks = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(p=0.2)
        self.final_head = HeightMulticlassHead()
        
    def forward(self,hidden_states,features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        #print(B, n_patch,hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        for i, up_block in enumerate(self.blocks):
            x = up_block(x)
            
        x = self.dropout(x)
        x = self.final_head(x)
        
        return x
