#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 17:07:00 2023

@author: Ekta
"""

import torch
import torch.nn as nn

from bblocks.backbone import Resnet34WithFPN

from bblocks.encoder_height import EncoderFLCW
from bblocks.mask_head_decoder_height import MaskHeadDecoder

class FisheyeBEVFormer(nn.Module):
    def __init__(self):
        super(FisheyeBEVFormer, self).__init__()
        self.backbone =  Resnet34WithFPN()

        self.encoder = EncoderFLCW()
        self.multiscale = True #False
        self.decoder = MaskHeadDecoder()
        self.deep_supervision = True

    def forward(self,front,left,rear,right,can_buses,prev_bev=None):      

        
        f_f  = self.backbone(front)
        f_l  = self.backbone(left)
        f_re  = self.backbone(rear)
        f_r  = self.backbone(right) 
        
        if self.multiscale:
            level0 = torch.cat((f_f['0'].unsqueeze(0),f_l['0'].unsqueeze(0),f_re['0'].unsqueeze(0),f_r['0'].unsqueeze(0)),axis=0).permute(1,0,2,3,4)
            level1 = torch.cat((f_f['1'].unsqueeze(0),f_l['1'].unsqueeze(0),f_re['1'].unsqueeze(0),f_r['1'].unsqueeze(0)),axis=0).permute(1,0,2,3,4)
            level2 = torch.cat((f_f['2'].unsqueeze(0),f_l['2'].unsqueeze(0),f_re['2'].unsqueeze(0),f_r['2'].unsqueeze(0)),axis=0).permute(1,0,2,3,4)
            level3 = torch.cat((f_f['3'].unsqueeze(0),f_l['3'].unsqueeze(0),f_re['3'].unsqueeze(0),f_r['3'].unsqueeze(0)),axis=0).permute(1,0,2,3,4)
            mlvl_feats = [level0,level1,level2,level3]            

        else:

            level0 = torch.cat((f_f[0].unsqueeze(0),f_l[0].unsqueeze(0),f_re[0].unsqueeze(0),f_r[0].unsqueeze(0)),axis=0).permute(1,0,2,3,4)
            mlvl_feats = level0.unsqueeze(0)
            
        bevfeatures = self.encoder(mlvl_feats,can_buses,prev_bev)
        output, inter_masks  = self.decoder(bevfeatures)

        if self.deep_supervision:
            return output, inter_masks, bevfeatures
        else:
            return output,bevfeatures
