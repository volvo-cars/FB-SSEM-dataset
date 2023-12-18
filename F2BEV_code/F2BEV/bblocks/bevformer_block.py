#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:47:35 2022

@author: Ekta
"""

import torch
import torch.nn as nn
from bblocks.spatial_cross_attention import SpatialCrossAttention
from bblocks.temporal_self_attention import TemporalSelfAttention
from bblocks.ffn import FFN

class BEVFormerBlock(nn.Module):
    def __init__(self):
        super(BEVFormerBlock, self).__init__()
        self.norm = nn.LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        self.bev_h = 50
        self.bev_w = 50
        self.ffn = FFN()
        self.sca = SpatialCrossAttention()
        self.tsa = TemporalSelfAttention()
        
    def forward(self,bev_query,key,value,bev_pos,spatial_shapes,level_start_index,reference_points_cam,bev_mask,ref_2d,prev_bev=None):
        bs = bev_query.size(0)
        #print(spatial_shapes)
        #print(reference_points_cam.shape)

        # x = self.sca(query = bev_query, key = key, value = value, query_pos = bev_pos,
        #              spatial_shapes = spatial_shapes, level_start_index = level_start_index,
        #              reference_points_cam = reference_points_cam,bev_mask = bev_mask )

        batch_reference_points_cam = reference_points_cam.repeat(1,bs,1,1,1)
        batch_bev_mask = bev_mask.repeat(1,bs,1,1)    

        ##add temporal here
        x = self.tsa(query = bev_query, key = prev_bev, value = prev_bev, query_pos = bev_pos,
                              reference_points = ref_2d, spatial_shapes=torch.tensor([[self.bev_h, self.bev_w]], device=bev_query.device),
                              level_start_index=torch.tensor([0], device=bev_query.device))
        x = self.norm(x)
        x = self.sca(query = x, key = key, value = value, query_pos = bev_pos,
                      spatial_shapes = spatial_shapes, level_start_index = level_start_index,
                      reference_points_cam = batch_reference_points_cam,bev_mask = batch_bev_mask )
        #print(x.shape)
        x = self.norm(x)
        x = self.ffn(x)
        x = self.norm(x)
        
        
        return x


        # x = self.sca(query = bev_query, key = key, value = value, residual = None, query_pos = bev_pos,
        #              key_padding_mask = None, reference_points = None, spatial_shapes = spatial_shapes, level_start_index = level_start_index,
        #              reference_points_cam = reference_points_cam,bev_mask =bev_mask )
        
        
