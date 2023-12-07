#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:09:49 2022

@author: Ekta
"""

# ------------------------------------------------------------------------------------------------# ------------------------------------------------------------------------------------------------
# Modified from https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from bblocks.deformable_attention_function import ms_deform_attn_core_pytorch #,MSDeformAttnFunction

#from bblocks.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0 #checked and same


class MSDeformAttn3D(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=6, batch_first= True):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model #bevformer calls this embed_dims
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        #self.output_proj = nn.Linear(d_model, d_model) ##TODO: this is new in my implementation
        self.batch_first = True
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        #xavier_uniform_(self.output_proj.weight.data)
        #constant_(self.output_proj.bias.data, 0.) ##these are all mostly doing the same thing; calculated guess


    def forward(self, query, query_pos, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        
        #######bev former has this
        
        if input_flatten is None:
            input_flatten = query
        # if query_pos is not None:
        #     query = query + query_pos  ##I think BEVformer had this but its an error
        

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            input_flatten = input_flatten.permute(1, 0, 2)    
            
        ##################    
        N, Len_q, _ = query.shape ##capital N is batchsize. i.e. bs in BEVformer
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None: ##TODO: Figure out the deal with masks
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        #print(attention_weights.shape)
            
        if reference_points.shape[-1] == 2:
            
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            #print(offset_normalizer)
            
            ##added by me : this is where sampling points are obtained in a manner that works with SCA
            N, Len_q, num_Z_anchors,xy = reference_points.shape
            #print(num_Z_anchors)
            reference_points = reference_points[:, :, None, None, None, :, :]
            #print(reference_points.shape)
            #print(sampling_offsets.shape)
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            #print(sampling_offsets.shape)
            N, Len_q, n_heads,n_levels,num_all_points,xy = sampling_offsets.shape
            #print(num_all_points)
            #print(num_Z_anchors)
            sampling_offsets = sampling_offsets.view(
                N, Len_q, n_heads, n_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)


            sampling_locations = reference_points + sampling_offsets
            N, Len_q,n_heads,n_levels, n_points,num_Z_anchors,xy = sampling_locations.shape
            
            
            assert num_all_points == n_points*num_Z_anchors
            
            sampling_locations = sampling_locations.view(N,Len_q,n_heads,n_levels,num_all_points,xy)
            
            ## commented by me: this is original Deformable attention
            # sampling_locations = reference_points[:, :, None, :, None, :] \
            #                      + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            ## commented by me: this is original Deformable attention
            # sampling_locations = reference_points[:, :, None, :, None, :2] \
            #                      + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            assert False
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        #print(value.dtype, input_spatial_shapes.type,input_level_start_index.shape,sampling_locations.type,attention_weights.type,type(self.im2col_step))
        # output = MultiScaleDeformableAttnFunction_fp32.apply(
        #     value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)

        output = ms_deform_attn_core_pytorch(value,input_spatial_shapes,sampling_locations,attention_weights)
        
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        #print(output.shape)    
        #output = self.output_proj(output) ##TODO: BEVFormer does not have this
        return output
