#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:12:46 2022

@author: Ekta
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

#import MultiScaleDeformableAttention as MSDA ##TODO: Installation for this comes from DETR repo -- need to build sth to get this

# class MSDeformAttnFunction(Function):
#     @staticmethod
#     @custom_fwd(cast_inputs=torch.float16)
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
#         """GPU version of multi-scale deformable attention.
#         Args:
#             value (Tensor): The value has shape
#                 (bs, num_keys, mum_heads, embed_dims//num_heads)
#             value_spatial_shapes (Tensor): Spatial shape of
#                 each feature map, has shape (num_levels, 2),
#                 last dimension 2 represent (h, w)
#             sampling_locations (Tensor): The location of sampling points,
#                 has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points, 2),
#                 the last dimension 2 represent (x, y).
#             attention_weights (Tensor): The weight of sampling points used
#                 when calculate the attention, has shape
#                 (bs ,num_queries, num_heads, num_levels, num_points),
#             im2col_step (Tensor): The step used in image to column.
#         Returns:
#             Tensor: has shape (N, Len_q, d_model)
#         """
        
#         ctx.im2col_step = im2col_step
#         #print(type(value),type(value_spatial_shapes),type(value_level_start_index),type(sampling_locations),type(attention_weights),type(ctx.im2col_step))

#         output = MSDA.ms_deform_attn_forward(
#             value = value, value_spatial_shapes=value_spatial_shapes, value_level_start_index=value_level_start_index, sampling_locations=sampling_locations, attention_weights=attention_weights, im2col_step = ctx.im2col_step)
#         ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
#         return output

#     @staticmethod
#     @once_differentiable
#     @custom_bwd
#     def backward(ctx, grad_output):
#         """GPU version of backward function.
#         Args:
#             grad_output (Tensor): Gradient
#                 of output tensor of forward.
#         Returns:
#              Tuple[Tensor]: Gradient
#                 of input tensors in forward.
#         """
#         value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
#         grad_value = torch.zeros_like(value)
#         grad_sampling_loc = torch.zeros_like(sampling_locations)
#         grad_attn_weight = torch.zeros_like(attention_weights)
        
#         MSDA.ms_deform_attn_backward(value,value_spatial_shapes,value_level_start_index,sampling_locations,attention_weights,grad_output.contiguous(),grad_value,grad_sampling_loc,grad_attn_weight,im2col_step=ctx.im2col_step)

#         return grad_value, None, None, \
#             grad_sampling_loc, grad_attn_weight, None
            

def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    #print(value.shape)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
