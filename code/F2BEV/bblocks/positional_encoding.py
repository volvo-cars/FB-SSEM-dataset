#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:32:38 2022

@author: Ekta
"""


import torch
import torch.nn as nn


### from: https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/models/utils/positional_encoding.html

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)),
            dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos
    

# import torch
# import torch.nn as nn
# from typing import Optional
# from torch import Tensor

# ### https://github.com/fundamentalvision/Deformable-DETR/blob/11169a60c33333af00a4849f1808023eba96a931/models/position_encoding.py

# class NestedTensor(object):
#     def __init__(self, tensors, mask: Optional[Tensor]):
#         self.tensors = tensors
#         self.mask = mask

#     def to(self, device, non_blocking=False):
#         # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device, non_blocking=non_blocking)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)

# class PositionEmbeddingLearned(nn.Module):
#     """
#     Absolute pos embedding, learned.
#     """
#     def __init__(self, num_pos_feats=256):
#         super().__init__()
#         self.row_embed = nn.Embedding(50, num_pos_feats)
#         self.col_embed = nn.Embedding(50, num_pos_feats)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.uniform_(self.row_embed.weight)
#         nn.init.uniform_(self.col_embed.weight)

#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         h, w = x.shape[-2:]
#         i = torch.arange(w, device=x.device)
#         j = torch.arange(h, device=x.device)
#         x_emb = self.col_embed(i)
#         y_emb = self.row_embed(j)
#         pos = torch.cat([
#             x_emb.unsqueeze(0).repeat(h, 1, 1),
#             y_emb.unsqueeze(1).repeat(1, w, 1),
#         ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
#         return pos