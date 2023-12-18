#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:19:10 2023

@author: Ekta
"""


import torch 
import torch.nn as nn
from bblocks.mask_head_pansegformer import MaskHead

class MaskHeadDecoderSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_h = 50
        self.bev_w = 50
        self.num_stuff_classes = 5
        self.embed_dims = 256
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.stuff_mask_head = MaskHead(num_decoder_layers=3,self_attn=True)

 
        # self._reset_parameters()

    # def _reset_parameters(self):


    def forward(self,bev_embed):
        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight,self.embed_dims,dim=1)
        bs = bev_embed.shape[0]
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(bs, -1, -1)
        stuff_query = stuff_query.unsqueeze(0).expand(bs, -1, -1)
        hw_lvl = torch.tensor([[self.bev_h, self.bev_w]], device=stuff_query.device)
        
        attn, masks, inter_query = self.stuff_mask_head(bev_embed,None,None,stuff_query, None, stuff_query_pos, 
                                                        hw_lvl)
        
        mask_stuff = attn.squeeze(-1)
        
        mask_stuff = mask_stuff.reshape(bs, self.num_stuff_classes, hw_lvl[0][0],hw_lvl[0][1])

        inter_masks = [m.squeeze(-1).reshape(bs, self.num_stuff_classes, hw_lvl[0][0],hw_lvl[0][1]) for m in masks]

        return mask_stuff, inter_masks


class MaskHeadDecoderHt(nn.Module):
    def __init__(self):
        super().__init__()
        self.bev_h = 50
        self.bev_w = 50
        self.num_stuff_classes = 3
        self.embed_dims = 256
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)
        self.stuff_mask_head = MaskHead(num_decoder_layers=3,self_attn=True)

 
        # self._reset_parameters()

    # def _reset_parameters(self):


    def forward(self,bev_embed):
        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight,self.embed_dims,dim=1)
        bs = bev_embed.shape[0]
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(bs, -1, -1)
        stuff_query = stuff_query.unsqueeze(0).expand(bs, -1, -1)
        hw_lvl = torch.tensor([[self.bev_h, self.bev_w]], device=stuff_query.device)
        
        attn, masks, inter_query = self.stuff_mask_head(bev_embed,None,None,stuff_query, None, stuff_query_pos, 
                                                        hw_lvl)
        
        mask_stuff = attn.squeeze(-1)
        
        mask_stuff = mask_stuff.reshape(bs, self.num_stuff_classes, hw_lvl[0][0],hw_lvl[0][1])

        inter_masks = [m.squeeze(-1).reshape(bs, self.num_stuff_classes, hw_lvl[0][0],hw_lvl[0][1]) for m in masks]

        return mask_stuff, inter_masks
