#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:45:39 2022

@author: Ekta
"""

import torch
import torch.nn as nn
import numpy as np

from bblocks.positional_encoding import LearnedPositionalEncoding
from bblocks.bevformer_block import BEVFormerBlock 
from torchvision.transforms.functional import rotate

from torch.nn.init import xavier_uniform_, constant_

class EncoderFLCW(nn.Module):
    def __init__(self,bev_h=50,bev_w=50,num_feature_levels = 4,num_cams=4,embed_dims=256):
        super(EncoderFLCW, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.final_reference_points_cam = nn.Parameter(torch.from_numpy(np.load('./pre_computation/unity_data/reference_points_cam.npy')).float(),requires_grad=False)
        self.final_bev_mask = nn.Parameter(torch.from_numpy(np.load('./pre_computation/unity_data/bev_mask.npy')).float(),requires_grad = False)
        self.positional_encoding = LearnedPositionalEncoding(num_feats=128, row_num_embed=50, col_num_embed=50)
        self.bevformer_encoder_block = BEVFormerBlock()
        self.use_cams_embeds = True 
        #self.query_embedding = nn.Embedding(self.num_query,self.embed_dims * 2)
        self.grid_length = [2/48,2/48] ##i got this wrong before 42b; 2m is 48 pixels
        self.use_can_bus = True
        self.use_shift = True
        self.can_bus_norm = True
        self.rotate_prev_bev = True
        self.init_layers()
        #self.rotate_center = [100,100] ##TODO: what is this? I want to use default i.e. center so i removed this
        
    def init_layers(self):
        
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(5, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)) 
        
        xavier_uniform_(self.level_embeds.data)
        xavier_uniform_(self.cams_embeds.data)
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))        
        
        

    def get_bev_features(self,mlvl_feats,bev_queries,bev_pos,can_buses,prev_bev=None):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

     
        ##todo: ADD shift here, prev bev here
        

        # obtain rotation angle and shift with ego motion
        # delta_x = np.array([each['can_bus'][0]
        #                    for each in can_buses])
        # delta_y = np.array([each['can_bus'][1]
        #                    for each in can_buses])
        # ego_angle = np.array(
        #     [each['can_bus'][-2] / np.pi * 180 for each in can_buses])
        delta_x = np.array([each[0]
                           for each in can_buses])
        delta_y = np.array([each[1]
                           for each in can_buses])
        ego_angle = np.array(
            [each[-2] for each in can_buses])
        
        #print('Delta x: ', delta_x)
        #print('Delta y: ', delta_y)
        #print('Ego angle: ', ego_angle)
        grid_length_y = self.grid_length[0]
        grid_length_x = self.grid_length[1]

        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        #print(translation_angle)
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    #rotation_angle = can_buses[i]['can_bus'][-1]
                    rotation_angle = can_buses[i][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, float(rotation_angle.numpy()))#,
                                          #center=self.rotate_center)  ##I am removing center
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        self.bev_h * self.bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        #print(can_buses)
        can_bus = bev_queries.new_tensor(
            [each.numpy() for each in can_buses])  # [:, :]
        # can_bus = bev_queries.new_tensor(
        #     [each['can_bus'] for each in can_buses])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus
        
        #print(torch.unique(self.level_embeds))
        
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            #print(feat.shape)
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) 
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype).to(feat.device)
                #print(self.cams_embeds[:, None, None, :].shape)
                #print(feat.shape)
                #print(torch.unique(self.cams_embeds[:, None, None, :].to(feat.dtype).cpu()))
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype).to(feat.device)
            
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device = bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #print(spatial_shapes)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        #print(feat_flatten.shape)
        return feat_flatten,spatial_shapes ,bev_queries,bev_pos,level_start_index,shift,prev_bev
    
    def get_2d_ref_points_for_tsa(self,bs,dtype,device):
        H = self.bev_h
        W = self.bev_w
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d
            
       
    def forward(self,mlvl_feats,can_buses,prev_bev=None):
        
        ##this is forward of BEVformerhead in bevformer
        dtype = mlvl_feats[0].dtype
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask_temp = torch.zeros((bs, self.bev_h, self.bev_w),
                                       device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask_temp).to(dtype)
        
        ###now get_bev_features
        feat_flatten,spatial_shapes,bev_queries,bev_pos,level_start_index,shift,prev_bev = self.get_bev_features(mlvl_feats,bev_queries,bev_pos,can_buses,prev_bev)
        
        
        ##this is just some part of encoder py...rest in BEVformerBlock
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        
        ref_2d = self.get_2d_ref_points_for_tsa(bs,bev_queries.dtype, bev_queries.device)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]
    
        
        bev_queries = bev_queries.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            #print(prev_bev.shape)
            prev_bev = torch.stack(
                [prev_bev, bev_queries], 1).reshape(bs*2, len_bev, -1)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
            
        #print(hybrid_ref_2d.shape)
        ### todo add some more shift here; 2d reference business
        #print(bev_queries.shape)
        output = self.bevformer_encoder_block(bev_query = bev_queries, key = feat_flatten, value = feat_flatten,
                                              bev_pos = bev_pos,spatial_shapes = spatial_shapes ,level_start_index = level_start_index,
                                              reference_points_cam = self.final_reference_points_cam, bev_mask = self.final_bev_mask, 
                                              ref_2d= hybrid_ref_2d,prev_bev=prev_bev)
        
        

        output = self.bevformer_encoder_block(bev_query = output, key = feat_flatten, value = feat_flatten,
                                              bev_pos = bev_pos,spatial_shapes = spatial_shapes ,level_start_index = level_start_index,
                                              reference_points_cam = self.final_reference_points_cam, bev_mask = self.final_bev_mask, 
                                              ref_2d= hybrid_ref_2d,prev_bev=prev_bev)
        
        output = self.bevformer_encoder_block(bev_query = output, key = feat_flatten, value = feat_flatten,
                                              bev_pos = bev_pos,spatial_shapes = spatial_shapes ,level_start_index = level_start_index,
                                              reference_points_cam = self.final_reference_points_cam, bev_mask = self.final_bev_mask, 
                                              ref_2d= hybrid_ref_2d,prev_bev=prev_bev)

        return output
        
        
        
        

    
class Encoder(nn.Module):
    def __init__(self,bev_h=50,bev_w=50,num_feature_levels = 4,num_cams=4,embed_dims=256):
        super(EncoderFLCW, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.final_reference_points_cam = nn.Parameter(torch.from_numpy(np.load('./pre_computation/for_height_map/reference_points_cam.npy')).float(),requires_grad=False)
        self.final_bev_mask = nn.Parameter(torch.from_numpy(np.load('./pre_computation/for_height_map/bev_mask.npy')).float(),requires_grad = False)
        self.positional_encoding = LearnedPositionalEncoding(num_feats=128, row_num_embed=50, col_num_embed=50)
        self.bevformer_encoder_block = BEVFormerBlock()
        self.use_cams_embeds = True 
        #self.query_embedding = nn.Embedding(self.num_query,self.embed_dims * 2)
        self.grid_length = [2/48,2/48] ##i got this wrong before 42b; 2m is 48 pixels
        self.use_can_bus = True
        self.use_shift = True
        self.can_bus_norm = True
        self.rotate_prev_bev = True
        self.init_layers()
        #self.rotate_center = [100,100] ##TODO: what is this? I want to use default i.e. center so i removed this

    def init_layers(self):

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(5, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)) 

        xavier_uniform_(self.level_embeds.data)
        xavier_uniform_(self.cams_embeds.data)
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))        



    def get_bev_features(self,mlvl_feats,bev_queries,bev_pos,can_buses,prev_bev=None):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)


        ##todo: ADD shift here, prev bev here
        # obtain rotation angle and shift with ego motion
        # delta_x = np.array([each['can_bus'][0]
        #                    for each in can_buses])
        # delta_y = np.array([each['can_bus'][1]
        #                    for each in can_buses])
        # ego_angle = np.array(
        #     [each['can_bus'][-2] / np.pi * 180 for each in can_buses])
        delta_x = np.array([each[0]
                           for each in can_buses])
        delta_y = np.array([each[1]
                           for each in can_buses])
        ego_angle = np.array(
            [each[-2] for each in can_buses])

        #print('Delta x: ', delta_x)
        #print('Delta y: ', delta_y)
        #print('Ego angle: ', ego_angle)
        grid_length_y = self.grid_length[0]
        grid_length_x = self.grid_length[1]

        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)

        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        #print(translation_angle)
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / self.bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / self.bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == self.bev_h * self.bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    #rotation_angle = can_buses[i]['can_bus'][-1]
                    rotation_angle = can_buses[i][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        self.bev_h, self.bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, float(rotation_angle.numpy()))#,
                                          #center=self.rotate_center)  ##I am removing center
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        self.bev_h * self.bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        #print(can_buses)
        can_bus = bev_queries.new_tensor(
            [each.numpy() for each in can_buses])  # [:, :]
        # can_bus = bev_queries.new_tensor(
        #     [each['can_bus'] for each in can_buses])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus
        
        #print(torch.unique(self.level_embeds))
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            #print(feat.shape)
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) 
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype).to(feat.device)
                #print(self.cams_embeds[:, None, None, :].shape)
                #print(feat.shape)
                #print(torch.unique(self.cams_embeds[:, None, None, :].to(feat.dtype).cpu()))
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype).to(feat.device)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device = bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #print(spatial_shapes)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        #print(feat_flatten.shape)
        return feat_flatten,spatial_shapes ,bev_queries,bev_pos,level_start_index,shift,prev_bev
    
    def get_2d_ref_points_for_tsa(self,bs,dtype,device):
        H = self.bev_h
        W = self.bev_w
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d


    def forward(self,mlvl_feats,can_buses,prev_bev=None):

        ##this is forward of BEVformerhead in bevformer
        dtype = mlvl_feats[0].dtype
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask_temp = torch.zeros((bs, self.bev_h, self.bev_w),
                                       device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask_temp).to(dtype)

        ###now get_bev_features
        feat_flatten,spatial_shapes,bev_queries,bev_pos,level_start_index,shift,prev_bev = self.get_bev_features(mlvl_feats,bev_queries,bev_pos,can_buses,prev_bev)


        ##this is just some part of encoder py...rest in BEVformerBlock
        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)

        ref_2d = self.get_2d_ref_points_for_tsa(bs,bev_queries.dtype, bev_queries.device)
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]
    

        bev_queries = bev_queries.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)

        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            #print(prev_bev.shape)
            prev_bev = torch.stack(
                [prev_bev, bev_queries], 1).reshape(bs*2, len_bev, -1)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        #print(hybrid_ref_2d.shape)
        ### todo add some more shift here; 2d reference business
        #print(bev_queries.shape)
        output = self.bevformer_encoder_block(bev_query = bev_queries, key = feat_flatten, value = feat_flatten,
                                              bev_pos = bev_pos,spatial_shapes = spatial_shapes ,level_start_index = level_start_index,
                                              reference_points_cam = self.final_reference_points_cam, bev_mask = self.final_bev_mask, 
                                              ref_2d= hybrid_ref_2d,prev_bev=prev_bev)



        output = self.bevformer_encoder_block(bev_query = output, key = feat_flatten, value = feat_flatten,
                                              bev_pos = bev_pos,spatial_shapes = spatial_shapes ,level_start_index = level_start_index,
                                              reference_points_cam = self.final_reference_points_cam, bev_mask = self.final_bev_mask, 
                                              ref_2d= hybrid_ref_2d,prev_bev=prev_bev)

        output = self.bevformer_encoder_block(bev_query = output, key = feat_flatten, value = feat_flatten,
                                              bev_pos = bev_pos,spatial_shapes = spatial_shapes ,level_start_index = level_start_index,
                                              reference_points_cam = self.final_reference_points_cam, bev_mask = self.final_bev_mask, 
                                              ref_2d= hybrid_ref_2d,prev_bev=prev_bev)

        return output
