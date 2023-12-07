#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 10:09:43 2023

@author: Ekta
"""

import torch
from torch.utils.data import Dataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor
#import matplotlib.pyplot as plt
#from torch.utils.data import DataLoader
import numpy as np
import os#,fnmatch
from torchvision.io import read_image
import random
class UnityImageDataset(Dataset):
    def __init__(self, bev_dirs, bev_depth_dirs, front_dirs, left_dirs, rear_dirs, right_dirs, image_lists, config_dirs, seq_len, datalengths, num_data_sequences, transform=None, target_transform=None):
        self.bev_dirs = bev_dirs
        self.bev_depth_dirs =  bev_depth_dirs 
        self.front_dirs = front_dirs
        self.left_dirs = left_dirs
        self.rear_dirs = rear_dirs
        self.right_dirs = right_dirs
        self.image_lists = image_lists
        self.config_dirs = config_dirs
        self.transform = transform
        self.target_transform = target_transform
        self.seq_len = seq_len
        self.datalengths = datalengths
        self.num_data_sequences = num_data_sequences

    def __len__(self):
        total = 0
        for count in self.datalengths:
            total = total + count
        return total
    
    def find_which_sequence(self,idx):

        eff_data_lens = [x for x in self.datalengths]

        
        currptr = 0
        nextptr =  eff_data_lens[0]
        
        for i in range(self.num_data_sequences):
            if i == 0:
                currptr = 0
                nextptr =  eff_data_lens[0]
                
                if idx > currptr -1 and idx < nextptr:
                    seq_idx = 0
            else:
                currptr = sum(eff_data_lens[:i])
                nextptr = sum(eff_data_lens[:i+1])
                if idx > currptr -1 and idx < nextptr:
                    seq_idx = i
                    
        

        return seq_idx
        
    def get_id_in_seq(self,seq_idx,idx):
        eff_data_lens = [x for x in self.datalengths]


        if seq_idx == 0:
            subtract = 0
        else:
            subtract = sum(eff_data_lens[:seq_idx])
        return idx - subtract
        
    def read_config_for_bevposrot(self,configdir,filename):
        with open(os.path.join(configdir, filename)) as f:
            lines = f.readlines()
            
        bpos = [float(lines[5].split(',')[1]),float(lines[5].split(',')[2]),float(lines[5].split(',')[3])]
        brot = [float(lines[5].split(',')[4]),float(lines[5].split(',')[5]),float(lines[5].split(',')[6])]
        return [bpos,brot]
    
    
    def __getitem__(self, idx):
        
        seq_idx = self.find_which_sequence(idx)
        
        bev_dir = self.bev_dirs[seq_idx]
        bev_depth_dir = self.bev_depth_dirs[seq_idx]
        image_list = self.image_lists[seq_idx]
        front_dir = self.front_dirs[seq_idx]
        left_dir = self.left_dirs[seq_idx]
        rear_dir = self.rear_dirs[seq_idx]
        right_dir = self.right_dirs[seq_idx]
        config_dir = self.config_dirs[seq_idx]
        
        idinseq = self.get_id_in_seq(seq_idx,idx)
        
        return_images_tensor = []
        return_starget = []
        return_htarget = []
        return_can_bus = []
        ##first image
        
        index_list = list(range(idinseq-self.seq_len, idinseq))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(idinseq)
        
        for idxctr,cidx in enumerate(index_list):
            cidx = max(0, cidx)

            star_path = os.path.join(bev_dir, image_list[cidx])
            star = read_image(star_path)[:,100:~99,100:~99]
            star = torch.mul(star.float(),1/255)

            htar_path = os.path.join(bev_depth_dir, image_list[cidx])
            htar = read_image(htar_path)[:,100:~99,100:~99]
            htar = torch.mul(htar.float(),1/255)
            if self.target_transform:
                star = self.target_transform(star)
                htar = self.target_transform(htar)
            inp = []
            for cam_views in [front_dir, left_dir, rear_dir, right_dir]:
                img_path = os.path.join(cam_views, image_list[cidx])
                image = read_image(img_path)
                image = torch.mul(image.float(),1/255)
                # if self.transform:
                #     image = self.transform(image)
                inp.append(image)
                
            [bpos,brot] = self.read_config_for_bevposrot(config_dir,image_list[cidx].split('.')[0]+'.txt')
            

            can_bus = np.zeros((5,))
            if cidx == 0:
                #pos
                can_bus[0] = 0
                can_bus[1] = 0
                can_bus[2] = 0
                #angle
                can_bus[3] = brot[1] #(90 - bevrot[num,1])/180*np.pi ##ego_angle is kept unchanged .. i.e. no delta ##before that 270 -
                can_bus[4] = 0
                
            else:
                [prev_bpos,prev_brot] = self.read_config_for_bevposrot(config_dir,image_list[cidx-1].split('.')[0]+'.txt')
                
                can_bus[0] = bpos[0] - prev_bpos[0]
                can_bus[1] = bpos[2] - prev_bpos[2]
                can_bus[2] = bpos[1] - prev_bpos[1]
                can_bus[3] = brot[1]
                
                can_bus[4] =  brot[1]  - prev_brot[1]

                            
            return_images_tensor.append(torch.stack(inp))
            return_starget.append(star)
            return_htarget.append(htar)
            return_can_bus.append(can_bus)
        
        
        if self.transform:
            return_images_tensor = self.transform(torch.cat(return_images_tensor, dim=0))

        #return_images = [list(torch.split(x, 4)) for x in list(torch.split(return_images, self.seq_len))] #because 4 camera views
        return_images = []
        for frameidx in range(self.seq_len):
            inp = []
            for camnum in range(4): #4 cam views
                inp.append(return_images_tensor[(4*frameidx) + camnum, :,:,:])
            return_images.append(inp)
            
        #return torch.cat((inp[0],inp[1],inp[2],inp[3]),axis=0), tar
        return [return_images,return_can_bus], [return_starget,return_htarget]
