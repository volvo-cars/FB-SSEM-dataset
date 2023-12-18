#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:47:19 2022

@author: Ekta
"""
import torch,time,ntpath #cv2
from torch import nn, optim
import numpy as np
from test_loader_single_task import UnityImageDataset
import os,fnmatch
from torch.utils.data import DataLoader
from model_f2bev_conv_st_height import FisheyeBEVFormer
import torchvision.transforms as T
from torchmetrics.functional import jaccard_index


def numpy_sigmoid(x):
    return 1/(1 + np.exp(-x))

if not os.path.exists('./predictions/'):
   os.makedirs('./predictions/')

if not os.path.exists('./predictions/f2bev_conv_st_height/'):
   os.makedirs('./predictions/f2bev_conv_st_height/')
if not os.path.exists('./predictions/f2bev_conv_st_height/bevfeatures'):
   os.makedirs('./predictions/f2bev_conv_st_height/features')
if not os.path.exists('./predictions/f2bev_conv_st_height/predfull/'):
   os.makedirs('./predictions/f2bev_conv_st_height/predfull/')
if not os.path.exists('./predictions/f2bev_conv_st_height/predfull/ce/'):
   os.makedirs('./predictions/f2bev_conv_st_height/predfull/ce/')

num_data_sequences = 20


bev_dirs = ['./data/images'+str(i)+'/test/depth' for i in range(num_data_sequences)]
front_dirs = ['./data/images'+str(i)+'/test/rgb/front' for i in range(num_data_sequences)]
left_dirs = ['./data/images'+str(i)+'/test/rgb/left' for i in range(num_data_sequences)]
rear_dirs = ['./data/images'+str(i)+'/test/rgb/rear' for i in range(num_data_sequences)]
right_dirs = ['./data/images'+str(i)+'/test/rgb/right' for i in range(num_data_sequences)]
config_dirs = ['./data/images'+str(i)+'/test/cameraconfig' for i in range(num_data_sequences)]


seq_len = 1

image_lists = []

datalengths = []

for bev_dir in bev_dirs:
    names = fnmatch.filter(os.listdir(bev_dir), '*.png')
    
    files = []
    for name in names:
        files.append(os.path.splitext(ntpath.basename(name))[0])

        filelist = sorted(files,key=int)

    image_lists.append([f + '.png' for f in filelist])
    datalengths.append(len(names))



transforms = torch.nn.Sequential(T.Resize((540,640)),)
target_transforms = torch.nn.Sequential(T.Grayscale(num_output_channels=1))

test_data = UnityImageDataset(bev_dirs = bev_dirs,front_dirs=front_dirs,left_dirs=left_dirs,rear_dirs=rear_dirs,right_dirs=right_dirs,image_lists=image_lists,config_dirs=config_dirs,seq_len= seq_len,datalengths = datalengths,num_data_sequences = num_data_sequences, 
                              transform = transforms, target_transform= target_transforms)


test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"
print(f"Using {device} device")

model = FisheyeBEVFormer().to(device)

checkpoint = torch.load('./f2bev_conv_st_height.pt')
model.load_state_dict(checkpoint['model_state_dict'])



def test_temporal(test_dataloader,seq_len,model,loss_fn,image_list):
    num_batches = len(test_dataloader)
    model.eval()
    test_loss = 0
    test_iou = 0
    with torch.no_grad():
        for batch_idx, (dataseq, targetseq) in enumerate(test_dataloader):
            inp_img_seq, can_buses_seq = dataseq
            bs = targetseq[0].shape[0]
            for ctr in range(seq_len):
                front = inp_img_seq[ctr][0]
                left = inp_img_seq[ctr][1]
                rear = inp_img_seq[ctr][2]
                right = inp_img_seq[ctr][3]



                target = targetseq[ctr]
                front = front.to(device)
                left = left.to(device)
                rear = rear.to(device)
                right = right.to(device)

                target = torch.squeeze(target,dim=1)
                idx2 = torch.where(target <= 0.35)
                idx0 = torch.where(target >= 0.69)
                target[target >= 0] = 1
                target[idx2] = 2
                target[idx0] = 0

                target = target.to(torch.int64).to(device)
                can_buses = can_buses_seq[ctr]


                if batch_idx == 0:
                    prev_bev = None

                else:
                    prev_bev = torch.Tensor([np.load('./predictions/f2bev_conv_st_height/bevfeatures/'+image_list[batch_idx-1].split('.')[0]+'.npy')]).to(device)
                pred, for_prev_bev = model(front,left,rear,right, list(can_buses),prev_bev)

                for i,p in enumerate(for_prev_bev):
                    np.save('./predictions/f2bev_conv_st_height/bevfeatures/'+image_list[batch_idx].split('.')[0]+'.npy',p.detach().cpu().numpy())
                for i,p in enumerate(pred):
                    np.save('./predictions/f2bev_conv_st_height/predfull/focal/'+image_list[batch_idx].split('.')[0]+'.npy',p.detach().cpu().numpy())
                test_loss += loss_fn(pred, target).item()
                test_iou += jaccard_index(pred,target.to(pred.device),num_classes=3,average='none')

    test_loss/= num_batches*seq_len
    test_iou /= num_batches    
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")    
    print(test_iou)
    return test_loss

all_images = []
for test_list in image_lists:
    all_images = all_images + test_list
    
loss = nn.CrossEntropyLoss() #nn.MSELoss() #BinaryFocalLoss(alpha=0.25,gamma=2)
test_temporal(test_dataloader,seq_len,model,loss,all_images)
