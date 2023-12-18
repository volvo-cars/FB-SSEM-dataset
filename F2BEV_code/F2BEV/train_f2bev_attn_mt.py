#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 12:15:50 2023

@author: Ekta
"""


import torch,ntpath,random
from torch import nn, optim
import numpy as np
from loader_multi_task import UnityImageDataset


import os,fnmatch
from torch.utils.data import DataLoader
from model_f2bev_attn_mt import FisheyeBEVFormer

import torchvision.transforms as T
from losses.focal import BinaryFocalLoss, CrossEntropyFocalLoss #FocalLoss
from losses.smoothness import MonodepthLoss

def numpy_sigmoid(x):
    return 1/(1 + np.exp(-x))



def gamma_correction(image):
    gamma = random.choice([0.8,1.0,1.2,1.4])
    return T.functional.adjust_gamma(image,gamma,gain = 1)

def plt_pred_image(pred):
    p = pred.detach().cpu().numpy()
    one = numpy_sigmoid(p[1,:,:])
    two = numpy_sigmoid(p[2,:,:])
    
    show = np.zeros((p.shape[1],p.shape[2]))
    show[np.where(one > 0.5)] = 1
    show[np.where(two > 0.5)] = 2
    
    return show
    

    
num_data_sequences = 20

bev_dirs = ['./data/images'+str(i)+'/train/seg/bev' for i in range(num_data_sequences)]
bev_depth_dirs = ['./data/images'+str(i)+'/train/depth' for i in range(num_data_sequences)]
front_dirs = ['./data/images'+str(i)+'/train/rgb/front' for i in range(num_data_sequences)]
left_dirs = ['./data/images'+str(i)+'/train/rgb/left' for i in range(num_data_sequences)]
rear_dirs = ['./data/images'+str(i)+'/train/rgb/rear' for i in range(num_data_sequences)]
right_dirs = ['./data/images'+str(i)+'/train/rgb/right' for i in range(num_data_sequences)]
config_dirs = ['./data/images'+str(i)+'/train/cameraconfig' for i in range(num_data_sequences)]
    
seq_len = 3

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


vtransforms = T.Compose([T.Resize((540,640))])   #,GammaCorrectionTransform())

transforms = T.Compose([T.Resize((540,640)), T.ColorJitter(brightness=.3, hue=.3, contrast = 0.2, saturation = 0.3),  T.Lambda(gamma_correction) ])   #,GammaCorrectionTransform())
target_transforms = T.Compose([T.Resize((50,50)),T.Grayscale(num_output_channels=1)])
 
   
training_data = UnityImageDataset(bev_dirs = bev_dirs,bev_depth_dirs = bev_depth_dirs,front_dirs=front_dirs,left_dirs=left_dirs,rear_dirs=rear_dirs,right_dirs=right_dirs,image_lists=image_lists,config_dirs=config_dirs,seq_len= seq_len,datalengths = datalengths,num_data_sequences = num_data_sequences, transform = transforms,target_transform=target_transforms)


vbev_dirs = ['./data/images'+str(i)+'/val/seg/bev' for i in range(num_data_sequences)]
vbev_depth_dirs = ['./data/images'+str(i)+'/val/depth' for i in range(num_data_sequences)]
vfront_dirs = ['./data/images'+str(i)+'/val/rgb/front' for i in range(num_data_sequences)]
vleft_dirs = ['./data/images'+str(i)+'/val/rgb/left' for i in range(num_data_sequences)]
vrear_dirs = ['./data/images'+str(i)+'/val/rgb/rear' for i in range(num_data_sequences)]
vright_dirs = ['./data/images'+str(i)+'/val/rgb/right' for i in range(num_data_sequences)]
vconfig_dirs = ['./data/images'+str(i)+'/val/cameraconfig' for i in range(num_data_sequences)]

vimage_lists = []
vdatalengths = []

for vbev_dir in vbev_dirs:
    vnames = fnmatch.filter(os.listdir(vbev_dir), '*.png')
    
    vfiles = []
    for vname in vnames:
        vfiles.append(os.path.splitext(ntpath.basename(vname))[0])
        
        vfilelist = sorted(vfiles,key=int)
        
    vimage_lists.append([f + '.png' for f in vfilelist])
    vdatalengths.append(len(vnames))

val_data = UnityImageDataset(bev_dirs = vbev_dirs,bev_depth_dirs=vbev_depth_dirs,front_dirs=vfront_dirs,left_dirs=vleft_dirs,rear_dirs=vrear_dirs,right_dirs=vright_dirs,image_lists=vimage_lists,config_dirs=vconfig_dirs,seq_len= seq_len,datalengths = vdatalengths, num_data_sequences = num_data_sequences, transform = vtransforms,target_transform=target_transforms)


train_dataloader = DataLoader(training_data, batch_size = 2, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=2, shuffle=False)



random_seed = 1 # or any of your favorite number
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



        
        
def train(epoch,model,train_dataloader,seq_len, criterion, optimizer):

    model.train()

    train_loss = 0
    num_batches = len(train_dataloader)
    # with torch.autograd.detect_anomaly():
    for batch_idx, (dataseq, targetseq) in enumerate(train_dataloader):
        
        inp_img_seq, can_buses_seq = dataseq
        stargetseq,htargetseq = targetseq
        bs = stargetseq[0].shape[0]
        for ctr in range(seq_len):
            front = inp_img_seq[ctr][0]
            left = inp_img_seq[ctr][1]
            rear = inp_img_seq[ctr][2]
            right = inp_img_seq[ctr][3]
            
            starget = stargetseq[ctr]
            htarget = htargetseq[ctr]

            front = front.to(device)
            left = left.to(device)
            rear = rear.to(device)
            right = right.to(device)
            


            starget = torch.squeeze(starget,dim=1)
            idx0 = torch.where(starget <= 0.02)
            starget[idx0] = 10
            idx1 = torch.where(starget <= 0.07)
            starget[idx1] = 11
            idx2 = torch.where(starget <= 0.22)
            starget[idx2] = 12
            idx3 = torch.where(starget <= 0.60)
            starget[idx3] = 13
            idx4 = torch.where(starget <= 1)
            starget[idx4] = 14
            starget = starget - 10
            starget = starget.to(torch.int64).to(device)
            htarget = torch.squeeze(htarget,dim=1)
            idx2 = torch.where(htarget <= 0.35)
            idx0 = torch.where(htarget >= 0.69)
            htarget[htarget >= 0] = 1
            htarget[idx2] = 2
            htarget[idx0] = 0
            htarget = htarget.to(torch.int64).to(device)

            can_buses = can_buses_seq[ctr]

            if ctr == 0:
                prev_bev = None
                
            optimizer.zero_grad()
            soutput, s_inter_outputs, houtput, h_inter_outputs, prev_bev_org = model(front,left,rear,right, list(can_buses),prev_bev)
            
                
            prev_bev = prev_bev_org.detach()
            
            sloss = criterion(soutput, starget)
            hloss = criterion(houtput, htarget)

            for s_inter_stage in s_inter_outputs:
                sloss += criterion(s_inter_stage,starget)

            for h_inter_stage in h_inter_outputs:
                hloss += criterion(h_inter_stage,htarget)
                
            loss = sloss + hloss 
    
            loss.backward()
            optimizer.step()
            
            train_loss += loss.data
            print('Train Epoch: {} [{}/{} Srno: {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * 1, len(train_dataloader), ctr,
                    100. * (batch_idx + 1) / len(train_dataloader), loss.data))
            
    train_loss/= num_batches*seq_len
    
    print(f"Train Error: Avg loss: {train_loss:>8f} \n")    

    return train_loss    


def val(epoch,test_dataloader,seq_len,model,criterion):
    num_batches = len(test_dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (dataseq, targetseq) in enumerate(test_dataloader):
            inp_img_seq, can_buses_seq = dataseq
            stargetseq,htargetseq = targetseq
            for ctr in range(seq_len):
                front = inp_img_seq[ctr][0]
                left = inp_img_seq[ctr][1]
                rear = inp_img_seq[ctr][2]
                right = inp_img_seq[ctr][3]
                
                starget = stargetseq[ctr]
                htarget = htargetseq[ctr]

                front = front.to(device)
                left = left.to(device)
                rear = rear.to(device)
                right = right.to(device)
                starget = torch.squeeze(starget,dim=1)
                idx0 = torch.where(starget <= 0.02)
                starget[idx0] = 10
                idx1 = torch.where(starget <= 0.07)
                starget[idx1] = 11
                idx2 = torch.where(starget <= 0.22)
                starget[idx2] = 12
                idx3 = torch.where(starget <= 0.60)
                starget[idx3] = 13
                idx4 = torch.where(starget <=1)
                starget[idx4] = 14
                starget = starget - 10
                starget = starget.to(torch.int64).to(device)

                            
                htarget = torch.squeeze(htarget,dim=1)
                idx2 = torch.where(htarget <= 0.35)
                idx0 = torch.where(htarget >= 0.69)
                htarget[htarget >= 0] = 1
                htarget[idx2] = 2
                htarget[idx0] = 0
                htarget = htarget.to(torch.int64).to(device)

                can_buses = can_buses_seq[ctr]

                if ctr == 0:
                    prev_bev = None
                spred, _, hpred, _, prev_bev = model(front,left,rear,right, list(can_buses),prev_bev)
                
                test_loss = test_loss + criterion(spred, starget).item() + criterion(hpred, htarget).item()  
                
        test_loss/= num_batches*seq_len
    
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")    

    return test_loss


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = FisheyeBEVFormer().to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0002)
criterionCE = nn.CrossEntropyLoss()
criterionFocal = CrossEntropyFocalLoss()

n_epochs = 5

PATH = './f2bev_attn_mt.pt'
min_val_loss = np.inf
min_epoch = n_epochs

for epoch in range(n_epochs):
    train_loss = train(epoch,model,train_dataloader,seq_len,criterionCE,optimizer)
    val_loss = val(epoch,val_dataloader,seq_len,model,criterionCE)
    
    if val_loss < min_val_loss:
        min_epoch = epoch
        min_val_loss = val_loss        
        torch.save({'epoch' : epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict' : optimizer.state_dict(), 'loss': val_loss}, PATH)
    else:
        if epoch > min_epoch:
            break


