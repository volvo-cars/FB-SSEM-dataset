#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 10:27:02 2023

@author: smartslab
"""

import cv2,fnmatch,os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R



files =  fnmatch.filter(os.listdir('./forward_looking_camera_model/data/bev'),'*.png')



def load_yaml(filename):
    content = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    K = content.getNode("K").mat()
    D = content.getNode("D").mat()
    xi = content.getNode("xi").mat()
    return K,D,xi
    

def convertPoseFromUnityToOpenCV(pose):
    opencvpose = {}
    for key,value in pose.items():
        # ======================= UnityToOpenCV: Flip y axis ========================
        opencvpose[key] = [[-1,1,-1]*value[0] , [1,-1,1]*value[1]]
        # ===========================================================================
    return opencvpose


def convertOpenCVPoseToRvecTvec(opencvpose):
    extrinsics = {}
    for key,value in opencvpose.items():
        rot = value[0]
        # ====================== Use "ZYX" for extrinsic rotation =================== 
        intermediater = R.from_euler('ZYX',[rot[2], rot[1], rot[0]])
        # =========================================================================== 
        rotmat = intermediater.as_matrix()
        # ========================= cam2world -> world2cam ==========================
        '''
        cam2world: R, t
        world2cam: R', t'
        R' = R.T
        t' = - R.T @ t
        '''
        rvec,_ = cv2.Rodrigues(rotmat.T) 
        tvec = -rotmat.T @ value[1].reshape(3,1)
        # ===========================================================================
        extrinsics[key] = [rvec,tvec]
    return extrinsics
        

def computeRealWorldLocationOfBEVPixels(h,w, resolution,height_anchors,scale,offset):
    z = np.arange(0, h, 1)
    x = np.arange(0, w, 1)
    zprime = ((offset/scale)+h/2-z)*resolution[1]  # offset for unity...unity origin at mid of the car rear and on the ground
    # ==============================================================================
    xprime = (x-w/2)*resolution[0]
    # ==============================================================================
    
    worldpoints = []
    bevpointsforworldpoints = []
    for j,tz in enumerate(zprime):
        for i,tx in enumerate(xprime):
            for ty in height_anchors:
                worldpoints.append([tx,-ty,tz]) # because opencv right handed coordinate system has x,z same as unity but y is downward
                bevpointsforworldpoints.append([x[i],z[j]])
                
    return np.asarray(worldpoints),np.asarray(bevpointsforworldpoints)



def getValidProjectPoints(imgpoints,validfisheyemask):
    validpointidxes = []
    validimgpoints = []
    inpfisheyeshape = validfisheyemask.shape
    imgpoints = imgpoints.astype(int)
    for i in range(len(imgpoints)):
        loc = imgpoints[i,:]
        if loc[0] > 0 and loc[0] < inpfisheyeshape[1]:
            if loc[1] > 0 and loc[1] < inpfisheyeshape[0]:        
                if validfisheyemask[loc[1],loc[0]] == 255:
                    validimgpoints.append(loc)
                    validpointidxes.append(i)
    return np.asarray(validimgpoints).astype(int),validpointidxes




unitypose = {}
unitypose['front'] = [(np.pi/180) *np.asarray([26,0,0]),np.asarray([0,0.406,3.873])]  ##xyz as per unity
unitypose['left'] = [(np.pi/180) *np.asarray([0,-90,0]),np.asarray([-1.024,0.8,2.053])] 
unitypose['rear'] = [(np.pi/180) *np.asarray([3,180,0]),np.asarray([0.132,0.744,-1.001])]
unitypose['right'] = [(np.pi/180) *np.asarray([0,90,0]),np.asarray([1.015,0.801,2.04])]

opencvpose = convertPoseFromUnityToOpenCV(unitypose)            
extrinsics = convertOpenCVPoseToRvecTvec(opencvpose)

for file in files[0:1]:
    

    unity_offset_for_orgin = 33 #pixels 56 is ffset for unity...unity origin at mid of the car rear wheel axis and on the ground
    #this 56 is considering 600*600 BEV. If I resize it this will change. If I crop it, it will not change
    
    batch_size = 1
    bev_h = 400 #height of the Unity generated bev i consider (if i crop then consider cropped size)
    bev_w = 400 #width of the Unity generated bev i consider (if i crop then consider  cropped size)
    
    bh = 50
    bw = 50
    bev_scale = int(bev_h/bh) ##600/50
    bevformer_bev_size = (bh,bw)

    
    K,D,xi = load_yaml('./forward_looking_camera_model/flcw_unity.yml')
    #resolution = bev_scale*np.asarray([0.036,0.042])
    resolution = bev_scale*np.asarray([2/48,2/48])      ## resolution: 48 pixels is 2m
    bev_mask = []

    height_anchors = [0-0.377, 0.25 -0.377,1.8 - 0.377] #[0, 0.25,1.8] #[0, 1.5, 3, 4.5] # in meters

    bev_mask = []
    reference_points_cam = []
    for camtype in ['front','left','rear','right']:

        img = cv2.imread('./forward_looking_camera_model/data/'+camtype+'/'+file)
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        validfisheyemask = np.load('./forward_looking_camera_model/masks/'+camtype+'.npy')
        
        if camtype == 'front' or camtype=='rear':
            validfisheyemask = np.flip(validfisheyemask)
            
        view_bev_mask = []
        view_reference_points_cam = []
        
        for height in height_anchors:
            
            worldpoints,bevptsforworldpoints = computeRealWorldLocationOfBEVPixels(bh,bw, resolution,[height],bev_scale,unity_offset_for_orgin)
            worldpoints = np.expand_dims(worldpoints, 0)

            imgpoints,_ = cv2.omnidir.projectPoints(worldpoints, extrinsics[camtype][0], extrinsics[camtype][1], K, xi[0,0], D)
            validimgpoints,valididxes = getValidProjectPoints(imgpoints[0,:,:], validfisheyemask)
            
            ## new addition
            filtered_valididxes = []
            for idx in valididxes:
                
                bloc = bevptsforworldpoints[idx,:]
                #print(bloc)
                if (camtype == 'front' and bloc[1] < 25) or  (camtype == 'left' and bloc[0] < 25) or (camtype == 'rear' and bloc[1] > 25) or (camtype == 'right' and bloc[0] > 25):
                    filtered_valididxes.append(idx)
                    
                    
            curr_bev_mask = np.zeros((bh*bw,))
            
            for fidx in filtered_valididxes:
                curr_bev_mask[fidx] = 1

            imgpoints[0,:,0] = imgpoints[0,:,0].astype(float)/w  ##normalize 
            imgpoints[0,:,1] = imgpoints[0,:,1].astype(float)/h        ##normalize 
            
            plt.figure()
            plt.imshow(np.reshape(curr_bev_mask,(50,50)))
            
            view_bev_mask.append(curr_bev_mask)
            view_reference_points_cam.append(imgpoints[0,:,:])
            
        view_bev_mask = np.expand_dims(np.transpose(np.asarray(view_bev_mask)), axis = 0)
        view_reference_points_cam = np.transpose(np.asarray(view_reference_points_cam))
        view_reference_points_cam = np.transpose(view_reference_points_cam,[1,2,0]) 
        view_reference_points_cam = np.expand_dims(view_reference_points_cam, axis = 0)   
        
        bev_mask.append(view_bev_mask)
        reference_points_cam.append(view_reference_points_cam)
    
    bev_mask = np.tile(np.asarray(bev_mask),(1,batch_size,1,1))
    reference_points_cam = np.tile(np.asarray(reference_points_cam),(1,batch_size,1,1,1))
                    
    np.save('./unity_data/bev_mask.npy',np.array(bev_mask,dtype=bool))
    np.save('./unity_data/reference_points_cam.npy',reference_points_cam)
