#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For quickloading and visualization
"""

# imports
import numpy as np
import h5py
import torch
import cv2

#### Load Encoded Data
data_path = 'datasets/DOTIE_Encoding/count_data/500.npy'
data = np.load(data_path)
splits = int(data.shape[3]/4)
data_shorter = data[0,:,:,:splits] + data[1,:,:,:splits]

# Load small part of the encoded data
events_single_car = data_shorter[:,:,23000:25000]
events_single_car_tensor = torch.tensor(data_shorter[:,:,23000:25000])

gray_path = 'datasets/DOTIE_Encoding/count_data/gray_ind.npy'
grayind = np.load(gray_path)
new_grayind = grayind[ np.searchsorted(grayind, 23000): np.searchsorted(grayind, 25000)+1]
save_indices  = np.arange(np.searchsorted(grayind, 23000), np.searchsorted(grayind, 25000)+1)

#### Import grayscale images for rendering
mvsec_path = 'datasets/outdoor_day2_data.hdf5'
d_set = h5py.File(mvsec_path, 'r')
gray_image = d_set['davis']['left']['image_raw']
new_grayimg = gray_image[save_indices]
new_grayind_reduced = new_grayind - 23000

hf = h5py.File('datasets/QuickLoads_mvsec.hdf5', 'w')
hf.create_dataset('event_data', data=events_single_car_tensor)
hf.create_dataset('grayind', data=new_grayind_reduced)
hf.create_dataset('gray_img', data=new_grayimg)
hf.close()

### Testing quickloading 
data_path_short = 'datasets/DOTIE_Encoding/QuickLoads_mvsec.hdf5'
d_set = h5py.File(data_path_short , 'r')
evnts_enc= torch.tensor(d_set['event_data'])
gray_idx = np.array(d_set['grayind'])
gray_imgs = np.array(d_set['gray_img'])

indx_for_gray= 0
blank_gray_image = np.zeros(gray_imgs[0].shape)
gryimg = np.array(blank_gray_image, dtype=np.uint8)

for curr_pos in range(evnts_enc.shape[2]):    
    if indx_for_gray < len(gray_idx):
        if int(gray_idx[indx_for_gray]) == curr_pos:
            gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)  
            indx_for_gray += 1
    
    visual_frame = np.array(evnts_enc[:, :, curr_pos])
    # normalize input
    evnt_frame = ((visual_frame - visual_frame.min()) * (1/(visual_frame.max() - visual_frame.min()) * 255)).astype('uint8')
    
    visualizing = np.concatenate((gryimg, evnt_frame), axis=1)
    cv2.imshow('QuickLoad', visualizing)
    cv2.waitKey(1)
    