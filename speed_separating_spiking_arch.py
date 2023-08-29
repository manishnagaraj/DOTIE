#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for the entire DOTIE framework (DOTIE + Clustering )
"""

# import necessary libraries
import snntorch as snn
import torch
import torch.nn as nn
import h5py
import numpy as np
import cv2

from visual_helpers import convert_to_contrast_3chnl, recover_fast_inputs, convert_to_3chnl
from Clustering_techniques import get_boundaries_DOTIE

if __name__=="__main__":

    #### Load Encoded Data
    data_path = 'datasets/QuickLoads_mvsec.hdf5'
    d_set = h5py.File(data_path , 'r')
    evnts_enc= torch.tensor(d_set['event_data'])
    gray_idx = np.array(d_set['grayind'])
    gray_imgs = np.array(d_set['gray_img'])

    #### IF USING FULL ENCODING UNCOMMENT THE FOLLOWING  and comment out previous loading statements ####
    '''
    evnts_enc = np.load('datasets/DOTIE_Encoding/count_data/500.npy')
    gray_idx = np.load('datasets/DOTIE_Encoding/count_data/gray_ind.npy')
    d_set = h5py.File('datasets/outdoor_day2_data.hdf5', 'r')
    gray_imgs = d_set['davis']['left']['image_raw']
    '''
    
    
    # Convolutional layer (3x3)
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False).to('cpu')
    # Set the weights manually
    conv1.weight = torch.nn.Parameter(torch.tensor(torch.ones_like(conv1.weight)*0.15))
    with torch.no_grad():
        conv1.weight[0, 0, 1, 1] = 0.2
    # spiking neuron parameters set manually
    snn1 = snn.Leaky(beta=0.3, reset_mechanism="subtract")
    mem_dir = snn1.init_leaky()

    indx_for_gray= 0
    blank_gray_image = np.zeros(gray_imgs[0].shape)
    gryimg = np.array(blank_gray_image, dtype=np.uint8)

    print('\n*****************************************************')
    print('\nProcessing Input Sequence')
    # Loop over the entire dataset once, loading each time instance (3rd dimension of data tensor) in each iteration
    for curr_pos in range(evnts_enc.shape[2]):
        
        print('\nFrame ', curr_pos)
        # convert input tensor into float type
        inp_img = evnts_enc[:, :, curr_pos].float()
        # Add two dimensions (batch size and channels)
        inp_img = inp_img[None, None, :]
        # Pass it through conv layer
        con_out = conv1(inp_img)
        # Pass the output (weighted sum through spiking layer, along with previous membrane potential)
        # Output -- spike output ; updated potential
        spk_dir, mem_dir = snn1(con_out, mem_dir)      
        
        
        ### Visualizations

        # Get Grayscale images
        if indx_for_gray < len(gray_idx):
            if int(gray_idx[indx_for_gray]) == curr_pos:
                gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)  
                indx_for_gray += 1
        gryimg_3chnl = convert_to_3chnl(gryimg)

        # Input Events
        visual_frame = np.array(evnts_enc[:, :, curr_pos])
        # normalize input
        evnt_frame = ((visual_frame - visual_frame.min()) * (1/(visual_frame.max() - visual_frame.min()) * 255)).astype('uint8')
       
        # convert 4dimensional spike output to 2 dimensional image
        spk_frame = torch.squeeze(spk_dir.detach())
        # binarize output
        spk_frame[spk_frame>0] = 255
        spk_frame = np.array(spk_frame, dtype=np.uint8)
        
        # Recover inputs that generate outputs (i.e., fast inputs)
        recovered_inputs = recover_fast_inputs(evnt_frame, spk_frame, recovery_neighborhood=12)
        recovered_inputs_3chnl = convert_to_3chnl(recovered_inputs)

        DOTIE_gt_box = get_boundaries_DOTIE(recovered_inputs_3chnl, eps_val=15, min_samples_val=10, mindiagonalsquared=2300)
        
        # Output visual
        visual =  np.concatenate((gryimg_3chnl, np.concatenate((convert_to_contrast_3chnl(evnt_frame), convert_to_contrast_3chnl(recovered_inputs)), axis=1)), axis=1)
        cv2.imshow('DOTIE Spiking Arch', cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
    print('\n******************Done****************************')
        