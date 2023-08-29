#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for comparing DOTIE to other algorithms
"""

# import necessary libraries
import snntorch as snn
import torch
import torch.nn as nn
import h5py
import numpy as np
import cv2

from visual_helpers import convert_to_contrast_3chnl, recover_fast_inputs, convert_to_3chnl
from Clustering_techniques import compare_all

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
    pos_changed_flag = False
    DOTIE_scores, GSCE_scores, Kmeans_scores, meanshift_scores, DBSCAN_scores, GMM_scores = [], [], [], [], [], []
    
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

    # Replace 300 with starting index
    while (int(gray_idx[indx_for_gray]) < 300):
        indx_for_gray += 1


    blank_gray_image = np.zeros(gray_imgs[0].shape)
    gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)

    print('\n*****************************************************')
    print('\nProcessing Input Sequence')
    # Loop over the indices 300 to 400, loading each time instance (3rd dimension of data tensor) in each iteration
    for curr_pos in range(300, 400):
        
        print('\nFrame ', curr_pos)
        
        # DOTIE framework
        inp_img = evnts_enc[:, :, curr_pos].float()
        inp_img = inp_img[None, None, :]
        con_out = conv1(inp_img)
        spk_dir, mem_dir = snn1(con_out, mem_dir)      
        
        
        ### Visualizations

        # Get Grayscale images
        if indx_for_gray < len(gray_idx):
            if int(gray_idx[indx_for_gray]) == curr_pos:
                gryimg = np.array(gray_imgs[indx_for_gray], dtype=np.uint8)
                pos_changed_flag = True  
                indx_for_gray += 1
        gryimg_3chnl = convert_to_3chnl(gryimg)

        # Input Events
        visual_frame = np.array(evnts_enc[:, :, curr_pos])
        # normalize input
        evnt_frame = ((visual_frame - visual_frame.min()) * (1/(visual_frame.max() - visual_frame.min()) * 255)).astype('uint8')
        evnt_frame_3chnl = convert_to_contrast_3chnl(evnt_frame)
        
        # DOTIE output
        spk_frame = torch.squeeze(spk_dir.detach())
        spk_frame[spk_frame>0] = 255
        spk_frame = np.array(spk_frame, dtype=np.uint8)
        recovered_inputs = recover_fast_inputs(evnt_frame, spk_frame, recovery_neighborhood=12)
        recovered_inputs_3chnl = convert_to_3chnl(recovered_inputs)

        # Compare techniques 
        gray_image_3chnl, DOTIE_img, GSCE_img, Kmeans_img, meanshift_img, DBSCAN_img, GMM_img, DOTIE_sc, GSCE_sc, Kmeans_sc, meanshift_sc, DBSCAN_sc, GMM_sc = compare_all(evnt_frame_3chnl, gryimg_3chnl, recovered_inputs_3chnl, evnt_frame, eps_val=15, mindiagonalsquared=2300, gsce_neighbors=100, withIoU=True)
        vis_1a = np.concatenate((gray_image_3chnl, evnt_frame_3chnl), axis=1)
        vis_1b = np.concatenate((DOTIE_img, DBSCAN_img), axis=1)
        vis_1 = np.concatenate((vis_1a, vis_1b), axis=1)

        vis_2a = np.concatenate((GSCE_img, Kmeans_img), axis=1)
        vis_2b = np.concatenate((meanshift_img, GMM_img), axis=1)
        vis_2 = np.concatenate((vis_2a, vis_2b), axis=1)
                               
        visual = np.concatenate((vis_1, vis_2), axis=0)
        cv2.imshow('Comparison of Algorithms', cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    print('\n******************Done****************************')