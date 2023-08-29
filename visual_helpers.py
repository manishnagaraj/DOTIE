#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 11:47:02 2023

@author: mnagara
"""
import numpy as np
import cv2

def convert_to_contrast_3chnl(grayarray, fgn=[153,0,17], bck=[252,246,245]):
    colorarray = np.zeros((grayarray.shape[0], grayarray.shape[1], 3))
    for x in range(grayarray.shape[0]):
        for y in range(grayarray.shape[1]):
            if grayarray[x,y] > 0:
                colorarray[x,y,:] = fgn
            else:
                colorarray[x,y,:] = bck
    return colorarray.astype(np.uint8)

def convert_3chnl_to_contrast_3chnl(grayarray, fgn=[153,0,17], bck=[252,246,245]):
    colorarray = np.zeros((grayarray.shape[0], grayarray.shape[1], 3))
    for x in range(grayarray.shape[0]):
        for y in range(grayarray.shape[1]):
            if grayarray[x,y,0] > 0:
                colorarray[x,y,:] = fgn
            else:
                colorarray[x,y,:] = bck
    return colorarray.astype(np.uint8)


def convert_to_3chnl(grayarray):
    colorarray = np.zeros((grayarray.shape[0], grayarray.shape[1], 3))
    colorarray[:,:,0] = grayarray
    colorarray[:,:,1] = grayarray
    colorarray[:,:,2] = grayarray
    return colorarray.astype(np.uint8)


def recover_fast_inputs(input_array, spk_output_array, recovery_neighborhood=5):
    kernel = np.ones((recovery_neighborhood, recovery_neighborhood),np.uint8)
    dilated_speedy_img = cv2.dilate(np.array(spk_output_array), kernel, iterations = 1)
    closing = cv2.morphologyEx(dilated_speedy_img, cv2.MORPH_CLOSE, kernel)
    masked_input = np.array(np.logical_and(input_array,closing))*input_array
    return masked_input