#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper codes for clustering
"""

import numpy as np
import cv2
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import silhouette_score

from pytorchyolo import detect, models
from visual_helpers import convert_to_contrast_3chnl

##################################################################
### Helpers for Metrics
##################################################################

def _highlight_only_correct_gt_box_(event_boxes, gray_boxes):
    no_gray_boxes = len(gray_boxes)
    no_event_boxes = len(event_boxes)
    score_map = np.zeros((no_event_boxes,no_gray_boxes))
    
    for evtid in range(no_event_boxes):
        for gryid in range(no_gray_boxes):
            score_map[evtid, gryid] = _compute_IOU_(event_boxes[evtid], gray_boxes[gryid])
    
    true_gray_boxes = []
    true_score = []
    for evtid in range(no_event_boxes):
        true_box_id = np.argmax(score_map[evtid, :])
        true_score += [np.max(score_map[evtid, :])]
        true_gray_boxes += [gray_boxes[true_box_id]]
        
    return true_gray_boxes, true_score

def _compute_IOU_(event_box, gt_box):
    xes, yes, xee, yee = event_box
    xgs, ygs, xge, yge, _, _ = gt_box
    aeb = _compboxarea_(xes, yes, xee, yee) #Area of event box --- length*width
    agb = _compboxarea_(xgs, ygs, xge, yge) #Area of ground box
    
    xis = max(xes, xgs)
    yis = max(yes, ygs)
    xie = min(xee, xge)
    yie = min(yee, yge)
    
    aib = _compboxarea_(xis, yis, xie, yie)
    aub = aeb + agb - aib
    return aib/aub

def _compboxarea_(x1, y1, x2, y2):
    length = x2-x1
    width = y2-y1
    area = length * width
    return area

def _compute_maxIOU_wrt_true_gt_(event_boxes, true_gt_box):
    scores = np.zeros(len(event_boxes))
    if len(scores)>0:
        for event_id in range(len(event_boxes)):
            temp_xgs, temp_ygs, temp_xge, temp_yge = true_gt_box
            temp_gt_box = (temp_xgs, temp_ygs, temp_xge, temp_yge, 0, 0)
            scores[event_id] = _compute_IOU_(event_boxes[event_id], temp_gt_box)
        
        true_score = np.max(scores)
        true_event_box = event_boxes[np.argmax(scores)]
        return true_score, true_event_box
    
    else:
        return 0, []

def _get_IoU_metrics_(IoUscores_array, threshold):
    flat_list = [item for sublist in IoUscores_array for item in sublist]
    FN = 0
    for val in IoUscores_array:
        if val == []:
            FN += 1
    true_positives = np.zeros((len(flat_list)))
    true_positives[np.array(flat_list)>=threshold] = 1
    false_positives = np.zeros((len(flat_list)))
    false_positives[np.array(flat_list)<threshold] = 1
    TP = sum(true_positives)
    FP = sum(false_positives)
    if TP == 0 :
        Precision = 0
        Recall = 0
        F_measure = 0
    else:
        Precision = (TP)/(TP+FP)
        Recall = (TP)/(TP+FN)
        F_measure = (2*Precision*Recall)/(Precision+Recall)
    mean_IoU = sum(flat_list)/len(flat_list)
    return Precision, Recall, F_measure, mean_IoU

##################################################################################
## Functions to get bounding boxes
##################################################################################

def get_boundaries_DOTIE(isolated_evnts_3chnl, eps_val=8, min_samples_val=10, mindiagonalsquared=100):
    isolated_evnts = isolated_evnts_3chnl[:,:,0]
    Ximage_x, Ximage_y = np.where(isolated_evnts > 0)
    if len(Ximage_x) < min_samples_val:
        return  None
    X = np.zeros((len(Ximage_x), 2))
    X[:,0], X[:,1] = Ximage_x, Ximage_y
    clustered_image = np.zeros((isolated_evnts.shape))
    for x, y in X:
        clustered_image[int(x),int(y)] = 255      
    db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
    db.fit(X)
    y_pred = db.fit_predict(X)    
    ev_box = []
    # loop around all cluster labels
    no_labels = y_pred.max()+1
    for lbl in range(no_labels):
        # print(lbl)
        y_lbl_3 = np.where(y_pred == lbl)
        x_vals = Ximage_x[y_lbl_3]
        y_vals = Ximage_y[y_lbl_3]
       
        diagon = ((x_vals.max()-x_vals.min())**2) + ((y_vals.max()-y_vals.min())**2)
        if diagon < mindiagonalsquared:
            pass
        else: 
            ev_box += [(int(y_vals.min()), int(x_vals.min()), int(y_vals.max()), int(x_vals.max()))]
    return ev_box


def getboundaries_other(event_input_3chnl, NEIGHBORS=100, eps_val=8, min_samples_val=10, mindiagonalsquared=100):
    image_array = event_input_3chnl[:,:,0]
    Ximage_x, Ximage_y = np.where(image_array > 0)
    selected_events = np.zeros((len(Ximage_x),3))
    for i in range(len(Ximage_x)):
            selected_events[i][0] = Ximage_x[i]
            selected_events[i][1] = Ximage_y[i]
            selected_events[i][2] = image_array[Ximage_x[i]][Ximage_y[i]]
                
    selected_events = np.asarray(selected_events)
    # pdb.set_trace()
    if len(selected_events) < NEIGHBORS:
        neighbors_temp = len(selected_events) - 1
    else:
        neighbors_temp = NEIGHBORS

    adMat = kneighbors_graph(selected_events, n_neighbors=neighbors_temp)
    max_score = -20
    opt_clusters = 2
    scores = []
    X = np.zeros((len(Ximage_x), 2))
    X[:,0], X[:,1] = Ximage_x, Ximage_y
    ev_box_gsce = []
    ev_box_kmeans = []
    ev_box_meanshift = []
    ev_box_dbscan = []
    ev_box_gmm = []
    
    print('predicting number of clusters...')
    for CLUSTERS in range(2, 7):
        clustering = SpectralClustering(n_clusters=CLUSTERS, random_state=0,
                                        affinity='precomputed_nearest_neighbors',
                                        n_neighbors=neighbors_temp, assign_labels='kmeans',
                                        n_jobs=-1).fit_predict(adMat)
        curr_score = silhouette_score(selected_events, clustering)
        scores.append(curr_score)
        if curr_score > max_score:
            max_score = curr_score
            opt_clusters = CLUSTERS
    print('clustering...')
    clustering = SpectralClustering(n_clusters=opt_clusters, random_state=0,
                                    affinity='precomputed_nearest_neighbors',
                                    n_neighbors=neighbors_temp, assign_labels='kmeans',
                                    n_jobs=-1).fit_predict(adMat)
    ## GSCE 
    # Outlay these clusters on input_event_map
    no_labels = clustering.max()+1
    for lbl in range(no_labels):
        # print(lbl)
        y_lbl_3 = np.where(clustering == lbl)
        x_vals = selected_events[y_lbl_3][:,0] 
        y_vals = selected_events[y_lbl_3][:,1]
        diagon = ((x_vals.max()-x_vals.min())**2) + ((y_vals.max()-y_vals.min())**2)
        if diagon < mindiagonalsquared:
            pass
        else: 
            ev_box_gsce += [(int(y_vals.min()), int(x_vals.min()), int(y_vals.max()), int(x_vals.max()))]
        
    ## KMEANS    
    clustering_kmeans = KMeans(n_clusters=opt_clusters, random_state=0).fit_predict(selected_events)
    # Outlay these clusters on input_event_map
    no_labels_kmeans = clustering_kmeans.max()+1
    for lbl in range(no_labels_kmeans):
        # print(lbl)
        y_lbl_3 = np.where(clustering_kmeans == lbl)
        x_vals = selected_events[y_lbl_3][:,0] 
        y_vals = selected_events[y_lbl_3][:,1]
        diagon = ((x_vals.max()-x_vals.min())**2) + ((y_vals.max()-y_vals.min())**2)
        if diagon < mindiagonalsquared:
            pass
        else: 
            ev_box_kmeans += [(int(y_vals.min()), int(x_vals.min()), int(y_vals.max()), int(x_vals.max()))]
        
        
    ## MEANSHIFT
    BW = estimate_bandwidth(selected_events)
    clustering_meanshift = MeanShift(bandwidth=BW).fit_predict(selected_events)
    # Outlay these clusters on input_event_map
    no_labels_meanshift = clustering_meanshift.max()+1
    for lbl in range(no_labels_meanshift):
        # print(lbl)
        y_lbl_3 = np.where(clustering_meanshift == lbl)
        x_vals = selected_events[y_lbl_3][:,0] 
        y_vals = selected_events[y_lbl_3][:,1]
        diagon = ((x_vals.max()-x_vals.min())**2) + ((y_vals.max()-y_vals.min())**2)
        if diagon < mindiagonalsquared:
            pass
        else: 
            ev_box_meanshift += [(int(y_vals.min()), int(x_vals.min()), int(y_vals.max()), int(x_vals.max()))]
        
    ## DBSCAN directly
    clustering_dbscan = DBSCAN(eps = eps_val, min_samples=min_samples_val).fit_predict(selected_events)
    # Outlay these clusters on input_event_map
    no_labels_dbscan = clustering_dbscan.max()+1
    for lbl in range(no_labels_dbscan):
        # print(lbl)
        y_lbl_3 = np.where(clustering_dbscan == lbl)
        x_vals = selected_events[y_lbl_3][:,0] 
        y_vals = selected_events[y_lbl_3][:,1]
        diagon = ((x_vals.max()-x_vals.min())**2) + ((y_vals.max()-y_vals.min())**2)
        if diagon < mindiagonalsquared:
            pass
        else: 
            ev_box_dbscan += [(int(y_vals.min()), int(x_vals.min()), int(y_vals.max()), int(x_vals.max()))]
            
    ## GMM
    clustering_gmm = GaussianMixture(n_components=opt_clusters, random_state=0).fit_predict(selected_events)
    # Outlay these clusters on input_event_map
    no_labels_gmm = clustering_gmm.max()+1
    for lbl in range(no_labels_gmm):
        # print(lbl)
        y_lbl_3 = np.where(clustering_gmm == lbl)
        x_vals = selected_events[y_lbl_3][:,0] 
        y_vals = selected_events[y_lbl_3][:,1]
        diagon = ((x_vals.max()-x_vals.min())**2) + ((y_vals.max()-y_vals.min())**2)
        if diagon < mindiagonalsquared:
            pass
        else: 
            ev_box_gmm+= [(int(y_vals.min()), int(x_vals.min()), int(y_vals.max()), int(x_vals.max()))]
   
    return ev_box_gsce, ev_box_kmeans, ev_box_meanshift, ev_box_dbscan, ev_box_gmm

def compare_all(evnt_inp_3chnl, gray_image_3chnl, isolated_evnts_3chnl, evnt_inp, eps_val=8, min_samples_val=10, mindiagonalsquared=100, gsce_neighbors=100, withIoU=True):
    # Load YOLO model 
    model = models.load_model("models/config/yolov3.cfg", "models/weights/yolov3.weights")
    yolo_boxes = detect.detect_image(model, np.array(gray_image_3chnl, dtype=np.uint8))
    

    # Initialize array for scores 
    DOTIE_sc, GSCE_sc, Kmeans_sc, meanshift_sc, DBSCAN_sc, GMM_sc = [], [], [], [], [], []
    # Initialize copies of input events to overlay bboxes
    contrasted_inp = convert_to_contrast_3chnl(evnt_inp)
    DOTIE_img, GSCE_img, Kmeans_img, meanshift_img, DBSCAN_img, GMM_img = contrasted_inp.copy(), contrasted_inp.copy(), contrasted_inp.copy(), contrasted_inp.copy(), contrasted_inp.copy(), contrasted_inp.copy()

    bbox_col = (0,0,0)
    other_bbox_col = (173,216,230)
    gt_color =(255,255,0)

    # Label the images
    cv2.putText(DOTIE_img, 'DOTIE', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)    
    cv2.putText(GSCE_img, 'GSCE', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)
    cv2.putText(Kmeans_img, 'K-Means', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2) 
    cv2.putText(meanshift_img, 'Meanshift', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2) 
    cv2.putText(DBSCAN_img, 'DBSCAN directly', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2) 
    cv2.putText(GMM_img, 'GMM', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2) 

    # Get boundaries using DOTIE
    DOTIE_bb = get_boundaries_DOTIE(isolated_evnts_3chnl, eps_val, min_samples_val, mindiagonalsquared)
    Yolo_bb_true = []

    if DOTIE_bb == None:
        return gray_image_3chnl, DOTIE_img, GSCE_img, Kmeans_img, meanshift_img, DBSCAN_img, GMM_img, [0], [0], [0], [0], [0], [0]

    for dbb in DOTIE_bb:
        yolo_box_true, yolo_box_score = _highlight_only_correct_gt_box_([dbb], yolo_boxes)
        yolo_score_str = "{:.4f}".format(yolo_box_score[0])
        gt_xs, gt_ys, gt_xe, gt_ye, _, _ = yolo_box_true[0]
        Yolo_bb_true += [(int(gt_xs), int(gt_ys), int(gt_xe), int(gt_ye))]
        cv2.rectangle(DOTIE_img, (dbb[0], dbb[1]), (dbb[2], dbb[3]), bbox_col, 3, cv2.LINE_AA)      
        if withIoU == True:
            DOTIE_sc += [yolo_box_score[0]]
            cv2.putText(DOTIE_img, 'IoU='+ yolo_score_str, (dbb[0], dbb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)

    # Mark the YOLO boxes considered on the input image as well
    if withIoU == True:
        for true_gt in Yolo_bb_true:
            gt_xs, gt_ys, gt_xe, gt_ye = true_gt
            cv2.rectangle(gray_image_3chnl, (int(gt_xs), int(gt_ys)), (int(gt_xe), int(gt_ye)), gt_color, 3, cv2.LINE_AA)    

    # Get bounding boxes from all other clustering techniques
    gsce_boxes, kmeans_boxes, meanshift_boxes, dbscan_boxes, gmm_boxes = getboundaries_other(isolated_evnts_3chnl, gsce_neighbors, eps_val, min_samples_val, mindiagonalsquared)

    # Process GSCE
    for gsce_box in gsce_boxes:
        cv2.rectangle(GSCE_img, (gsce_box[0], gsce_box[1]), (gsce_box[2], gsce_box[3]), other_bbox_col, 3, cv2.LINE_AA) 
    for true_gt in Yolo_bb_true:
        if withIoU == True:
            max_score, true_evt_box = _compute_maxIOU_wrt_true_gt_(gsce_boxes, true_gt)
            GSCE_sc += [max_score]
            gsce_score_str = "{:.4f}".format(max_score)
            if len(true_evt_box) == 4:
                cv2.rectangle(GSCE_img, (true_evt_box[0], true_evt_box[1]), (true_evt_box[2], true_evt_box[3]), bbox_col,3, cv2.LINE_AA) 
                cv2.putText(GSCE_img, 'IoU='+gsce_score_str, (true_evt_box[0], true_evt_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)
                if true_evt_box[0] < 0 or true_evt_box[1]-10 < 0:
                    cv2.putText(GSCE_img, 'IoU='+gsce_score_str, (true_evt_box[0]+20, true_evt_box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)

    # Process KMEANS
    for kmeans_box in kmeans_boxes:
        cv2.rectangle(Kmeans_img, (kmeans_box[0], kmeans_box[1]), (kmeans_box[2], kmeans_box[3]), other_bbox_col, 3, cv2.LINE_AA) 
    for true_gt in Yolo_bb_true:
        if withIoU == True:
            max_score, true_evt_box = _compute_maxIOU_wrt_true_gt_(kmeans_boxes, true_gt)
            Kmeans_sc += [max_score]
            kmeans_score_str = "{:.4f}".format(max_score)
            if len(true_evt_box) == 4:
                cv2.rectangle(Kmeans_img, (true_evt_box[0], true_evt_box[1]), (true_evt_box[2], true_evt_box[3]), bbox_col,3, cv2.LINE_AA) 
                cv2.putText(Kmeans_img, 'IoU='+kmeans_score_str, (true_evt_box[0], true_evt_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)
                if true_evt_box[0] < 0 or true_evt_box[1]-10 < 0:
                    cv2.putText(Kmeans_img, 'IoU='+kmeans_score_str, (true_evt_box[0]+20, true_evt_box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)

    # Process meanshift
    for meanshift_box in meanshift_boxes:
        cv2.rectangle(meanshift_img, (meanshift_box[0], meanshift_box[1]), (meanshift_box[2], meanshift_box[3]), other_bbox_col, 3, cv2.LINE_AA) 
    for true_gt in Yolo_bb_true:
        if withIoU == True:
            max_score, true_evt_box = _compute_maxIOU_wrt_true_gt_(meanshift_boxes, true_gt)
            meanshift_sc += [max_score]
            meanshift_score_str = "{:.4f}".format(max_score)
            if len(true_evt_box) == 4:
                cv2.rectangle(meanshift_img, (true_evt_box[0], true_evt_box[1]), (true_evt_box[2], true_evt_box[3]), bbox_col,3, cv2.LINE_AA) 
                cv2.putText(meanshift_img, 'IoU='+meanshift_score_str, (true_evt_box[0], true_evt_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)
                if true_evt_box[0] < 0 or true_evt_box[1]-10 < 0:
                    cv2.putText(meanshift_img, 'IoU='+meanshift_score_str, (true_evt_box[0]+20, true_evt_box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)

    # Process DBSCAN
    for dbscan_box in dbscan_boxes:
        cv2.rectangle(DBSCAN_img, (dbscan_box[0], dbscan_box[1]), (dbscan_box[2], dbscan_box[3]), other_bbox_col, 3, cv2.LINE_AA) 
    for true_gt in Yolo_bb_true:
        if withIoU == True:
            max_score, true_evt_box = _compute_maxIOU_wrt_true_gt_(dbscan_boxes, true_gt)
            DBSCAN_sc += [max_score]
            DBSCAN_score_str = "{:.4f}".format(max_score)
            if len(true_evt_box) == 4:
                cv2.rectangle(DBSCAN_img, (true_evt_box[0], true_evt_box[1]), (true_evt_box[2], true_evt_box[3]), bbox_col,3, cv2.LINE_AA) 
                cv2.putText(DBSCAN_img, 'IoU='+DBSCAN_score_str, (true_evt_box[0], true_evt_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)
                if true_evt_box[0] < 0 or true_evt_box[1]-10 < 0:
                    cv2.putText(DBSCAN_img, 'IoU='+DBSCAN_score_str, (true_evt_box[0]+20, true_evt_box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)

    # Process GMM
    for gmm_box in gmm_boxes:
        cv2.rectangle(GMM_img, (gmm_box[0], gmm_box[1]), (gmm_box[2], gmm_box[3]), other_bbox_col, 3, cv2.LINE_AA) 
    for true_gt in Yolo_bb_true:
        if withIoU == True:
            max_score, true_evt_box = _compute_maxIOU_wrt_true_gt_(gmm_boxes, true_gt)
            GMM_sc += [max_score]
            GMM_score_str = "{:.4f}".format(max_score)
            if len(true_evt_box) == 4:
                cv2.rectangle(GMM_img, (true_evt_box[0], true_evt_box[1]), (true_evt_box[2], true_evt_box[3]), bbox_col,3, cv2.LINE_AA) 
                cv2.putText(GMM_img, 'IoU='+GMM_score_str, (true_evt_box[0], true_evt_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)
                if true_evt_box[0] < 0 or true_evt_box[1]-10 < 0:
                    cv2.putText(GMM_img, 'IoU='+GMM_score_str, (true_evt_box[0]+20, true_evt_box[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bbox_col, 2)

    
    return gray_image_3chnl, DOTIE_img, GSCE_img, Kmeans_img, meanshift_img, DBSCAN_img, GMM_img, DOTIE_sc, GSCE_sc, Kmeans_sc, meanshift_sc, DBSCAN_sc, GMM_sc 

