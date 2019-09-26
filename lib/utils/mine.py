from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from copy import deepcopy
import pickle
import numpy as np

from torchvision.ops import nms
import torch
import cv2

from model.test import im_detect

def mine(roidb_im, model_iters, tag, net, mine_threshold, classes):
    save_path = './mined_pseudo/' + tag + '_' + str(model_iters)
    file_path = os.path.join(save_path, 'roidb_im_pred')
    
    if not os.path.exists('./mined_pseudo'):
        os.mkdir('./mined_pseudo')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if os.path.exists(file_path):
        with open (file_path, 'rb') as fp:
            roidb_im_pred = pickle.load(fp)
        print("Restore mined data...")
        return roidb_im_pred

    roidb_im_pred = deepcopy(roidb_im)
    
    for j in range(len(roidb_im_pred)):
        # keep the ground truth annotations and boxes
        roidb_im_pred[j]['gt_boxes'] = roidb_im_pred[j]['boxes']
        roidb_im_pred[j]['real_gt_classes'] = roidb_im_pred[j]['gt_classes']
        
        # pop bbox annotation
        roidb_im_pred[j].pop('boxes')
        roidb_im_pred[j].pop('gt_classes')
        roidb_im_pred[j].pop('max_classes')
        roidb_im_pred[j].pop('max_overlaps')
        roidb_im_pred[j].pop('gt_overlaps')
        if 'seg_areas' in roidb_im_pred:
            roidb_im_pred[j].pop('seg_areas')

    # 計算images總張數 (不包含Filpped照片)
    num_image = int(len(roidb_im_pred)/2)
    
    for i in range(num_image): # loop for every image

        if (i+1) % 100 == 0:
            print("processing...", i+1, '/', num_image)
        
        im = cv2.imread(roidb_im_pred[i]['image'])
        scores, boxes = im_detect(net, im)
        
        ### mine for each class
            # mining process of NOTERCNN: 
                # condiction 1: predicted label matches with the image-level groundtruth label;
                # condiction 2: the box’s confidence score is the highest among all boxes with the same label
                # condiction 3: confidence score is higher than a threshold (0.99)
        CONF_THRESH = mine_threshold
        NMS_THRESH = 0.3
        
        for cls_ind, cls in enumerate(classes[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]

            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            
            keep = nms(torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores), NMS_THRESH)
            dets = dets[keep.numpy(), :]
            
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]  # condiction 3
            if len(inds) == 0:
                continue;
            
            bboxes = dets[inds, :4]
            pred_scores = dets[inds, -1]
            
            for bbox, score in zip(bboxes, pred_scores):
                if cls_ind in roidb_im_pred[i]['image-label']: #condiction 1
                    
                    # flip bbox
                    oldx1 = np.array([bbox]).astype('int32')[0][0]
                    oldx2 = np.array([bbox]).astype('int32')[0][2]
                    y1 = np.array([bbox]).astype('int32')[0][1]
                    y2 = np.array([bbox]).astype('int32')[0][3]
                    x1 = roidb_im_pred[i]['width'] - oldx2 - 1
                    x2 = roidb_im_pred[i]['width'] - oldx1 - 1
                    new_box = np.array([x1, y1, x2, y2]).astype('int32')

                    if 'gt_classes' in roidb_im_pred[i].keys():
                        roidb_im_pred[i]['gt_classes'] = np.concatenate((roidb_im_pred[i]['gt_classes'], 
                                                                         np.array([cls_ind]).astype('int32')), axis=0)
                        roidb_im_pred[i]['boxes'] = np.concatenate((roidb_im_pred[i]['boxes'], 
                                                                    np.array([bbox]).astype('int32')), axis=0)
                        roidb_im_pred[i]['max_overlaps'] = np.concatenate((roidb_im_pred[i]['max_overlaps'], 
                                                                           np.array([1.]).astype('float32')), axis=0)
                        roidb_im_pred[i]['pred_score'] = np.concatenate((roidb_im_pred[i]['pred_score'], 
                                                                           np.array([score]).astype('float32')), axis=0)
                        
                        # for flipped images
                        roidb_im_pred[i+num_image]['gt_classes'] = np.concatenate((roidb_im_pred[i+num_image]['gt_classes'], 
                                                                         np.array([cls_ind]).astype('int32')), axis=0)
                        roidb_im_pred[i+num_image]['boxes'] = np.concatenate((roidb_im_pred[i+num_image]['boxes'], 
                                                                    np.array([new_box]).astype('int32')), axis=0)
                        roidb_im_pred[i+num_image]['max_overlaps'] = np.concatenate((roidb_im_pred[i+num_image]['max_overlaps'], 
                                                                           np.array([1.]).astype('float32')), axis=0)
                        roidb_im_pred[i+num_image]['pred_score'] = np.concatenate((roidb_im_pred[i+num_image]['pred_score'], 
                                                                           np.array([score]).astype('float32')), axis=0)
                    else:
                        roidb_im_pred[i]['boxes'] = np.array([bbox]).astype('int32')
                        roidb_im_pred[i]['gt_classes'] = np.array([cls_ind]).astype('int32')
                        roidb_im_pred[i]['max_overlaps'] = np.array([1.]).astype('float32')
                        roidb_im_pred[i]['pred_score'] = np.array([score]).astype('float32')

                        # for flipped images
                        roidb_im_pred[i+num_image]['boxes'] = np.array([new_box]).astype('int32')
                        roidb_im_pred[i+num_image]['gt_classes'] = np.array([cls_ind]).astype('int32')
                        roidb_im_pred[i+num_image]['max_overlaps'] = np.array([1.]).astype('float32')
                        roidb_im_pred[i+num_image]['pred_score'] = np.array([score]).astype('float32')

    # 去除沒有pseudo gt的照片
    temp = []
    for i in range(len(roidb_im_pred)):
        if 'gt_classes' in roidb_im_pred[i].keys():
            temp.append(roidb_im_pred[i])
    roidb_im_pred = temp

    with open(file_path, 'wb') as fp:
        pickle.dump(roidb_im_pred, fp)
    
    return roidb_im_pred