# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
      'num_images ({}) must divide BATCH_SIZE ({})'. \
      format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where(
            roidb[0]['gt_classes'] !=
            0 & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

    if roidb[0]['data_type'] == 'image-level':
        blobs['seed_or_mined'] = 'mined'
        blobs['pred_score'] = roidb[0]['pred_score']
    else:
        blobs['seed_or_mined'] = 'seed'
    
    return blobs


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        
        # 如果是mining照片，且要做blur的話，則除mining的boxes外，做模糊
        #print("cfg.TRAIN.BLUR: ", cfg.TRAIN.BLUR)
        if roidb[i]['data_type'] == 'image-level' and cfg.TRAIN.BLUR != 0:
            blurred_img = cv2.blur(im, (cfg.TRAIN.BLUR, cfg.TRAIN.BLUR))
            mask = np.zeros((roidb[i]['height'], roidb[i]['width'], 3), dtype=np.uint8)
            
            # 掃過所有boxes，在boxes涵蓋的mask區域設為255,255,255
            for j in range(len(roidb[i]['boxes'])):
                (x1, y1) = roidb[i]['boxes'][j][0:2]
                (x2, y2) = roidb[i]['boxes'][j][2:]
                mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
            im = np.where(mask==np.array([255, 255, 255]), im, blurred_img)
        
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales
