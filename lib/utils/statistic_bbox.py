import numpy as np

def statistic_bbox(dic, dic_im):
    """ Statistic number of bbox of seed and image-level data for each class
    Parameters
    ----------
    dic: seed roidb dictionary
    dic_im: image-level roidb dictionary
    
    Returns
    -------
    num_bbox: list for number of 20 class's bbox
    num_bbox_im: list for number of 20 class's bbox
    """
    num_bbox = [0] * 20
    num_bbox_im = [0] * 20

    for d in dic:
        for c in d['gt_classes']:
            num_bbox[c-1] += 1

    for d in dic_im:
        for c in d['gt_classes']:
            num_bbox_im[c-1] += 1
    
    print("Statistic for seed data bbox: ", num_bbox)
    print("Statistic for image-level data bbox: ", num_bbox_im)
    
    return num_bbox, num_bbox_im