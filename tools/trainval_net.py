# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

from utils.statistic_bbox import statistic_bbox
from utils.mine import mine

def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--weight',
        dest='weight',
        help='initialize with pretrained model weights',
        type=str)
    parser.add_argument(
        '--imdb',
        dest='imdb_name',
        help='dataset to train on',
        default='voc_2007_trainval',
        type=str)
    parser.add_argument(
        '--imdbval',
        dest='imdbval_name',
        help='dataset to validate on',
        default='voc_2007_test',
        type=str)
    parser.add_argument(
        '--epochs',
        dest='epochs',
        help='number of epochs to train',
        default=20,
        type=int)
    parser.add_argument(
        '--model_iters',
        dest='model_iters',
        help='number of model_iters to train',
        default=5,
        type=int)
    parser.add_argument(
        '--seedpercen',
        dest='seed_percentage',
        help='number of seed image percentage',
        default=20,
        type=int)
    parser.add_argument(
        '--t_mine',
        dest='t_mine',
        help='threshold for mining pseudo',
        default=0.99,
        type=float)
    parser.add_argument(
        '--tag', dest='tag', help='tag of the model', default=None, type=str)
    parser.add_argument(
        '--blur', dest='blur', default=0, type=int)
    parser.add_argument(
        '--net',
        dest='net',
        help='vgg16, res50, res101, res152, mobile',
        default='res50',
        type=str)
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    """
  Combine multiple roidbs
  """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
        
    # blur在mining外的區域 or not
    cfg.TRAIN.BLUR = args.blur
    
    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set
    imdb, roidb = combined_roidb(args.imdb_name)
    print('{:d} roidb entries'.format(len(roidb)))

    # split seed data and image-level data
    roidb_noflip = roidb[0:int(len(roidb)/2)]
    roidb_flip = roidb[int(len(roidb)/2):]
    index = np.random.permutation(len(roidb_noflip))
    index_seed = sorted(index[0:int(len(roidb_noflip) * args.seed_percentage / 100)])
    index_im = sorted(index[int(len(roidb_noflip) * args.seed_percentage / 100):])
    roidb_im = list(np.array(roidb_noflip)[index_im]) + list(np.array(roidb_flip)[index_im])
    roidb = list(np.array(roidb_noflip)[index_seed]) + list(np.array(roidb_flip)[index_seed])
    print('{:d} seed roidb entries'.format(len(roidb)))
    print('{:d} image-level roidb entries'.format(len(roidb_im)))
    statistic_bbox(roidb, roidb_im)

    # add data_type to roidb and roidb_im
    # set roidb data_type = 'seed'
    for i in range(len(roidb)):
        roidb[i]['data_type'] = 'seed'
    # set roidb_im data_type = 'image-level' and add image-level labels
    for i in range(len(roidb_im)):
        roidb_im[i]['data_type'] = 'image-level'
        roidb_im[i]['image-label'] = np.unique(roidb_im[i]['gt_classes'])

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = combined_roidb(args.imdbval_name)
    print('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip
    # set roidb data_type = 'seed'
    for i in range(len(valroidb)):
        valroidb[i]['data_type'] = 'seed'

    # load network
    if args.net == 'vgg16':
        net = vgg16()
    elif args.net == 'res50':
        net = resnetv1(num_layers=50)
    elif args.net == 'res101':
        net = resnetv1(num_layers=101)
    elif args.net == 'res152':
        net = resnetv1(num_layers=152)
    elif args.net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    # Train the model 0 ~ N (args.model_iters)
    roidb_im_pred = []
    steps_for_stage = []
    total_step = 0
    
    for i in range(args.model_iters):
        if i == 0:
            iters = len(roidb) * args.epochs
            roidb_final = roidb
        else:
            iters = (len(roidb_im_pred) + len(roidb)) * args.epochs
            roidb_final = roidb + roidb_im_pred
        print("len(roidb): ", len(roidb))    
        print("len(roidb_im_pred): ", len(roidb_im_pred))
        print("len(roidb_final): ", len(roidb_final))
        total_step += iters
        
        steps_for_stage.append((i, total_step))
        with open(output_dir + '/steps_for_stage.txt', 'w') as fp:
            fp.write("iter, step\n")
            for i, s in steps_for_stage:
                fp.write("%s, %s\n" % (i, s))
                
        print("=== start training model {}, steps: {} ===".format(i, iters))
        
        train_net(
            net,
            imdb,
            roidb_final,
            valroidb,
            output_dir,
            tb_dir,
            pretrained_model=args.weight,
            max_iters=total_step)
        
        roidb_im_pred = mine(roidb_im, model_iters=i, tag=args.tag, net=net, mine_threshold=args.t_mine, classes=imdb._classes)