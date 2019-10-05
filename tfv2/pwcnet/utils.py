import json
import sys
import shutil
import numpy as np
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser


def show_progress(epoch, batch, batch_total, **kwargs):
    message = f'\r{epoch} epoch: [{batch}/{batch_total}'
    for key, item in kwargs.items():
        message += f', {key}: {item}'
    sys.stdout.write(message+']')
    sys.stdout.flush()


def save_config(config, filename = None):
    if not isinstance(config, (dict, OrderedDict)):
        raise TypeError('arg config must be a dict or OrderedDict')
    config = OrderedDict(config)

    if filename is None:
        filename = 'config_' + datetime.now().strftime('%Y-%m-%d-%H-%M') + '.json'

    with open(filename, 'w') as f:
        json.dump(config, f, indent = 4)
    print(f'Given config has been successfully saved to {filename}.')


def makeColorwheel():

    #  color encoding scheme
    
    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3]) # r g b

    col = 0
    #RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY

    #YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255;
    col += YG;

    #GC
    colorwheel[col:GC+col, 1]= 255 
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC;

    #CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB;

    #BM
    colorwheel[col:BM+col, 2]= 255 
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM;

    #MR
    colorwheel[col:MR+col, 2] = 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return colorwheel

def computeColor(u, v):

    colorwheel = makeColorwheel();
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v) 

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0 
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1) # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
    k1 = k0+1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1],3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:,i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx]*(1-col[idx]) # increase saturation with radius    
        col[~idx] *= 0.75 # out of range
        img[:,:,2-i] = np.floor(255*col).astype(np.uint8)

    return img.astype(np.uint8)


def vis_flow(flow):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10
    
    u = flow[:,:,0]
    v = flow[:,:,1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    #fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0 
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u,u)+np.multiply(v,v))
    maxrad = max([maxrad, np.amax(rad)])
 
    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    img = computeColor(u, v)
    return img[:,:,[2,1,0]]


def prepare_parser():
    parser = ArgumentParser(description='Training config')
    # Dataset
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Target dataset name')
    parser.add_argument('-dd', '--dataset_dir', type=str, required=True,
                        help='Target dataset directory (required)')
    parser.add_argument('--mode', type=str, default='clearn',
                        help='Target path (only required for MPI-Sintel dataset')
    # Iteration config
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of epochs [100]')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Batch size [16]')
    parser.add_argument('-v', '--validation_split', type=float, default=0.1,
                        help='Validtion split ratio [0.1]')
    parser.add_argument('--debug', action='store_true',
                        help='Debug execution')
    # Data pipeline configs
    parser.add_argument('--random_scale', type=float, default=0.8,
                        help='Random scale [0.8]')
    parser.add_argument('--crop_shape', nargs=2, type=int, default=[384, 448],
                        help='Crop shape for images. [384, 448]')
    parser.add_argument('-hflip', '--horizontal_flip', action='store_true',
                        help='Enable left-right flip in preprocessing')
    # Learning configs
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate [0.001]')
    parser.add_argument('--gamma', type=float, default=0.0004,
                        help='Weight decay coefficient [0.0004]')
    return parser

