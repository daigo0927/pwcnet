import argparse
import tensorflow as tf
import numpy as np
import torch
from torch.utils import data

from model import PWCNet
from dataset import get_loader


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.sess = tf.Session()
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        pass

    def _build_graph(self):
        pass

    def train(self):
        for e in range(self.args.n_epoch):
            for i, (images, flows_gt) in enumerate(self.trainloader):
                pass
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'SintelClean',
                        help = 'Target dataset, default: SintelClean]')
    parser.add_argument('dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
