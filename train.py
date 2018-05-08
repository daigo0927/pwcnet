import argparse
import tensorflow as tf
import numpy as np
import torch
from torch.utils import data

from model import PWCNet
from dataset import get_dataset
from losses import L1loss, L2loss, EPE, multiscale_loss, multirobust_loss


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.sess = tf.Session()
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        dataset = get_dataset(self.args.dataset)
        data_args = {'dataset_dir':self.dataset_dir, 'cropper':self.crop_type,
                    'crop_shape':self.crop_shape, 'resize_shape':self.resize_shape,
                    'crop_scale':self.resize_scale}
        train_dataset = dataset(train_or_test = 'train', **dataargs)
        eval_dataset = dataset(train_or_test = 'test', **dataargs)

        load_args = {'batch_size': self.args.batch_size, 'shuffle':True,
                     'num_workers':args.num_workers, 'pin_memory':True}
        self.num_batches = int(len(train_dataset.samples)/self.args.batch_size)
        self.train_loader = data.DataLoader(train_dataset, **load_args)
        self.eval_loader = data.DataLoader(eval_dataset, **load_args)
        
    def _build_graph(self):
        self.images = tf.placeholder(tf.float32, shape = [None, 2]+args.image_size+[3],
                                     name = 'images')
        self.flows_gt = tf.placeholder(tf.float32, shape = [None]+args.image_size+[2],
                                       name = 'flows')
        self.model = PWCNet(self.args.num_levels, self.args.search_range,
                            self.args.output_level, self.args.batch_norm)
        self.flows_pyramid, self.summaries = self.model(self.images[:,0], self.images[:,1])

        weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.model.vars])
        self.loss, self.epe, self.loss_levels, self.epe_levels \
            = multiscale_loss(self.flows_gt, self.flows_pyramid, self.args.weights)\
            + self.args.gamma * weights_l2
        self.optimizer = tf.train.AdamOptimizer()\
                         .minimize(self.loss, var_list = self.model.vars)

        self.saver = tf.train.Saver()

        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())
            
    def train(self):
        for e in range(self.args.n_epoch):
            for i, (images, flows_gt) in enumerate(self.trainloader):
                pass
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'SintelClean',
                        help = 'Target dataset, default: SintelClean')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')

    parser.add_argument('--output_level', type = float, default = 4,
                        help = 'Final output level for estimated flow')
    parser.add_argument('--search_range', type = int, default = 4,
                        help = 'Search range for cost-volume calculation')
    
    parser.add_argument('--weights', nargs = '+', type = float,
                        default = [0.32, 0.08, 0.02, 0.01, 0.005],
                        help = 'Weights for each pyramid loss')
    parser.add_argument('--gamma', type = float, default = 0.0004,
                        help = 'Coefficient for weight decay')
    parser.add_argument('--epsilon', type - float, default = 0.01,
                        help = 'Small constant for robust loss')
    parser.add_argument('--q', type = float, default = 0.4,
                        help = 'Tolerance constant for outliear flow')
    
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
