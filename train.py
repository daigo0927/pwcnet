import os
import argparse
import time
import tensorflow as tf
import numpy as np
from datetime import datetime
from torch.utils import data
from functools import partial
from tqdm import tqdm

from datahandler.flow import get_dataset
from model import PWCDCNet
from losses import L1loss, L2loss, EPE, multiscale_loss, multirobust_loss
from utils import save_config, ExperimentSaver
from flow_utils import vis_flow, vis_flow_pyramid


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        dset = get_dataset(self.args.dataset)
        data_args = {'dataset_dir':self.args.dataset_dir, "origin_size":None,
                     'crop_type':self.args.crop_type, 'crop_shape':self.args.crop_shape,
                     'resize_shape':self.args.resize_shape, 'resize_scale':self.args.resize_scale}
        tset = dset(train_or_val = 'train', **data_args)
        vset = dset(train_or_val = 'val', **data_args)
        self.image_size = tset.image_size

        load_args = {'batch_size': self.args.batch_size, 'num_workers':self.args.num_workers,
                     'drop_last':True, 'pin_memory':True}
        self.num_batches = int(len(tset.samples)/self.args.batch_size)
        print(f'Found {len(tset.samples)} samples -> {self.num_batches} mini-batches')
        self.tloader = data.DataLoader(tset, shuffle = True, **load_args)
        self.vloader = data.DataLoader(vset, shuffle = False, **load_args)
        
    def _build_graph(self):
        # Input images and ground truth optical flow definition
        with tf.name_scope('Data'):
            self.images = tf.placeholder(tf.float32, shape = (self.args.batch_size, 2, *self.image_size, 3),
                                         name = 'images')
            images_0, images_1 = tf.unstack(self.images, axis = 1)
            self.flows_gt = tf.placeholder(tf.float32, shape = (self.args.batch_size, *self.image_size, 2),
                                           name = 'flows')

        # Model inference via PWCNet
        model = PWCDCNet(num_levels = self.args.num_levels,
                         search_range = self.args.search_range,
                         warp_type = self.args.warp_type,
                         use_dc = self.args.use_dc,
                         output_level = self.args.output_level,
                         name = 'pwcdcnet')
        flows_final, self.flows = model(images_0, images_1)
        target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope = 'pwcdcnet/fp_extractor')[::6]
        target_weights += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope = 'pwcdcnet/optflow')[::12]

        # Loss calculation
        with tf.name_scope('Loss'):
            if self.args.loss is 'multiscale':
                criterion = multiscale_loss
            else:
                criterion =\
                  partial(multirobust_loss, epsilon = self.args.epsilon, q = self.args.q)
            
            _loss = criterion(self.flows_gt, self.flows, self.args.weights)
            weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in model.vars])
            loss = _loss + self.args.gamma*weights_l2

            epe = EPE(self.flows_gt, flows_final)

        # Gradient descent optimization
        with tf.name_scope('Optimize'):
            self.global_step = tf.train.get_or_create_global_step()
            if self.args.lr_scheduling:
                boundaries = [200000, 250000, 300000, 350000, 4000000]
                values = [self.args.lr/(2**i) for i in range(len(boundaries)+1)]
                lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
            else:
                lr = self.args.lr

            self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)\
                             .minimize(loss, var_list = model.vars)
            with tf.control_dependencies([self.optimizer]):
                self.optimizer = tf.assign_add(self.global_step, 1)

        # Initialization
        self.saver = tf.train.Saver(model.vars)
        self.sess.run(tf.global_variables_initializer())
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)

        # Summarize
        # Original PWCNet loss
        sum_loss = tf.summary.scalar('loss/pwc', loss)
        # EPE for both domains
        sum_epe = tf.summary.scalar('EPE/source', epe)
        # Merge summaries
        self.merged = tf.summary.merge([sum_loss, sum_epe])

        logdir = 'logs/history_' + datetime.now().strftime('%Y-%m-%d-%H-%M')
        self.twriter = tf.summary.FileWriter(logdir+'/train', graph = self.sess.graph)
        self.vwriter = tf.summary.FileWriter(logdir+'/val', graph = self.sess.graph)

        self.exp_saver = ExperimentSaver(logdir = logdir, parse_args = self.args)

        print(f'Graph building completed, histories are logged in {logdir}')

            
    def train(self):
        for e in tqdm(range(self.args.num_epochs)):
            # Training
            for images, flows_gt in self.tloader:
                images = images.numpy()/255.0
                flows_gt = flows_gt.numpy()

                _, g_step = self.sess.run([self.optimizer, self.global_step],
                                          feed_dict = {self.images: images,
                                                       self.flows_gt: flows_gt})

                if g_step%1000 == 0:
                    summary = self.sess.run(self.merged,
                                            feed_dict = {self.images: images,
                                                         self.flows_gt: flows_gt})
                    self.twriter.add_summary(summary, g_step)

            # Validation
            for images_val, flows_gt_val in self.vloader:
                images_val = images_val.numpy()/255.0
                flows_gt_val = flows_gt_val.numpy()

                summary = self.sess.run(self.merged,
                                        feed_dict = {self.images: images_val,
                                                     self.flows_gt: flows_gt_val})
                self.vwriter.add_summary(summary, g_step)
            # Collect convolution weights and biases
            # summary_plus = self.sess.run(self.merged_plus)
            # self.vwriter.add_summary(summary_plus, g_step)

            # visualize estimated optical flow
            if self.args.visualize:
                if not os.path.exists('./figure'):
                    os.mkdir('./figure')
                # Estimated flow values are downscaled, rescale them compatible to the ground truth
                flow_set = []
                flows_val = self.sess.run(self.flows, feed_dict = {self.images: images_val,
                                                                   self.flows_gt: flows_gt_val})
                for l, flow in enumerate(flows_val):
                    upscale = 20/2**(self.args.num_levels-l)
                    flow_set.append(flow[0]*upscale)
                flow_gt = flows_gt_val[0]
                images_v = images_val[0]
                vis_flow_pyramid(flow_set, flow_gt, images_v,
                                 f'./figure/flow_{str(e+1).zfill(4)}.pdf')

            if not os.path.exists('./model'):
                os.mkdir('./model')
            self.saver.save(self.sess, f'./model/model_{e+1}.ckpt')

        
        self.twriter.close()
        self.vwriter.close()
        self.exp_saver.append(['./figure', './model'])
        self.exp_saver.save()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type = str, default = 'SintelClean',
                        help = 'Target dataset, [SintelClean]')
    parser.add_argument('-dd', '--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    parser.add_argument('-e', '--num_epochs', type = int, default = 100,
                        help = '# of epochs [100]')
    parser.add_argument('-b', '--batch_size', type = int, default = 4,
                        help = 'Batch size [4]')
    parser.add_argument('-nw', '--num_workers', type = int, default = 2,
                        help = '# of workers for data loading [2]')

    parser.add_argument('--crop_type', type = str, default = 'random',
                        help = 'Crop type for raw data [random]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [384, 448],
                        help = 'Crop shape for raw data [384, 448]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = None,
                        help = 'Resize shape for raw data [None]')
    parser.add_argument('--resize_scale', type = float, default = None,
                        help = 'Resize scale for raw data [None]')

    parser.add_argument('--num_levels', type = int, default = 6,
                        help = '# of levels for feature extraction [6]')
    parser.add_argument('--search_range', type = int, default = 4,
                        help = 'Search range for cost-volume calculation [4]')
    parser.add_argument('--warp_type', default = 'bilinear', choices = ['bilinear', 'nearest'],
                        help = 'Warping protocol, [bilinear] or nearest')
    parser.add_argument('--use-dc', dest = 'use_dc', action = 'store_true',
                        help = 'Enable dense connection in optical flow estimator, [diabled] as default')
    parser.add_argument('--no-dc', dest = 'use_dc', action = 'store_false',
                        help = 'Disable dense connection in optical flow estimator, [disabled] as default')
    parser.set_defaults(use_dc = False)
    parser.add_argument('--output_level', type = int, default = 4,
                        help = 'Final output level for estimated flow [4]')

    parser.add_argument('--loss', default = 'multiscale', choices = ['multiscale', 'robust'],
                        help = 'Loss function choice in [multiscale/robust]')
    parser.add_argument('--lr', type = float, default = 1e-4,
                        help = 'Learning rate [1e-4]')
    parser.add_argument('--lr_scheduling', dest = 'lr_scheduling', action = 'store_true',
                        help = 'Enable learning rate scheduling, [enabled] as default')
    parser.add_argument('--no-lr_scheduling', dest = 'lr_scheduling', action = 'store_false',
                        help = 'Disable learning rate scheduling, [enabled] as default')
    parser.set_defaults(lr_scheduling = True)
    parser.add_argument('--weights', nargs = '+', type = float,
                        default = [0.32, 0.08, 0.02, 0.01, 0.005],
                        help = 'Weights for each pyramid loss')
    parser.add_argument('--gamma', type = float, default = 0.0004,
                        help = 'Coefficient for weight decay [4e-4]')
    parser.add_argument('--epsilon', type = float, default = 0.02,
                        help = 'Small constant for robust loss [0.02]')
    parser.add_argument('--q', type = float, default = 0.4,
                        help = 'Tolerance constant for outliear flow [0.4]')

    parser.add_argument('-v', '--visualize', dest = 'visualize', action = 'store_true',
                        help = 'Enable estimated flow visualization, [enabled] as default')
    parser.add_argument('--no-visualize', dest = 'visualize', action = 'store_false',
                        help = 'Disable estimated flow visualization, [enabled] as default')
    parser.set_defaults(visualize = True)
    parser.add_argument('-r', '--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint file [None]')
    
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
