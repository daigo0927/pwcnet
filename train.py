import os
import argparse
import time
import tensorflow as tf
import numpy as np
from functools import partial

from model import PWCDCNet
from dataset_tf.flow import get_dataset
from losses import L1loss, L2loss, EPE, multiscale_loss, multirobust_loss
from utils import show_progress
from flow_utils import vis_flow_pyramid


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        pipe = get_dataset(self.args.dataset)
        shared_keys = ['dataset_dir', 'crop_type', 'crop_shape', 'resize_shape',
                       'batch_size', 'num_parallel_calls']
        shared_args = {}
        for key in shared_keys:
            shared_args[key] = getattr(self.args, key)

        with tf.name_scope('IO'):
            tset = pipe(train_or_val = 'train', shuffle = True, **shared_args)
            titer = tset.make_one_shot_iterator()
            self.images, self.flows_gt = titer.get_next()
            self.images.set_shape((self.args.batch_size, 2, *self.args.crop_shape, 3))
            self.flows_gt.set_shape((self.args.batch_size, *self.args.crop_shape, 2))
            
            vset = pipe(train_or_val = 'val', shuffle = False, **shared_args)
            viter = vset.make_one_shot_iterator()
            self.images_v, self.flows_gt_v = viter.get_next()
            self.images_v.set_shape((self.args.batch_size, 2, *self.args.crop_shape, 3))
            self.flows_gt_v.set_shape((self.args.batch_size, *self.args.crop_shape, 2))

            self.num_batches = len(tset.samples)//self.args.batch_size
            self.num_batches_v = len(vset.samples)//self.args.batch_size
        
        with tf.name_scope('Forward'):
            model = PWCDCNet(num_levels = self.args.num_levels,
                             search_range = self.args.search_range,
                             warp_type = self.args.warp_type,
                             use_dc = self.args.use_dc,
                             output_level = self.args.output_level,
                             name = 'pwcdcnet')
            self.flows_final, self.flows = model(self.images[:,0], self.images[:,1])
            self.flows_final_v, self.flows_v \
                = model(self.images_v[:,0], self.images_v[:,1], reuse = True)

        with tf.name_scope('Loss'):
            if self.args.loss == 'multiscale':
                criterion = multiscale_loss
            else:
                criterion =\
                  partial(multirobust_loss, epsilon = self.args.epsilon, q = self.args.q)
            
            _loss = criterion(self.flows_gt, self.flows, self.args.weights)
            _loss_v = criterion(self.flows_gt_v, self.flows_v, self.args.weights)
            weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in model.vars])
            self.loss = _loss + self.args.gamma*weights_l2
            self.loss_v = _loss_v + self.args.gamma*weights_l2

            self.epe = EPE(self.flows_gt, self.flows_final)
            self.epe_v = EPE(self.flows_gt_v, self.flows_final_v)

        with tf.name_scope('Optimize'):
            if self.args.lr_scheduling:
                self.global_step = tf.train.get_or_create_global_step()
                boundaries = [200000, 400000, 600000, 800000, 1000000]
                values = [self.args.lr/(2**i) for i in range(len(boundaries)+1)]
                lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
            else:
                self.global_step = tf.constant(0)
                lr = self.args.lr

            self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)\
                             .minimize(self.loss, var_list = model.vars)
            with tf.control_dependencies([self.optimizer]):
                self.optimizer = tf.assign_add(self.global_step, 1)
            
        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())            
                    
    def train(self):
        train_start = time.time()
        for e in range(self.args.num_epochs):
            for i in range(self.num_batches):
                time_s = time.time()
                _, loss, epe = self.sess.run([self.optimizer, self.loss, self.epe])

                if i%20 == 0:
                    batch_time = time.time() - time_s
                    kwargs = {'loss':loss, 'epe':epe, 'batch time':batch_time}
                    show_progress(e+1, i+1, self.num_batches, **kwargs)

            loss_vals, epe_vals = [], []
            for i in range(self.num_batches_v):
                flows_val, loss_val, epe_val \
                    = self.sess.run([self.flows_v, self.loss_v, self.epe_v])

                loss_vals.append(loss_val)
                epe_vals.append(epe_val)
                
            g_step = self.sess.run(self.global_step)
            print(f'\r{e+1} epoch validation, loss: {np.mean(loss_vals)}, epe: {np.mean(epe_vals)}'\
                  +f', global step: {g_step}, elapsed time: {time.time()-train_start} sec.')
            
            # visualize estimated optical flow
            if self.args.visualize:
                if not os.path.exists('./figure'):
                    os.mkdir('./figure')
                # Estimated flow values are downscaled, rescale them compatible to the ground truth
                flow_set = []
                for l, flow in enumerate(flows):
                    upscale = 20/2**(self.args.num_levels-l)
                    flow_set.append(flow[0]*upscale)
                flow_gt = flows_gt_val[0]
                images_v = images_val[0]
                vis_flow_pyramid(flow_set, flow_gt, images_v,
                                 f'./figure/flow_{str(e+1).zfill(4)}.pdf')

            if not os.path.exists('./model'):
                os.mkdir('./model')
            self.saver.save(self.sess, f'./model/model_{e+1}.ckpt')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'SintelClean',
                        help = 'Target dataset, [SintelClean]')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    parser.add_argument('--num_epochs', type = int, default = 100,
                        help = '# of epochs [100]')
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'Batch size [4]')
    parser.add_argument('--num_parallel_calls', type = int, default = 2,
                        help = '# of parallel calls for data loading [2]')

    parser.add_argument('--crop_type', type = str, default = 'random',
                        help = 'Crop type for raw data [random]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [384, 448],
                        help = 'Crop shape for raw data [384, 448]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = None,
                        help = 'Resize shape for raw data [None]')

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
    parser.add_argument('--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint file [None]')
    
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    # os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    trainer = Trainer(args)
    trainer.train()
