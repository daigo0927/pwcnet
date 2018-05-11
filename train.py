import os
import argparse
import tensorflow as tf
import numpy as np
import torch
from torch.utils import data

from model import PWCNet
from dataset import get_dataset
from losses import L1loss, L2loss, EPE, multiscale_loss, multirobust_loss
from utils import show_progress
from flow_utils import vis_flow_pyramid


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.sess = tf.Session()
        self._build_dataloader()
        self._build_graph()

    def _build_dataloader(self):
        dataset = get_dataset(self.args.dataset)
        data_args = {'dataset_dir':self.args.dataset_dir, 'cropper':self.args.crop_type,
                    'crop_shape':self.args.crop_shape, 'resize_shape':self.args.resize_shape,
                    'resize_scale':self.args.resize_scale}
        train_dataset = dataset(train_or_test = 'train', **data_args)
        eval_dataset = dataset(train_or_test = 'test', **data_args)

        load_args = {'batch_size': self.args.batch_size,
                     'num_workers':self.args.num_workers, 'pin_memory':True}
        self.num_batches = int(len(train_dataset.samples)/self.args.batch_size)
        self.train_loader = data.DataLoader(train_dataset, shuffle = True, **load_args)
        self.eval_loader = data.DataLoader(eval_dataset, shuffle = False, **load_args)
        
    def _build_graph(self):
        self.images = tf.placeholder(tf.float32, shape = [None, 2]+args.image_size+[3],
                                     name = 'images')
        self.flows_gt = tf.placeholder(tf.float64, shape = [None]+args.image_size+[2],
                                       name = 'flows')
        self.model = PWCNet(self.args.num_levels, self.args.search_range,
                            self.args.output_level, self.args.batch_norm,
                            self.args.context)
        self.finalflow, self.flows_pyramid, self.summaries \
            = self.model(self.images[:,0], self.images[:,1])

        if self.args.loss is 'mutiscale':
            self.criterion = mutiscale_loss
        else:
            self.criterion = multirobust_loss
            
        self.loss, self.epe, self.loss_levels, self.epe_levels \
            = self.criterion(self.flows_gt, self.flows_pyramid, self.args.weights)
        weights_l2 = tf.reduce_sum([tf.nn.l2_loss(var) for var in self.model.vars])
        self.loss_reg = self.loss + self.args.gamma*weights_l2
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.args.lr)\
                         .minimize(self.loss_reg, var_list = self.model.vars)
        self.saver = tf.train.Saver()

        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            self.sess.run(tf.global_variables_initializer())
            
    def train(self):
        for e in range(self.args.n_epoch):
            for i, (images, flows_gt) in enumerate(self.train_loader):
                images = images.numpy()/255.0
                flows_gt = flows_gt.numpy()

                _, loss_reg, epe \
                    = self.sess.run([self.optimizer, self.loss_reg, self.epe],
                                    feed_dict = {self.images: images,
                                                 self.flows_gt: flows_gt})

                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, loss_reg, epe)

            loss_evals, epe_evals = [], []
            for images_eval, flows_gt_eval in self.eval_loader:
                images_eval = images_eval.numpy()/255.0
                flows_gt_eval = flows_gt_eval.numpy()

                flows_pyramid, loss_eval, epe_eval \
                    = self.sess.run([self.flows_pyramid, self.loss_reg, self.epe],
                                    feed_dict = {self.images: images_eval,
                                                 self.flows_gt: flows_gt_eval})
                loss_evals.append(loss_eval)
                epe_evals.append(epe_eval)
            print(f'\r{e+1} epoch evaluation, loss: {np.mean(loss_evals)}, epe: {np.mean(epe_evals)}')
            
            # visualize estimated optical flow
            if self.args.visualize:
                if not os.path.exists('./figure'):
                    os.mkdir('./figure')
                flow_pyramid = [f_py[0] for f_py in flows_pyramid]
                flow_gt = flows_gt_eval[0]
                image = images_eval[0, 0]*255
                vis_flow_pyramid(flow_pyramid, flow_gt, image,
                                 f'./figure/flow_{str(e).zfill(4)}.pdf')

            if not os.path.exists('./model'):
                os.mkdir('./model')
            self.saver.save(self.sess, f'./model/model_{e}.ckpt')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'SintelClean',
                        help = 'Target dataset, [SintelClean]')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    parser.add_argument('--n_epoch', type = int, default = 100,
                        help = '# of epochs [100]')
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'Batch size [4]')
    parser.add_argument('--num_workers', type = int, default = 8,
                        help = '# of workers for data loading [8]')

    parser.add_argument('--crop_type', type = str, default = 'random',
                        help = 'Crop type for raw data [random]')
    parser.add_argument('--crop_shape', nargs = 2, type = int, default = [384, 448],
                        help = 'Crop shape for raw data [384, 448]')
    parser.add_argument('--resize_shape', nargs = 2, type = int, default = None,
                        help = 'Resize shape for raw data [None]')
    parser.add_argument('--resize_scale', type = float, default = None,
                        help = 'Resize scale for raw data [None]')
    parser.add_argument('--image_size', nargs = 2, type = int, default = [384, 448],
                        help = 'Image size to be processed [384, 448]')

    parser.add_argument('--num_levels', type = int, default = 6,
                        help = '# of levels for feature extraction [6]')
    parser.add_argument('--output_level', type = int, default = 4,
                        help = 'Final output level for estimated flow')
    parser.add_argument('--search_range', type = int, default = 4,
                        help = 'Search range for cost-volume calculation')
    parser.add_argument('--batch_norm', type = str, default = False,
                        help = 'Whether utilize batchnormalization [False]')
    parser.add_argument('--context', default = 'all', choices = ['all', 'final'],
                        help = 'How insert context network [all/final]')

    parser.add_argument('--loss', default = 'multiscale', choices = ['multiscale', 'robust'],
                        help = 'Loss function choice in [multiscale/robust]')
    parser.add_argument('--lr', type = float, default = 1e-4,
                        help = 'Learning rate [1e-4]')
    parser.add_argument('--weights', nargs = '+', type = float,
                        default = [0.32, 0.08, 0.02, 0.01, 0.005],
                        help = 'Weights for each pyramid loss')
    parser.add_argument('--gamma', type = float, default = 0.0004,
                        help = 'Coefficient for weight decay [4e-4]')
    parser.add_argument('--epsilon', type = float, default = 0.02,
                        help = 'Small constant for robust loss [0.02]')
    parser.add_argument('--q', type = float, default = 0.4,
                        help = 'Tolerance constant for outliear flow [0.4]')

    parser.add_argument('--visualize', action = 'store_true',
                        help = 'Stored option for visualize and estimated flow')
    parser.add_argument('--resume', type = str, default = None,
                        help = 'Learned parameter checkpoint file [None]')
    
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
