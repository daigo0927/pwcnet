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

        load_args = {'batch_size': self.args.batch_size, 'shuffle':True,
                     'num_workers':self.args.num_workers, 'pin_memory':True}
        self.num_batches = int(len(train_dataset.samples)/self.args.batch_size)
        self.train_loader = data.DataLoader(train_dataset, **load_args)
        self.eval_loader = data.DataLoader(eval_dataset, **load_args)
        
    def _build_graph(self):
        self.images_0 = tf.placeholder(tf.float32, shape = [4]+args.image_size+[3],
                                       name = 'images_0')
        self.images_1 = tf.placeholder(tf.float32, shape = [4]+args.image_size+[3],
                                       name = 'images_1')
        self.flows_gt = tf.placeholder(tf.float32, shape = [4]+args.image_size+[2],
                                       name = 'flows')
        self.model = PWCNet(self.args.num_levels, self.args.search_range,
                            self.args.output_level, self.args.batch_norm)
        self.flows_pyramid, self.summaries = self.model(self.images_0, self.images_1)

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
            for i, (images, flows_gt) in enumerate(self.train_loader):
                images = images.numpy()/255.0
                flows_gt = flows_gt.numpy()

                loss, epe, loss_levels, epe_levels \
                    = self.sess.run([self.loss, self.epe, self.loss_levels, self.epe_levels],
                                    feed_dict = {self.images: images,
                                                 self.flows_gt: flows_gt})
                if i%10 == 0:
                    show_progress(e+1, i+1, self.num_batches, loss, epe)

            loss_evals, epe_evals = [], []
            for images_eval, flows_gt_eval in self.eval_loader:
                images_eval = images_eval.numpy()/255.0
                flows_gt_eval = flows_gt_eval.numpy()

                loss_eval, epe_eval \
                    = self.sess.run([self.loss, self.epe],
                                    feed_dict = {self.images: images_eval,
                                                 self.flows_gt: flows_gt_eval})
                loss_evals.append(loss_eval)
                epe_evals.append(epe_eval)
            print(f'\r{e+1} epoch evaluation, loss: {np.mean(loss_evals)}, epe: {np.mean(epe_evals)}')

            if not os.path.exists('./model'):
                os.mkdir('./model')
            self.saver.save(self.sess, f'./model/model_{e}.ckpt')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str, default = 'SintelClean',
                        help = 'Target dataset, [SintelClean]')
    parser.add_argument('--dataset_dir', type = str, required = True,
                        help = 'Directory containing target dataset')
    parser.add_argument('--batch_size', type = int, default = 4,
                        help = 'Batch size [4]')
    parser.add_argument('--num_workers', type = int, default = 8,
                        help = '# worker for data loading [8]')

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
        
    parser.add_argument('--weights', nargs = '+', type = float,
                        default = [0.32, 0.08, 0.02, 0.01, 0.005],
                        help = 'Weights for each pyramid loss')
    parser.add_argument('--gamma', type = float, default = 0.0004,
                        help = 'Coefficient for weight decay')
    parser.add_argument('--epsilon', type = float, default = 0.01,
                        help = 'Small constant for robust loss')
    parser.add_argument('--q', type = float, default = 0.4,
                        help = 'Tolerance constant for outliear flow')

    parser.add_argument('--resume', type = str,
                        help = 'Learned parameter checkpoint file')
    
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    trainer = Trainer(args)
    trainer.train()
