import os
import re
import argparse
import numpy as np
import tensorflow as tf
import imageio

from model import PWCNet
from flow_utils import vis_flow_pyramid

class Tester(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        img1_path, img2_path = self.args.input_images
        img1, img2 = map(imageio.imread, (img1_path, img2_path))
        self.images = np.array([img1, img2])/255.0 # shape(2, h, w, 3)
        self.images_tf = tf.expand_dims(tf.convert_to_tensor(self.images, dtype = tf.float32),
                                        axis = 0) # shape(1, 2, h, w, 3)

        self.model = PWCNet()
        self.finalflow, self.flow_pyramid, _ \
          = self.model(self.images_tf[:,0], self.images_tf[:,1])

        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            print('!!! Test with un-learned model !!!')
            self.sess.run(tf.global_variables_initializer())

    def test(self):
        flow_pyramid = self.sess.run(self.flow_pyramid)
        flow_pyramid = [fpy[0] for fpy in flow_pyramid]
        if not os.path.exists('./test_figure'):
            os.mkdir('./test_figure')
        fname = '_'.join(re.split('[/.]', self.args.input_images[0])[-3:-1])
        vis_flow_pyramid(flow_pyramid, images = self.images, filename = f'./test_figure/test_{fname}.pdf')
        print('Figure saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images', type = str, nargs = 2, required = True,
                        help = 'Target images (required)')
    parser.add_argument('--resume', type =  str, default = None,
                        help = 'Learned parameter checkpoint file [None]')
    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    tester = Tester(args)
    tester.test()
