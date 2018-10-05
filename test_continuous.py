import os
import re
import time
from tqdm import tqdm
import argparse
import numpy as np
import tensorflow as tf
import imageio

from model import PWCDCNet
from flow_utils import vis_flow_pyramid

def factor_crop(image, factor = 64):
    assert image.ndim == 3
    h, w, _ = image.shape
    image = image[:factor*(h//factor), :factor*(w//factor)]
    return image

class Tester(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        self.images = tf.placeholder(tf.float32, shape = (1, 2, None, None, 3))
        
        self.model = PWCDCNet()
        self.flow_final, self.flows \
          = self.model(self.images[:,0], self.images[:,1])

        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            print('!!! Test with un-learned model !!!')
            self.sess.run(tf.global_variables_initializer())

    def test(self):
        if not os.path.exists('./test_figure'):
            os.mkdir('./test_figure')
            
        image_path_pairs = zip(self.args.input_images[:-1], self.args.input_images[1:])
        for img1_path, img2_path in tqdm(image_path_pairs, desc = 'Processing'):
            images = list(map(imageio.imread, (img1_path, img2_path)))
            images = list(map(factor_crop, images))
            images = np.array(images)/255.
            images_expand = np.expand_dims(images, axis = 0)

            flows = self.sess.run(self.flows, feed_dict = {self.images: images_expand})
            
            flow_set = []
            for l, flow in enumerate(flows):
                upscale = 20/2**(self.model.num_levels-l)
                flow_set.append(flow[0]*upscale)

            dname, fname = re.split('[/.]', img1_path)[-3:-1]
            if not os.path.exists(f'./test_figure/{dname}'):
                os.mkdir(f'./test_figure/{dname}')
            vis_flow_pyramid(flow_set, images = images,
                             filename = f'./test_figure/{dname}/{fname}.png')
        print('Figure saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_images', type = str, nargs = '+', required = True,
                        help = 'Target images (required)')
    parser.add_argument('-r', '--resume', type =  str, default = None,
                        help = 'Learned parameter checkpoint file [None]')
    args = parser.parse_args()

    # Expand wild-card
    if '*' in args.input_images:
        from glob import glob
        args.input_images = glob(args.input_images)
    if len(args.input_images) < 2:
        raise ValueError('# of input images must be >= 2')
        
    print(args.resume)
    for i, image in enumerate(args.input_images):
        print(image)
        if i == 5:
            print(f'... and more ({len(args.input_images)} images)')
            break

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')

    tester = Tester(args)
    tester.test()
