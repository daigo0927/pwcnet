import tensorflow as tf
from functools import partial

from utils import get_grid


def _conv_block(filters, kernel_size = (3, 3), strides = (1, 1), batch_norm = False):
    def f(x):
        x = tf.layers.Conv2D(filters, kernel_size,
                             strides, 'same')(x)
        if batch_norm:
            x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x
    return f


class FeaturePyramidExtractor(object):

    def __init__(self, num_levels = 6, batch_norm = False, name = 'fp_extractor'):
        self.num_levels = num_levels
        self.filters_list = [16, 32, 64, 96, 128, 192]
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
                
            feature_pyramid = []
            for l in range(self.num_levels):
                x = _conv_block(self.filters_list[l], (3, 3),
                                (2, 2), self.batch_norm)(x)
                x = _conv_block(self.filters_list[l], (3, 3),
                                (1, 1), self.batch_norm)(x)
                feature_pyramid.append(x)

            # return feature pyramid by ascent order
            return feature_pyramid[::-1] 

class WarpingLayer(object):

    def __init__(self, name = 'warping'):
        self.name = name

    def __call__(self, x, flow):
        # expect shape
        # x:(#batch, height, width, #channel)
        # flow:(#batch, height, width, 2)
        with tf.variable_scope(self.name) as vs:
            grid_b, grid_y, grid_x = get_grid(x)
            flow = tf.cast(flow, tf.int32)
            warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
            warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
            warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
            warped_x = tf.gather_nd(x, warped_indices)
            return warped_x


def pad2d(x, vpad, hpad):
    return tf.pad(x, [[0, 0], vpad, hpad, [0, 0]])

def crop2d(x, vcrop, hcrop):
    return tf.keras.layers.Cropping2D([vcrop, hcrop])(x)

def get_cost(x, warped, shift):
    v, h = shift # vertical/horizontal element
    vt, vb, hl, hr =  max(v,0), abs(min(v,0)), max(h,0), abs(min(h,0)) # top/bottom left/right
    x_pad = pad2d(x, [vt, vb], [hl, hr])
    warped_pad = pad2d(warped, [vb, vt], [hr, hl])
    cost_pad = x_pad*warped_pad
    return tf.reduce_sum(crop2d(cost_pad, [vt, vb], [hl, hr]), axis = 3)
            
class CostVolumeLayer(object):

    def __init__(self, search_range = 4, name = 'cost_volume'):
        self.s_range = search_range
        self.name = name

    def __call__(self, x, warped, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            b, h, w, f = tf.unstack(tf.shape(x))
            cost_length = (2*self.s_range+1)**2

            cost = [0]*cost_length
            cost[0] = tf.reduce_sum(warped*x, axis  = 3)
            I = 1
            get_c = partial(get_cost, x, warped)
            for i in range(1, self.s_range+1):
                cost[I] = get_c(shift = [-i, 0]); I+=1
                cost[I] = get_c(shift = [i, 0]); I+=1
                cost[I] = get_c(shift = [0, -i]); I+=1
                cost[I] = get_c(shift = [0, i]); I+=1

                for j in range(1, self.s_range+1):
                    cost[I] = get_c(shift = [-i, -j]); I+=1
                    cost[I] = get_c(shift = [i, j]); I+=1
                    cost[I] = get_c(shift = [-i, j]); I+=1
                    cost[I] = get_c(shift = [i, -j]); I+=1

            return tf.stack(cost, axis = 3) / cost_length
            
        
class OpticalFlowEstimator(object):

    def __init__(self, batch_norm, name = 'of_estimator'):
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, x, cost, flow, reuse = True):
        with tf.variable_scope(self.name) as vs:
            flow = tf.cast(flow, dtype = tf.float32)
            x = tf.concat([x, cost, flow], axis = 3)
            x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(96, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(64, (3, 3), (1, 1), self.batch_norm)(x)
            feature = _conv_block(32, (3, 3), (1, 1), self.batch_norm)(x)
            flow = tf.layers.Conv2D(2, (3, 3), (1, 1), padding = 'same')(feature)

            return feature, flow # x:processed feature, w:processed flow

    
class ContextNetwork(object):

    def __init__(self, name = 'context'):
        self.name = name

    def __call__(self, feature, flow, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            x = tf.concat([feature, flow], axis = 3)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (2, 2))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (4, 4))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(96, (3, 3), (1, 1),'same',
                                 dilation_rate = (8, 8))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1),'same',
                                 dilation_rate = (16, 16))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(2, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            return x+flow
