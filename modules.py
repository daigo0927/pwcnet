import numpy as np
import tensorflow as tf
from functools import partial

from utils import get_grid


# Feature pyramid extractor module simple/original -----------------------
class FeaturePyramidExtractor(object):
    def __init__(self, num_levels = 6, name = 'fp_extractor'):
        self.num_levels = num_levels
        self.filters = [16, 32, 64, 96, 128, 192]
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
                
            feature_pyramid = []
            for l in range(self.num_levels):
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                feature_pyramid.append(x)
                
            # return feature pyramid by ascent order
            return feature_pyramid[::-1]

        
class FeaturePyramidExtractor_custom(object):
    def __init__(self, num_levels = 6, name = 'fp_extractor'):
        self.num_levels = num_levels
        self.filters = [16, 32, 64, 96, 128, 192]
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
                
            feature_pyramid = []
            for l in range(self.num_levels):
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                feature_pyramid.append(x)
                
            # return feature pyramid by ascent order
            return feature_pyramid[::-1]


# Warping layer ---------------------------------
def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

class WarpingLayer(object):
    def __init__(self, warp_type = 'nearest', name = 'warping'):
        self.warp = warp_type
        self.name = name

    def __call__(self, x, flow):
        # expect shape
        # x:(#batch, height, width, #channel)
        # flow:(#batch, height, width, 2)
        with tf.name_scope(self.name) as ns:
            assert self.warp in ['nearest', 'bilinear']
            if self.warp == 'nearest':
                x_warped = nearest_warp(x, flow)
            else:
                x_warped = bilinear_warp(x, flow)
            return x_warped


# Cost volume layer -------------------------------------
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

    def __call__(self, x, warped):
        with tf.name_scope(self.name) as ns:
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

            cost = tf.stack(cost, axis = 3)/cost_length
            cost = tf.nn.leaky_relu(cost, 0.1)
            return cost
            

# Optical flow estimator module simple/original -----------------------------------------
class OpticalFlowEstimator(object):
    def __init__(self, name = 'of_estimator'):
        self.batch_norm = False
        self.name = name

    def __call__(self, cost, x, flow):
        with tf.variable_scope(self.name) as vs:
            flow = tf.cast(flow, dtype = tf.float32)
            x = tf.concat([cost, x, flow], axis = 3)
            x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(96, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(64, (3, 3), (1, 1), self.batch_norm)(x)
            feature = _conv_block(32, (3, 3), (1, 1), self.batch_norm)(x)
            flow = tf.layers.Conv2D(2, (3, 3), (1, 1), padding = 'same')(feature)

            return feature, flow # x:processed feature, w:processed flow


class OpticalFlowEstimator_custom(object):
    def __init__(self, name = 'of_estimator'):
        self.name = name

    def __call__(self, cost, feature_0 = None, flow_up = None, feature_up = None,
                 is_output = False):
        with tf.variable_scope(self.name) as vs:
            if feature_0 is None and flow_up is None and feature_up is None:
                x = cost
            elif feature_0 is not None and flow_up is not None and feature_up is not None:
                x = tf.concat([cost, feature_0, flow_up, feature_up], axis = 3)
            else:
                raise ValueError('Invalid value in feature_0, flow_up, or feature_up')

            # DenseNet
            conv = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            conv = tf.nn.leaky_relu(conv, 0.1)
            x = tf.concat([conv, x], axis = 3)
            conv = tf.layers.Conv2D(128, (3, 3), (1, 1), 'same')(x)
            conv = tf.nn.leaky_relu(conv, 0.1)
            x = tf.concat([conv, x], axis = 3)
            conv = tf.layers.Conv2D(96, (3, 3), (1, 1), 'same')(x)
            conv = tf.nn.leaky_relu(conv, 0.1)
            x = tf.concat([conv, x], axis = 3)
            conv = tf.layers.Conv2D(64, (3, 3), (1, 1), 'same')(x)
            conv = tf.nn.leaky_relu(conv, 0.1)
            x = tf.concat([conv, x], axis = 3)
            conv = tf.layers.Conv2D(32, (3, 3), (1, 1), 'same')(x)
            conv = tf.nn.leaky_relu(conv, 0.1)
            x = tf.concat([conv, x], axis = 3)
            flow = tf.layers.Conv2D(2, (3, 3), (1, 1), 'same')(x)

            if is_output:
                return x, flow
            else:
                flow_up = tf.layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same')(x)
                feature_up = tf.layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same')(x)
                return flow, flow_up, feature_up


# Context module -----------------------------------------------
class ContextNetwork(object):
    def __init__(self, name = 'context'):
        self.name = name

    def __call__(self, feature, flow):
        with tf.variable_scope(self.name) as vs:
            x = tf.concat([feature, flow], axis = 3)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (2, 2))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (4, 4))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(96, (3, 3), (1, 1),'same',
                                 dilation_rate = (8, 8))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1),'same',
                                 dilation_rate = (16, 16))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.1)
            x = tf.layers.Conv2D(2, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            return x+flow

        
# Utility function for guided filter
# really thanks for the original DeepGuidedFilter implementation (https://github.com/wuhuikai/DeepGuidedFilter)
def _diff_x(image, r):
    assert image.shape.ndims == 4
    left = image[:, r:2*r+1]
    middle = image[:, 2*r+1:] - image[:, :-2*r-1]
    right = image[:, -1:] - image[:, -2*r-1:-r-1]
    return tf.concat([left, middle, right], axis = 1)

def _diff_y(image, r):
    assert image.shape.ndims == 4
    left = image[:, :, r:2*r+1]
    middle = image[:, :, 2*r+1:] - image[:, :, :-2*r-1]
    right = image[:, :, -1:] - image[:, :, -2*r-1:-r-1]
    return tf.concat([left, middle, right], axis = 2)
        
def _box_filter(x, r):
    assert x.shape.ndims == 4
    return _diff_y(tf.cumsum(_diff_x(tf.cumsum(x, axis = 1), r), axis = 2), r)

# I try to implement fast guided filter (https://arxiv.org/abs/1505.00996)
class FastGuidedFilter(object):
    def __init__(self, r, channel_p, downscale = 2,
                 eps = 1e-8, name = 'guide'):
        self.r = r # box range
        self.channel_p = channel_p
        self.ds = downscale # downscale ratio for fast coeffcients calculation
        self.eps = eps # small constant
        self.name = name

    def __call__(self, p, I): # p:filtering input, I:guidance image
        with tf.name_scope(self.name) as ns:
            guider = GFCore(I, self.r, self.ds, self.eps)
            return guider.guide(p, self.channel_p)


class GFCore(object):
    def __init__(self, I, r, downscale, eps):
        self.I = I
        self.r = int(r/downscale)
        self.ds = downscale # downscale = 1: normal guided filter
        self.eps = eps
        
        self._init_guide()
        
    def _init_guide(self):
        _, h_i, w_i, c_i = tf.unstack(tf.shape(self.I))
        self.h_down = tf.cast(h_i/self.ds, tf.int32)
        self.w_down = tf.cast(w_i/self.ds, tf.int32)
        self.I_down = tf.image.resize_images(self.I, (self.h_down, self.w_down))
        
        self.N = _box_filter(tf.ones((1, self.h_down, self.w_down, 1),
                                     dtype = self.I.dtype), r = self.r)
        self.mean_I = _box_filter(self.I_down, self.r) / self.N
        self.var_I = _box_filter(self.I_down*self.I_down, self.r) / self.N

    def guide(self, p, channel_p):
        _, h_p, w_p, c_p = tf.unstack(tf.shape(p))
        tf.assert_equal(c_p, channel_p)
        tf.assert_equal(tf.cast([h_p/self.ds, w_p/self.ds], tf.int32),
                        [self.h_down, self.w_down])
        p_down = tf.image.resize_images(p, (self.h_down, self.w_down))
        
        q = [0]*channel_p
        for c in range(channel_p):
            p_c = tf.expand_dims(p_down[:,:,:,c], axis = 3)
            mean_p = _box_filter(p_c, self.r) / self.N
            cov_Ip = _box_filter(self.I_down*p_c, self.r) / self.N - self.mean_I*mean_p
            
            A_c = cov_Ip / (self.var_I+self.eps)
            b_c = mean_p - tf.expand_dims(tf.reduce_sum(A_c*self.mean_I, axis = 3),
                                          axis = 3)

            mean_A_c = _box_filter(A_c, self.r) / self.N
            mean_A_c = tf.image.resize_images(mean_A_c, (h_p, w_p))
            mean_b_c = _box_filter(b_c, self.r) / self.N
            mean_b_c = tf.image.resize_images(mean_b_c, (h_p, w_p))

            q_c = tf.expand_dims(tf.reduce_sum(mean_A_c*self.I, axis = 3), axis = 3)\
                  + mean_b_c
            q[c] = q_c

        return tf.concat(q, axis = 3)
