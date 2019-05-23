import numpy as np
import tensorflow as tf
from functools import partial



def _conv_block(filters, kernel_size = (3, 3), strides = (1, 1), batch_norm = False):
    def f(x):
        x = tf.layers.Conv2D(filters, kernel_size,
                             strides, 'same')(x)
        if batch_norm:
            x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x
    return f


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
    """ Feature pyramid extractor module"""
    def __init__(self, num_levels = 6, name = 'fp_extractor'):
        self.num_levels = num_levels
        self.filters = [16, 32, 64, 96, 128, 192]
        self.name = name

    def __call__(self, images, reuse = True):
        """
        Args:
        - images (batch, h, w, 3): input images

        Returns:
        - features_pyramid (batch, h_l, w_l, nch_l) for each scale levels:
          extracted feature pyramid (deep -> shallow order)
        """
        with tf.variable_scope(self.name, reuse = reuse) as vs:
            features_pyramid = []
            x = images
            for l in range(self.num_levels):
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (2, 2), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                x = tf.layers.Conv2D(self.filters[l], (3, 3), (1, 1), 'same')(x)
                x = tf.nn.leaky_relu(x, 0.1)
                features_pyramid.append(x)
                
            # return feature pyramid by ascent order
            return features_pyramid[::-1]
        

# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

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

def get_cost(features_0, features_0from1, shift):
    """
    Calculate cost volume for specific shift

    - inputs
    features_0 (batch, h, w, nch): feature maps at time slice 0
    features_0from1 (batch, h, w, nch): feature maps at time slice 0 warped from 1
    shift (2): spatial (vertical and horizontal) shift to be considered

    - output
    cost (batch, h, w): cost volume map for the given shift
    """
    v, h = shift # vertical/horizontal element
    vt, vb, hl, hr =  max(v,0), abs(min(v,0)), max(h,0), abs(min(h,0)) # top/bottom left/right
    f_0_pad = pad2d(features_0, [vt, vb], [hl, hr])
    f_0from1_pad = pad2d(features_0from1, [vb, vt], [hr, hl])
    cost_pad = f_0_pad*f_0from1_pad
    return tf.reduce_mean(crop2d(cost_pad, [vt, vb], [hl, hr]), axis = 3)
            
class CostVolumeLayer(object):
    """ Cost volume module """
    def __init__(self, search_range = 4, name = 'cost_volume'):
        self.s_range = search_range
        self.name = name

    def __call__(self, features_0, features_0from1):
        with tf.name_scope(self.name) as ns:
            b, h, w, f = tf.unstack(tf.shape(features_0))
            cost_length = (2*self.s_range+1)**2

            get_c = partial(get_cost, features_0, features_0from1)
            cv = [0]*cost_length
            depth = 0
            for v in range(-self.s_range, self.s_range+1):
                for h in range(-self.s_range, self.s_range+1):
                    cv[depth] = get_c(shift = [v, h])
                    depth += 1

            cv = tf.stack(cv, axis = 3)
            cv = tf.nn.leaky_relu(cv, 0.1)
            return cv
            

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
    """ Optical flow estimator module """
    def __init__(self, use_dc = False, name = 'of_estimator'):
        """
        Args:
        - use_dc (bool): optional bool to use dense-connection, False as default
        - name: module name
        """
        self.filters = [128, 128, 96, 64, 32]
        self.use_dc = use_dc
        self.name = name

    def __call__(self, cv, features_0 = None, flows_up_prev = None, features_up_prev = None,
                 is_output = False):
        """
        Args:
        - cv (batch, h, w, nch_cv): cost volume
        - features_0 (batch, h, w, nch_f0): feature map at time slice t
        - flows_up_prev (batch, h, w, 2): upscaled optical flow passed from previous OF-estimator
        - features_up_prev (batch, h, w, nch_fup): upscaled feature map passed from previous OF-estimator
        - is_output (bool): whether at output level or not

        Returns:
        - flows (batch, h, w, 2): convolved optical flow

        and 
        is_output: False
        - flows_up (batch, 2*h, 2*w, 2): upsampled optical flow
        - features_up (batch, 2*h, 2*w, nch_f): upsampled feature map

        is_output: True
        - features (batch, h, w, nch_f): convolved feature map
        """
        with tf.variable_scope(self.name) as vs:
            features = cv
            for f in [features_0, flows_up_prev, features_up_prev]:
                if f is not None:
                    features = tf.concat([features, f], axis = 3)

            for f in self.filters:
                conv = tf.layers.Conv2D(f, (3, 3), (1, 1), 'same')(features)
                conv = tf.nn.leaky_relu(conv, 0.1)
                if self.use_dc:
                    features = tf.concat([conv, features], axis = 3)
                else:
                    features = conv

            flows = tf.layers.Conv2D(2, (3, 3), (1, 1), 'same')(features)
            if flows_up_prev is not None:
                # Residual connection
                flows += flows_up_prev

            if is_output:
                return flows, features
            else:
                _, h, w, _ = tf.unstack(tf.shape(flows))
                flows_up = tf.image.resize_bilinear(flows, (2*h, 2*w))
                features_up = tf.image.resize_bilinear(features, (2*h, 2*w))
                return flows, flows_up, features_up

            

# Context module -----------------------------------------------
class ContextNetwork(object):
    """ Context module """
    def __init__(self, name = 'context'):
        self.name = name

    def __call__(self, flows, features):
        """
        Args:
        - flows (batch, h, w, 2): optical flow
        - features (batch, h, w, 2): feature map passed from previous OF-estimator

        Returns:
        - flows (batch, h, w, 2): convolved optical flow
        """
        with tf.variable_scope(self.name) as vs:
            x = tf.concat([flows, features], axis = 3)
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
            return flows + x
