import tensorflow as tf
from tensorflow.keras import layers


def meshgrid(inputs):
    bs, h, w, _ = tf.unstack(tf.shape(inputs))
    gb, gy, gx = tf.meshgrid(tf.range(bs),
                             tf.range(h),
                             tf.range(w),
                             indexing='ij')
    return gb, gy, gx


def nearest_warp(image, flow):
    _, h, w, _ = tf.unstack(tf.shape(image))
    gb, gy, gx = meshgrid(image)

    flow = tf.cast(flow, tf.int32)
    fx, fy = tf.unstack(flow, 2, axis=-1)

    gy_warped = gy + fy
    gx_warped = gx + fx
    gy_warped = tf.clip_by_value(gy_warped, 0, h - 1)
    gx_warped = tf.clip_by_value(gx_warped, 0, w - 1)
    i_warped = tf.stack([gb, gy_warped, gx_warped], axis=-1)

    pullback = tf.gather_nd(image, i_warped)
    return pullback


def bilinear_warp(image, flow):
    _, h, w, _ = tf.unstack(tf.shape(image))
    gb, gy, gx = meshgrid(image)

    fx, fy = tf.unstack(flow, 2, axis=-1)
    fx_floor = tf.floor(fx)
    fx_ceil = fx_floor + 1.0
    fy_floor = tf.floor(fy)
    fy_ceil = fy_floor + 1.0

    # Interporation weights
    w_ff = tf.expand_dims((fy_ceil - fy) * (fx_ceil - fx), axis=-1)
    w_fc = tf.expand_dims((fy_ceil - fy) * (fx - fx_floor), axis=-1)
    w_cf = tf.expand_dims((fy - fy_floor) * (fx_ceil - fx), axis=-1)
    w_cc = tf.expand_dims((fy - fy_floor) * (fx - fx_floor), axis=-1)

    # Cast to int32
    fx_floor = tf.cast(fx_floor, tf.int32)
    fx_ceil = tf.cast(fx_ceil, tf.int32)
    fy_floor = tf.cast(fy_floor, tf.int32)
    fy_ceil = tf.cast(fy_ceil, tf.int32)

    # Indices
    gy_warped_floor = tf.clip_by_value(gy + fy_floor, 0, h - 1)
    gy_warped_ceil = tf.clip_by_value(gy + fy_ceil, 0, h - 1)
    gx_warped_floor = tf.clip_by_value(gx + fx_floor, 0, w - 1)
    gx_warped_ceil = tf.clip_by_value(gx + fx_ceil, 0, w - 1)

    i_warped_ff = tf.stack([gb, gy_warped_floor, gx_warped_floor], axis=-1)
    i_warped_fc = tf.stack([gb, gy_warped_floor, gx_warped_ceil], axis=-1)
    i_warped_cf = tf.stack([gb, gy_warped_ceil, gx_warped_floor], axis=-1)
    i_warped_cc = tf.stack([gb, gy_warped_ceil, gx_warped_ceil], axis=-1)

    # Gather pixels
    pb_ff = tf.gather_nd(image, i_warped_ff)
    pb_fc = tf.gather_nd(image, i_warped_fc)
    pb_cf = tf.gather_nd(image, i_warped_cf)
    pb_cc = tf.gather_nd(image, i_warped_cc)

    # Interporation
    pullback = w_ff * pb_ff + w_fc * pb_fc + w_cf * pb_cf + w_cc * pb_cc
    return pullback


def dense_warp(image, flow, interpolation='bilinear'):
    """ Pull back post-warp image with dense flow vectors.

    Args:
      image: A 4-D float Tensor with shape [batch, height, width, channels].
      flow: A 4-D float Tensor with shape [batch, height, width, 2].
      interpolation: A string either 'nearest' or 'bilinear' specifying
        interpolation method.
      
    Returns:
      A 4-D float Tensor with shape [batch, height, width, channels] and
        same type as the input image.
    """
    if interpolation == 'nearest':
        pullback = nearest_warp(image, flow)
    elif interpolation == 'bilinear':
        pullback = bilinear_warp(image, flow)
    else:
        raise KeyError('invalid interpolation: %s' % interpolation)
    return pullback


"""
core_costvol.py
Computes cross correlation between two feature maps.
Written by Phil Ferriere
Licensed under the MIT License (see LICENSE for details)
Based on:
    - https://github.com/tensorpack/tensorpack/blob/master/examples/OpticalFlow/flownet_models.py
        Written by Patrick Wieschollek, Copyright Yuxin Wu
        Apache License 2.0
"""


def cost_volume(c1, warp, search_range):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        c1: Level of the feature pyramid of Image1
        warp: Warped level of the feature pyramid of image22
        search_range: Search range (maximum displacement)
    """
    padded_lvl = tf.pad(warp, [[0, 0], [search_range, search_range],
                               [search_range, search_range], [0, 0]])
    _, h, w, _ = tf.unstack(tf.shape(c1))
    max_offset = search_range * 2 + 1

    cv = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            slice = tf.slice(padded_lvl, [0, y, x, 0], [-1, h, w, -1])
            cost = tf.reduce_mean(c1 * slice, axis=3, keepdims=True)
            cv.append(cost)
    cv = tf.concat(cv, axis=3)
    return cv


def ConvBlock(filters, leak_rate=0.1, **kwargs):
    seq = tf.keras.Sequential([
        layers.Conv2D(filters, 3, 2, 'same'),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(filters, 3, 1, 'same'),
        layers.LeakyReLU(leak_rate)
    ], **kwargs)
    return seq


def DeepConvBlock(filters, leak_rate=0.1, **kwargs):
    seq = tf.keras.Sequential([
        layers.Conv2D(filters, 3, 2, 'same'),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(filters, 3, 1, 'same'),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(filters, 3, 1, 'same'),
        layers.LeakyReLU(leak_rate)
    ], **kwargs)
    return seq


class FlowBlock(layers.Layer):
    def __init__(self, leak_rate=0.1, is_output=False, **kwargs):
        super().__init__(**kwargs)
        self.leak_rate = leak_rate
        self.is_output = is_output

        self.seq = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), (1, 1), 'same'),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(128, (3, 3), (1, 1), 'same'),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(96, (3, 3), (1, 1), 'same'),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(64, (3, 3), (1, 1), 'same'),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(32, (3, 3), (1, 1), 'same'),
            layers.LeakyReLU(leak_rate),
        ], **kwargs)
        self.conv_f = layers.Conv2D(2, 3, 1, 'same')

        if not is_output:
            self.deconv = layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same')
            self.upfeat = layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same')

    def call(self, inputs):
        x = tf.concat(inputs, axis=-1)
        x = self.seq(x)
        flow = self.conv_f(x)
        upflow = self.deconv(flow)
        upfeat = self.upfeat(x)

        if not self.is_output:
            upflow = self.deconv(flow)
            upfeat = self.upfeat(x)
            return [flow, upflow, upfeat]
        else:
            return flow


class DenseFlowBlock(layers.Layer):
    def __init__(self, leak_rate=0.1, is_output=False, **kwargs):
        super().__init__(**kwargs)
        self.leak_rate = leak_rate
        self.is_output = is_output

        self.conv_1 = layers.Conv2D(128, 3, 1, 'same')
        self.conv_2 = layers.Conv2D(128, 3, 1, 'same')
        self.conv_3 = layers.Conv2D(96, 3, 1, 'same')
        self.conv_4 = layers.Conv2D(64, 3, 1, 'same')
        self.conv_5 = layers.Conv2D(32, 3, 1, 'same')
        self.conv_f = layers.Conv2D(2, 3, 1, 'same')
        self.act = layers.LeakyReLU(leak_rate)

        if not is_output:
            self.deconv = layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same')
            self.upfeat = layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same')

    def call(self, inputs):
        x = tf.concat(inputs, axis=-1)
        x = tf.concat([self.act(self.conv_1(x)), x], axis=-1)
        x = tf.concat([self.act(self.conv_2(x)), x], axis=-1)
        x = tf.concat([self.act(self.conv_3(x)), x], axis=-1)
        x = tf.concat([self.act(self.conv_4(x)), x], axis=-1)
        x = tf.concat([self.act(self.conv_5(x)), x], axis=-1)
        flow = self.conv_f(x)

        if not self.is_output:
            upflow = self.deconv(flow)
            upfeat = self.upfeat(x)
            return [flow, upflow, upfeat]
        else:
            return flow


def ContextBlock(leak_rate=0.1, **kwargs):
    seq = tf.keras.Sequential([
        layers.Conv2D(128, 3, 1, 'same', dilation_rate=1),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(128, 3, 1, 'same', dilation_rate=2),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(128, 3, 1, 'same', dilation_rate=4),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(96, 3, 1, 'same', dilation_rate=8),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(64, 3, 1, 'same', dilation_rate=16),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(32, 3, 1, 'same', dilation_rate=1),
        layers.LeakyReLU(leak_rate),
        layers.Conv2D(2, 3, 1, 'same')
    ], **kwargs)
    return seq
