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


class ConvBlock(layers.Layer):
    def __init__(self, filters, rate=0.1, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.filters = filters
        self.rate = rate

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(self.filters, 3, 2, 'same')
        self.conv_2 = layers.Conv2D(self.filters, 3, 1, 'same')

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = tf.nn.leaky_relu(x, self.rate)
        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, self.rate)
        return x


class FlowBlock(layers.Layer):
    def __init__(self, rate=0.1, **kwargs):
        super(FlowBlock, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        self.conv_1 = layers.Conv2D(128, 3, 1, 'same')
        self.conv_2 = layers.Conv2D(128, 3, 1, 'same')
        self.conv_3 = layers.Conv2D(96, 3, 1, 'same')
        self.conv_4 = layers.Conv2D(64, 3, 1, 'same')
        self.conv_5 = layers.Conv2D(32, 3, 1, 'same')
        self.conv_f = layers.Conv2D(2, 3, 1, 'same')
        self.deconv = layers.Conv2DTranspose(2, 4, 2, 'same')
        self.upfeat = layers.Conv2DTranspose(2, 4, 2, 'same')

    def call(self, inputs):
        x = tf.concat(inputs, axis=-1)
        x = self.conv_1(x)
        x = tf.nn.leaky_relu(x, self.rate)
        x = self.conv_2(x)
        x = tf.nn.leaky_relu(x, self.rate)
        x = self.conv_3(x)
        x = tf.nn.leaky_relu(x, self.rate)
        x = self.conv_4(x)
        x = tf.nn.leaky_relu(x, self.rate)
        x = self.conv_5(x)
        x = tf.nn.leaky_relu(x, self.rate)
        flow = self.conv_f(x)
        upflow = self.deconv(flow)
        upfeat = self.upfeat(x)
        return [flow, upflow, upfeat]
