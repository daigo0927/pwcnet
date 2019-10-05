import tensorflow as tf
from tensorflow.keras import layers


class ConvBlock(layers.Layer):
    def __init__(self, filters, name='block'):
        super().__init__(name=name)
        self.filters = filters
        
        self.conv1 = layers.Conv2D(filters, (3, 3), (2, 2), 'same', name='conv1')
        self.conv2 = layers.Conv2D(filters, (3, 3), (1, 1), 'same', name='conv2')
        self.conv3 = layers.Conv2D(filters, (3, 3), (1, 1), 'same', name='conv3')
        self.act = layers.LeakyReLU(0.1)

    def call(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return x
        

class FeaturePyramidExtractor(layers.Layer):
    def __init__(self, name='extractor'):
        super().__init__(name=name)

        self.block1 = ConvBlock( 16, name='block1')
        self.block2 = ConvBlock( 32, name='block2')
        self.block3 = ConvBlock( 64, name='block3')
        self.block4 = ConvBlock( 96, name='block4')
        self.block5 = ConvBlock(128, name='block5')
        self.block6 = ConvBlock(196, name='block6')

    def call(self, x):
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)
        c6 = self.block6(c5)

        return [c6, c5, c4, c3, c2, c1]


def warp(x, flow):
    bs, h, w, nch = tf.unstack(tf.shape(x))
    gb, gy, gx = tf.meshgrid(tf.range(bs), tf.range(h), tf.range(w), indexing='ij')
    gb = tf.cast(gb, tf.float32)
    gy = tf.cast(gy, tf.float32)
    gx = tf.cast(gx, tf.float32)

    fx, fy = tf.unstack(flow, axis=-1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0 + 1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0 + 1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(gy + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(gy + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(gx + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(gx + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([gb, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([gb, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([gb, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([gb, gy_1, gx_1], axis = 3), tf.int32)

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
    

def pad2d(x, hpad, wpad):
    return tf.pad(x, [[0, 0], hpad, wpad, [0, 0]])

def crop2d(x, hcrop, wcrop):
    _, h, w, _ = tf.unstack(tf.shape(x))
    return x[:, hcrop[0]:h-hcrop[1], wcrop[0]:w-wcrop[1], :]

def cost(x1, warped, shift):
    sh, sw = shift
    hpad = [max(sh, 0), abs(min(sh, 0))]
    wpad = [max(sw, 0), abs(min(sw, 0))]
    hpad_r = hpad[::-1]
    wpad_r = wpad[::-1]
    
    x1 = pad2d(x1, hpad, wpad)
    warped = pad2d(warped, hpad_r, wpad_r)
    cost_ = x1*warped
    cost_ = crop2d(cost_, hpad, wpad)
    return tf.reduce_mean(cost_, axis=3)


class CostVolume(layers.Layer):
    def __init__(self,
                 max_displacement=4,
                 name='cost_volume'):
        super().__init__(name=name)
        self.max_displacement = max_displacement
        self.act = layers.LeakyReLU(0.1)

    def call(self, x1, warped):
        md = self.max_displacement
        
        cv = []
        for sh in range(-md, md+1):
            for sw in range(-md, md+1):
                c = cost(x1, warped, [sh, sw])
                cv.append(c)

        cv = tf.stack(cv, axis=3)
        return self.act(cv)


class OpticalFlowEstimator(layers.Layer):
    def __init__(self,
                 upsample=True,
                 name='flow_estimator'):
        super().__init__(name=name)
        self.upsample = upsample

        self.conv1 = layers.Conv2D(128, (3, 3), (1, 1), 'same', name='conv1')
        self.conv2 = layers.Conv2D(128, (3, 3), (1, 1), 'same', name='conv2')
        self.conv3 = layers.Conv2D( 96, (3, 3), (1, 1), 'same', name='conv3')
        self.conv4 = layers.Conv2D( 64, (3, 3), (1, 1), 'same', name='conv4')
        self.conv5 = layers.Conv2D( 32, (3, 3), (1, 1), 'same', name='conv5')
        self.act = layers.LeakyReLU(0.1)

        self.toflow = layers.Conv2D(2, (3, 3), (1, 1), 'same', name='toflow')

        if upsample:
            self.deconv = layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same',
                                                 name='deconv')
            self.upfeat = layers.Conv2DTranspose(2, (4, 4), (2, 2), 'same',
                                                 name='upfeat')

    def call(self, cv, x1=None, upflow=None, upfeat=None):
        x = [f for f in [cv, x1, upflow, upfeat] if f is not None]
        x = tf.concat(x, axis=3)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        flow = self.toflow(x)
        
        if upflow is not None:
            flow += upflow

        if self.upsample:
            upflow = self.deconv(flow)
            upfeat = self.upfeat(x)
            return flow, upflow, upfeat
        else:
            return flow, x


class ContextNetwork(layers.Layer):
    def __init__(self, name='context'):
        super().__init__(name=name)

        self.conv1 = layers.Conv2D(128, (3, 3), (1, 1), 'same',
                                   dilation_rate=(1, 1), name='conv1')
        self.conv2 = layers.Conv2D(128, (3, 3), (1, 1), 'same',
                                   dilation_rate=(2, 2), name='conv2')
        self.conv3 = layers.Conv2D(128, (3, 3), (1, 1), 'same',
                                   dilation_rate=(4, 4), name='conv3')
        self.conv4 = layers.Conv2D( 96, (3, 3), (1, 1), 'same',
                                   dilation_rate=(8, 8), name='conv4')
        self.conv5 = layers.Conv2D( 64, (3, 3), (1, 1), 'same',
                                   dilation_rate=(16, 16), name='conv5')
        self.conv6 = layers.Conv2D( 32, (3, 3), (1, 1), 'same',
                                   dilation_rate=(1, 1), name='conv6')
        self.conv7 = layers.Conv2D(  2, (3, 3), (1, 1), 'same',
                                   dilation_rate=(1, 1), name='conv7')
        self.act = layers.LeakyReLU(0.1)

    def call(self, flow, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.act(self.conv7(x))
        return flow + x
