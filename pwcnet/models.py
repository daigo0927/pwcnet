"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Daigo Hirooka
"""

import tensorflow as tf
from tensorflow.keras import layers

from modules import dense_warp, cost_volume
from modules import ConvBlock, DeepConvBlock, FlowBlock, DenseFlowBlock, ContextBlock


class PWCNet(tf.keras.Model):
    def __init__(self,
                 filters_list=[16, 32, 64, 96, 128, 192],
                 leak_rate=0.1,
                 max_displacement=4,
                 **kwargs):
        super(PWCNet, self).__init__(**kwargs)
        self.filters_list = filters_list
        self.leak_rate = leak_rate
        self.max_displacement = max_displacement

        self.cblock_1 = ConvBlock(filters_list[0], leak_rate)
        self.cblock_2 = ConvBlock(filters_list[1], leak_rate)
        self.cblock_3 = ConvBlock(filters_list[2], leak_rate)
        self.cblock_4 = ConvBlock(filters_list[3], leak_rate)
        self.cblock_5 = ConvBlock(filters_list[4], leak_rate)
        self.cblock_6 = ConvBlock(filters_list[5], leak_rate)

        self.fblock_6 = FlowBlock(leak_rate)
        self.fblock_5 = FlowBlock(leak_rate)
        self.fblock_4 = FlowBlock(leak_rate)
        self.fblock_3 = FlowBlock(leak_rate)
        self.fblock_2 = FlowBlock(leak_rate, is_output=True)
        self.context = ContextBlock(leak_rate)

        self.upsample = layers.UpSampling2D(size=(4, 4),
                                            interpolation='bilinear')

    def call(self, inputs, training=None):
        image_1 = inputs[0]
        c_1_1 = self.cblock_1(image_1)
        c_1_2 = self.cblock_2(c_1_1)
        c_1_3 = self.cblock_3(c_1_2)
        c_1_4 = self.cblock_4(c_1_3)
        c_1_5 = self.cblock_5(c_1_4)
        c_1_6 = self.cblock_6(c_1_5)

        image_2 = inputs[1]
        c_2_1 = self.cblock_1(image_2)
        c_2_2 = self.cblock_2(c_2_1)
        c_2_3 = self.cblock_3(c_2_2)
        c_2_4 = self.cblock_4(c_2_3)
        c_2_5 = self.cblock_5(c_2_4)
        c_2_6 = self.cblock_6(c_2_5)

        corr_6 = cost_volume(c_1_6, c_2_6, self.max_displacement)
        corr_6 = tf.nn.leaky_relu(corr_6, self.leak_rate)
        flow_6, upflow_6, upfeat_6 = self.fblock_6([corr_6])

        warp_5 = dense_warp(c_2_5, upflow_6 * 0.625)
        corr_5 = cost_volume(c_1_5, warp_5, self.max_displacement)
        corr_5 = tf.nn.leaky_relu(corr_5, self.leak_rate)
        flow_5, upflow_5, upfeat_5 = self.fblock_5(
            [corr_5, c_1_5, upflow_6, upfeat_6])

        warp_4 = dense_warp(c_2_4, upflow_5 * 1.25)
        corr_4 = cost_volume(c_1_4, warp_4, self.max_displacement)
        corr_4 = tf.nn.leaky_relu(corr_4, self.leak_rate)
        flow_4, upflow_4, upfeat_4 = self.fblock_4(
            [corr_4, c_1_4, upflow_5, upfeat_5])

        warp_3 = dense_warp(c_2_3, upflow_4 * 2.5)
        corr_3 = cost_volume(c_1_3, warp_3, self.max_displacement)
        corr_3 = tf.nn.leaky_relu(corr_3, self.leak_rate)
        flow_3, upflow_3, upfeat_3 = self.fblock_3(
            [corr_3, c_1_3, upflow_4, upfeat_4])

        warp_2 = dense_warp(c_2_2, upflow_3 * 5.0)
        corr_2 = cost_volume(c_1_2, warp_2, self.max_displacement)
        corr_2 = tf.nn.leaky_relu(corr_2, self.leak_rate)
        flow_2 = self.fblock_2([corr_2, c_1_2, upflow_3, upfeat_3])

        flow_2 = flow_2 + self.context(flow_2)
        flow = self.upsample(flow_2) * 20

        if training:
            return [flow_2, flow_3, flow_4, flow_5, flow_6]
        else:
            return flow


class PWCDCNet(tf.keras.Model):
    def __init__(self,
                 filters_list=[16, 32, 64, 96, 128, 192],
                 leak_rate=0.1,
                 max_displacement=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters_list = filters_list
        self.leak_rate = leak_rate
        self.max_displacement = max_displacement

        self.cblock_1 = DeepConvBlock(filters_list[0], leak_rate)
        self.cblock_2 = DeepConvBlock(filters_list[1], leak_rate)
        self.cblock_3 = DeepConvBlock(filters_list[2], leak_rate)
        self.cblock_4 = DeepConvBlock(filters_list[3], leak_rate)
        self.cblock_5 = DeepConvBlock(filters_list[4], leak_rate)
        self.cblock_6 = DeepConvBlock(filters_list[5], leak_rate)

        self.fblock_6 = DenseFlowBlock(leak_rate)
        self.fblock_5 = DenseFlowBlock(leak_rate)
        self.fblock_4 = DenseFlowBlock(leak_rate)
        self.fblock_3 = DenseFlowBlock(leak_rate)
        self.fblock_2 = DenseFlowBlock(leak_rate, is_output=True)
        self.context = ContextBlock(leak_rate)

        self.upsample = layers.UpSampling2D(size=(4, 4),
                                            interpolation='bilinear')

    def call(self, inputs, training=None):
        image_1, image_2 = inputs
        c_1_1 = self.cblock_1(image_1)
        c_1_2 = self.cblock_2(c_1_1)
        c_1_3 = self.cblock_3(c_1_2)
        c_1_4 = self.cblock_4(c_1_3)
        c_1_5 = self.cblock_5(c_1_4)
        c_1_6 = self.cblock_6(c_1_5)

        c_2_1 = self.cblock_1(image_2)
        c_2_2 = self.cblock_2(c_2_1)
        c_2_3 = self.cblock_3(c_2_2)
        c_2_4 = self.cblock_4(c_2_3)
        c_2_5 = self.cblock_5(c_2_4)
        c_2_6 = self.cblock_6(c_2_5)

        corr_6 = cost_volume(c_1_6, c_2_6, self.max_displacement)
        corr_6 = tf.nn.leaky_relu(corr_6, self.leak_rate)
        flow_6, upflow_6, upfeat_6 = self.fblock_6([corr_6])

        warp_5 = dense_warp(c_2_5, upflow_6 * 0.625)
        corr_5 = cost_volume(c_1_5, warp_5, self.max_displacement)
        corr_5 = tf.nn.leaky_relu(corr_5, self.leak_rate)
        flow_5, upflow_5, upfeat_5 = self.fblock_5(
            [corr_5, c_1_5, upflow_6, upfeat_6])

        warp_4 = dense_warp(c_2_4, upflow_5 * 1.25)
        corr_4 = cost_volume(c_1_4, warp_4, self.max_displacement)
        corr_4 = tf.nn.leaky_relu(corr_4, self.leak_rate)
        flow_4, upflow_4, upfeat_4 = self.fblock_4(
            [corr_4, c_1_4, upflow_5, upfeat_5])

        warp_3 = dense_warp(c_2_3, upflow_4 * 2.5)
        corr_3 = cost_volume(c_1_3, warp_3, self.max_displacement)
        corr_3 = tf.nn.leaky_relu(corr_3, self.leak_rate)
        flow_3, upflow_3, upfeat_3 = self.fblock_3(
            [corr_3, c_1_3, upflow_4, upfeat_4])

        warp_2 = dense_warp(c_2_2, upflow_3 * 5.0)
        corr_2 = cost_volume(c_1_2, warp_2, self.max_displacement)
        corr_2 = tf.nn.leaky_relu(corr_2, self.leak_rate)
        flow_2 = self.fblock_2([corr_2, c_1_2, upflow_3, upfeat_3])

        flow_2 = flow_2 + self.context(flow_2)
        flow = self.upsample(flow_2) * 20

        if training:
            return [flow_2, flow_3, flow_4, flow_5, flow_6]
        else:
            return flow


if __name__ == '__main__':
    model = PWCDCNet(name='pwcdcnet')

    @tf.function
    def forward(x_1, x_2):
        return model([x_1, x_2])

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs_test/func/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

    x_1 = tf.random.uniform((1, 384, 448, 3))
    x_2 = tf.random.uniform((1, 384, 448, 3))

    tf.summary.trace_on(graph=True, profiler=True)
    y = forward(x_1, x_2)
    with writer.as_default():
        tf.summary.trace_export(name='pwcdcnet_trace',
                                step=0,
                                profiler_outdir=logdir)
