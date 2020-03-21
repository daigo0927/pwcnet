"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Daigo Hirooka
"""

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from .modules import dense_warp, ConvBlock, FlowBlock


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

        self.cost_volume = tfa.layers.CorrelationCost(
            kernel_size=1,
            max_displacement=max_displacement,
            stride_1=1,
            stride_2=1,
            pad=max_displacement,
            data_format='channels_last')

        self.fblock_6 = FlowBlock(leak_rate)
        self.fblock_5 = FlowBlock(leak_rate)
        self.fblock_4 = FlowBlock(leak_rate)
        self.fblock_3 = FlowBlock(leak_rate)
        self.fblock_2 = FlowBlock(leak_rate)

        self.context = tf.keras.Sequential([
            layers.Conv2D(128, (3, 3), (1, 1), 'same', dilation_rate=(1, 1)),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(128, (3, 3), (1, 1), 'same', dilation_rate=(2, 2)),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(128, (3, 3), (1, 1), 'same', dilation_rate=(4, 4)),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(96, (3, 3), (1, 1), 'same', dilation_rate=(8, 8)),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(64, (3, 3), (1, 1), 'same', dilation_rate=(16, 16)),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(32, (3, 3), (1, 1), 'same', dilation_rate=(1, 1)),
            layers.LeakyReLU(leak_rate),
            layers.Conv2D(2, (3, 3), (1, 1), 'same')
        ])
        self.upsample = layers.UpSampling2D(size=(4, 4),
                                            interpolation='bilinear')

    def call(self, inputs):
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

        cost_6 = self.cost_volume([c_1_6, c_2_6])
        cost_6 = tf.nn.leaky_relu(cost_6, self.leak_rate)
        flow_6, upflow_6, upfeat_6 = self.fblock_6([cost_6])

        warp_5 = dense_warp(c_2_5, upflow_6 * 0.625)
        cost_5 = self.cost_volume([c_1_5, warp_5])
        cost_5 = tf.nn.leaky_relu(cost_5, self.leak_rate)
        flow_5, upflow_5, upfeat_5 = self.fblock_5(
            [cost_5, c_1_5, upflow_6, upfeat_6])

        warp_4 = dense_warp(c_2_4, upflow_5 * 1.25)
        cost_4 = self.cost_volume([c_1_4, warp_4])
        cost_4 = tf.nn.leaky_relu(cost_4, self.leak_rate)
        flow_4, upflow_4, upfeat_4 = self.fblock_4(
            [cost_4, c_1_4, upflow_5, upfeat_5])

        warp_3 = dense_warp(c_2_3, upflow_4 * 2.5)
        cost_3 = self.cost_volume([c_1_3, warp_3])
        cost_3 = tf.nn.leaky_relu(cost_3, self.leak_rate)
        flow_3, upflow_3, upfeat_3 = self.fblock_3(
            [cost_3, c_1_3, upflow_4, upfeat_4])

        warp_2 = dense_warp(c_2_2, upflow_3 * 5.0)
        cost_2 = self.cost_volume([c_1_2, warp_2])
        cost_2 = tf.nn.leaky_relu(cost_2, self.leak_rate)
        flow_2, _, _ = self.fblock_2([cost_2, c_1_2, upflow_3, upfeat_3])

        flow_2 = flow_2 + self.context(flow_2)
        flow = self.upsample(flow_2) * 20

        return [flow_2, flow_3, flow_4, flow_5, flow_6], flow
