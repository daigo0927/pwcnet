import tensorflow as tf
import ops
from tensorflow.keras import layers


class PWCNet(tf.keras.Model):
    def __init__(self,
                 max_displacement=4,
                 name='pwcnet'):
        super().__init__(name=name)
        self.max_displacement = max_displacement

        self.extractor = ops.FeaturePyramidExtractor(name='extractor')
        self.warp = ops.warp
        self.cost_volume = ops.CostVolume(max_displacement, name='cost_volume')

        self.estimator2 = ops.OpticalFlowEstimator(upsample=False,
                                                   name='estimator2')
        self.estimator3 = ops.OpticalFlowEstimator(name='estimator3')
        self.estimator4 = ops.OpticalFlowEstimator(name='estimator4')
        self.estimator5 = ops.OpticalFlowEstimator(name='estimator5')
        self.estimator6 = ops.OpticalFlowEstimator(name='estimator6')

        self.context = ops.ContextNetwork(name='context')
        self.resize = layers.UpSampling2D((4, 4), interpolation='bilinear',
                                          name='resize')

    def call(self, x1, x2):
        c16, c15, c14, c13, c12, c11 = self.extractor(x1)
        c26, c25, c24, c23, c22, c21 = self.extractor(x2)

        cv = self.cost_volume(c16, c26)
        flow6, upflow, upfeat = self.estimator6(cv, c16)

        warp5 = self.warp(c25, upflow*0.625)
        cv = self.cost_volume(c15, warp5)
        flow5, upflow, upfeat = self.estimator5(cv, c15, upflow, upfeat)

        warp4 = self.warp(c24, upflow*1.25)
        cv = self.cost_volume(c14, warp4)
        flow4, upflow, upfeat = self.estimator4(cv, c14, upflow, upfeat)

        warp3 = self.warp(c23, upflow*2.5)
        cv = self.cost_volume(c13, warp3)
        flow3, upflow, upfeat = self.estimator3(cv, c13, upflow, upfeat)

        warp2 = self.warp(c22, upflow*5.0)
        cv = self.cost_volume(c12, warp2)
        flow2, x = self.estimator2(cv, c12, upflow, upfeat)

        flow2 = self.context(flow2, x)
        flow = self.resize(flow2)*20.0
        return flow, [flow2, flow3, flow4, flow5, flow6]
