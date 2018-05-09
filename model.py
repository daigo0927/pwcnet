import tensorflow as tf

from modules import *


class PWCNet(object):

    def __init__(self, num_levels = 6, search_range = 4,
                 output_level = 4, batch_norm = False,
                 name = 'pwcnet'):
        self.num_levels = num_levels
        self.s_range = search_range
        self.output_level = output_level
        self.batch_norm = batch_norm
        self.name = name

        self.fp_extractor = FeaturePyramidExtractor(self.num_levels, batch_norm)
        self.warp_layer = WarpingLayer()
        self.cv_layer = CostVolumeLayer(search_range)
        self.of_estimators = [OpticalFlowEstimator(self.batch_norm,
                                                   name = f'of_level{l}')\
                              for l in range(self.num_levels)]
        self.context = ContextNetwork()

    def __call__(self, images_0, images_1):
        with tf.variable_scope(self.name) as vs:

            pyramid_0 = self.fp_extractor(images_0, reuse = False) + [images_0]
            pyramid_1 = self.fp_extractor(images_1) + [images_1]

            flows = []
            summaries = {'feature_1_warps':[]}

            # coarse to fine processing
            for l, (feature_0, feature_1) in enumerate(zip(pyramid_0, pyramid_1)):
                print(f'Level {l}')
                b, h, w, _ = tf.unstack(tf.shape(feature_0))
                
                if l == 0:
                    flow = tf.zeros((b, h, w, 2), dtype = tf.int32)
                    reuse = False
                else:
                    flow = tf.image.resize_bilinear(flow, (h, w))*2
                    reuse = True

                # warping -> costvolume -> optical flow estimation
                feature_1_warped = self.warp_layer(feature_1, flow)
                cost = self.cv_layer(feature_0, feature_1_warped, reuse = reuse)
                feature, flow = self.of_estimators[l](feature_0, cost, flow)

                # context considering process
                flow = self.context(feature, flow, reuse = reuse)

                # stop processing at the defined level
                if l == self.output_level:
                    upscale = 2**(self.num_levels - self.output_level)
                    flow = tf.image.resize_bilinear(flow, (h*upscale, w*upscale))*upscale
                    flows.append(flow)
                    summaries['feature_1_warps'].append(feature_1_warped)
                    break
                else:
                    flows.append(flow)
                    summaries['feature_1_warps'].append(feature_1_warped)

            return flows, summaries

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name in var.name]
