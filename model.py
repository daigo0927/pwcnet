import tensorflow as tf

from modules import *


class PWCNet(object):
    def __init__(self, num_levels = 6, search_range = 4, warp_type = 'bilinear',
                 output_level = 4, name = 'pwcnet'):
        self.num_levels = num_levels
        self.s_range = search_range
        self.warp_type = warp_type
        assert output_level < num_levels, 'Should set output_level < num_levels'
        self.output_level = output_level
        self.name = name

        self.fp_extractor = FeaturePyramidExtractor(self.num_levels)
        self.warp_layer = WarpingLayer(self.warp_type)
        self.cv_layer = CostVolumeLayer(self.s_range)
        self.of_estimators = [OpticalFlowEstimator(self.batch_norm,
                                                   name = f'optflow_{l}')\
                              for l in range(self.num_levels)]
        # self.contexts = ContextNetwork()
        assert self.context in ['all', 'final'], 'context argument should be all/final'
        if self.context is 'all':
            self.context_nets = [ContextNetwork(name = f'context_{l}')\
                                 for l in range(self.num_levels)]
        else:
            self.context_net = ContextNetwork(name = 'context')

    def __call__(self, images_0, images_1):
        with tf.variable_scope(self.name) as vs:

            pyramid_0 = self.fp_extractor(images_0, reuse = False)
            pyramid_1 = self.fp_extractor(images_1)

            flows = []
            # coarse to fine processing
            for l, (feature_0, feature_1) in enumerate(zip(pyramid_0, pyramid_1)):
                print(f'Level {l}')
                b, h, w, _ = tf.unstack(tf.shape(feature_0))
                
                if l == 0:
                    flow = tf.zeros((b, h, w, 2), dtype = tf.float32)
                else:
                    flow = tf.image.resize_bilinear(flow, (h, w))*2

                # warping -> costvolume -> optical flow estimation
                feature_1_warped = self.warp_layer(feature_1, flow)
                cost = self.cv_layer(feature_0, feature_1_warped)
                feature, flow = self.of_estimators[l](feature_0, cost, flow)

                # context considering process all/final
                if self.context is 'all':
                    flow = self.context_nets[l](feature, flow)
                elif l == self.output_level: 
                    flow = self.context_net(feature, flow)

                flows.append(flow)
                
                # stop processing at the defined level
                if l == self.output_level:
                    upscale = 2**(self.num_levels - self.output_level)
                    print(f'Finally upscale flow by {upscale}.')
                    finalflow = tf.image.resize_bilinear(flow, (h*upscale, w*upscale))*upscale
                    break

            return finalflow, flows, pyramid_0

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


class PWCDCNet(object):
    def __init__(self, num_levels = 6, search_range = 4,
                 warp_type = 'bilinear', use_dc = False,
                 output_level = 4, name = 'pwcdcnet'):
        self.num_levels = num_levels
        self.s_range = search_range
        self.warp_type = warp_type
        self.use_dc = use_dc
        assert output_level < num_levels, 'Should set output_level < num_levels'
        self.output_level = output_level
        self.name = name

        self.fp_extractor = FeaturePyramidExtractor_custom(self.num_levels)
        self.warp_layer = WarpingLayer(self.warp_type)
        self.cv_layer = CostVolumeLayer(search_range)
        self.of_estimators = [OpticalFlowEstimator_custom(use_dc = self.use_dc, name = f'optflow_{l}')\
                              for l in range(self.num_levels)]
        self.context = ContextNetwork(name = 'context')
        # Upscale factors from deep -> shallow level
        self.scales = [None, 0.625, 1.25, 2.5, 5.0, 10., 20.]

    def __call__(self, images_0, images_1, with_features = False, reuse = False):
        with tf.variable_scope(self.name, reuse = reuse) as vs:
            pyramid_0 = self.fp_extractor(images_0, reuse = reuse)
            pyramid_1 = self.fp_extractor(images_1)

            flows_pyramid = []
            flows_up, features_up = None, None
            for l, (features_0, features_1) in enumerate(zip(pyramid_0, pyramid_1)):
                print(f'Level {l}')

                # Warping operation
                if l == 0:
                    features_1_warped = features_1
                else:
                    features_1_warped = self.warp_layer(features_1, flows_up*self.scales[l])

                # Cost volume calculation
                cv = self.cv_layer(features_0, features_1_warped)
                # Optical flow estimation
                if l < self.output_level:
                    flows, flows_up, features_up \
                        = self.of_estimators[l](cv, features_0, flows_up, features_up)
                else:
                    # At output level
                    flows, features = self.of_estimators[l](cv, features_0, flows_up, features_up,
                                                            is_output = True)
                    # Context processing
                    flows = self.context(flows, features)
                    flows_pyramid.append(flows)
                    # Obtain finally scale-adjusted flow
                    upscale = 2**(self.num_levels-self.output_level)
                    _, h, w, _ = tf.unstack(tf.shape(flows))
                    flows_final = tf.image.resize_bilinear(flows, (h*upscale, w*upscale))*20.

                    if with_features:
                        return flows_final, flows_pyramid, pyramid_0
                    else:
                        return flows_final, flows_pyramid

                flows_pyramid.append(flows)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
