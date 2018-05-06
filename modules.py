import tensorflow as tf

from utils import get_grid


def _conv_block(filters, kernel_size = (3, 3), strides = (1, 1), batch_norm = False):
    def f(x):
        x = tf.layers.Conv2D(filters, kernel_size,
                             strides, 'same')(x)
        if batch_norm:
            x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x
    return f


class FeaturePyramidExtractor(object):

    def __init__(self, num_levels = 6, batch_norm = False, name = 'fp_extractor'):
        self.num_levels = num_levels
        self.filters_list = [16, 32, 64, 96, 128, 192]
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
                
            feature_pyramid = []
            for l in range(self.num_levels):
                x = _conv_block(self.filters_list[l], (3, 3),
                                (2, 2), self.batch_norm)(x)
                x = _conv_block(self.filters_list[l], (3, 3),
                                (1, 1), self.batch_norm)(x)
                feature_pyramid.append(x)

            # return feature pyramid by ascent order
            return feature_pyramid[::-1] 

class WarpingLayer(object):

    def __init__(self, name = 'warping'):
        self.name = name

    def __call__(self, x, flow):
        # expect shape
        # x:(#batch, height, width, #channel)
        # flow:(#batch, height, width, 2)
        with tf.variable_scope(self.name) as vs:
            grid_b, grid_y, grid_x = get_grid(x)
            warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
            warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
            warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
            warped_x = tf.gather_nd(x, warped_indices)
            return warped_x

            
class CostVolumeLayer(object):

    def __init__(self, search_range = 4, name = 'cost_volume'):
        self.s_range = search_range
        self.name = name

    def __call__(self, x, warped, reuse = True):
        with tf.variable_scope(self.name) as vs:
            b, h, w, f = tf.unstack(tf.shape(x))
            cost_length = (2*self.s_range+1)**2
            cost_init = [tf.Variable(tf.zeros(shape = (b, h, w)),
                                     validate_shape = False) for _ in range(cost_length)]
            tf.variables_initializer(cost_init)

            cost = [0]*cost_length
            cost[0] = tf.assign(cost_init[0], tf.reduce_sum(warped*x, axis = 3))

            I = 1
            for i in range(1, self.s_range+1):
                cost[I] = tf.assign(cost_init[I][:,i:,:],
                                    tf.reduce_sum(warped[:,:-i,:,:] * x[:,i:,:,:], axis = 3)); I+=1
                cost[I] = tf.assign(cost_init[I][:,:-i,:],
                                    tf.reduce_sum(warped[:,i:,:,:] * x[:,:-i,:,:], axis = 3)); I+=1
                cost[I] = tf.assign(cost_init[I][:,:,i:],
                                    tf.reduce_sum(warped[:,:,:-i,:] * x[:,:,i:,:], axis = 3)); I+=1
                cost[I] = tf.assign(cost_init[I][:,:,:-i],
                                    tf.reduce_sum(warped[:,:,i:,:] * x[:,:,:-i,:], axis = 3)); I+=1

                for j in range(1, self.s_range+1):
                    cost[I] = tf.assign(cost_init[I][:,i:,j:],
                                        tf.reduce_sum(warped[:,:-i,:-j,:] * x[:,i:,j:,:], axis = 3)); I+=1
                    cost[I] = tf.assign(cost_init[I][:,:-i,:-j],
                                        tf.reduce_sum(warped[:,i:,j:,:] * x[:,:-i,:-j,:], axis = 3)); I+=1
                    cost[I] = tf.assign(cost_init[I][:,i:,:-j],
                                        tf.reduce_sum(warped[:,:-i,j:,:] * x[:,i:,:-j,:], axis = 3)); I+=1
                    cost[I] = tf.assign(cost_init[I][:,:-i,j:],
                                        tf.reduce_sum(warped[:,i:,:-j,:] * x[:,:-i,j:,:], axis = 3)); I+=1

            return tf.stack(cost, axis = 3) / cost_length
            
        
class OpticalFlowEstimator(object):

    def __init__(self, batch_norm, name = 'of_estimator'):
        self.batch_norm = batch_norm
        self.name = name

    def __call__(self, x, cost, flow, reuse = True):
        with tf.variable_scope(self.name) as vs:
            flow = tf.cast(flow, dtype = tf.float32)
            x = tf.concat([x, cost, flow], axis = 3)
            x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(128, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(96, (3, 3), (1, 1), self.batch_norm)(x)
            x = _conv_block(64, (3, 3), (1, 1), self.batch_norm)(x)
            feature = _conv_block(32, (3, 3), (1, 1), self.batch_norm)(x)
            flow = tf.layers.Conv2D(2, (3, 3), (1, 1), padding = 'same')(feature)

            return feature, flow # x:processed feature, w:processed flow

    
class ContextNetwork(object):

    def __init__(self, name = 'context'):
        self.name = name

    def __call__(self, feature, flow, reuse = True):
        with tf.variable_scope(self.name) as vs:
            x = tf.concat([feature, flow], axis = 3)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (2, 2))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(128, (3, 3), (1, 1),'same',
                                 dilation_rate = (4, 4))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(96, (3, 3), (1, 1),'same',
                                 dilation_rate = (8, 8))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(64, (3, 3), (1, 1),'same',
                                 dilation_rate = (16, 16))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(32, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            x = tf.nn.leaky_relu(x, 0.2)
            x = tf.layers.Conv2D(2, (3, 3), (1, 1),'same',
                                 dilation_rate = (1, 1))(x)
            return x+flow
