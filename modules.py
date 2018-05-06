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
        self.base_filters = 16
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:
            feature_pyramid = []
            for l in range(self.num_levels):
                x = _conv_block(self.base_filters*2**l, (3, 3),
                                (2, 2), batch_norm)(x)
                x = _conv_block(self.base_filters*2**l, (3, 3),
                                (1, 1), batch_norm)(x)
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
            warped_indices = tf.stack([grid_b, warped_y, warped_x], axis = 3)
            
            warped_x = tf.gather_nd(x, warped_indices)
            return warped_x

            
class CostVolumeLayer(object):

    def __init__(self, search_range = 4, name = 'cost_volume'):
        self.s_range = search_range
        self.name = name

    def __call__(self, x, warped):
        with tf.variable_scope(self.name) as vs:
            b, h, w, f = tf.unstack(tf.shape(x))
            cost_length = (2*self.s_range+1)**2
            # cost = tf.Variable(tf.zeros(shape = (b, h, w, cost_length)),
            #                    validate_shape = False)
            cost = [tf.Variable(tf.zeros(shape = (b, h, w)),
                                validate_shape = False) for _ in range(cost_length)]

            # cost = tf.Variable(tf.zeros((b, h, w)), validate_shape = False)
            # cost_volume = [0]*2
            # c_ = [0]*cost_length
            # c_[0] = tf.assign(cost[:,:,:,0], tf.reduce_sum(warped*x, axis = 3))[:,:,:,0]
            cost[0] = tf.assign(cost[0], tf.reduce_sum(warped*x, axis = 3))

            # cost[1] = tf.assign(cost[:, 5:, 5:], tf.reduce_sum(warped[:, :-5, :-5]*x[:,5:,5:], axis = 3))
            
            # tf.assign(cost[:,:,:,0], tf.reduce_sum(warped*x, axis = 3))

            I = 1
            for i in range(1, self.s_range+1):
                cost[I] = tf.assign(cost[I][:,i:,:],
                                    tf.reduce_sum(warped[:,:-i,:,:] * x[:,i:,:,:], axis = 3)); I+=1
                cost[I] = tf.assign(cost[I][:,:-i,:],
                                    tf.reduce_sum(warped[:,i:,:,:] * x[:,:-i,:,:], axis = 3)); I+=1
                cost[I] = tf.assign(cost[I][:,:,i:],
                                    tf.reduce_sum(warped[:,:,:-i,:] * x[:,:,i:,:], axis = 3)); I+=1
                cost[I] = tf.assign(cost[I][:,:,:-i],
                                    tf.reduce_sum(warped[:,:,i:,:] * x[:,:,:-i,:], axis = 3)); I+=1

                for j in range(1, self.s_range+1):
                    cost[I] = tf.assign(cost[I][:,i:,j:],
                                        tf.reduce_sum(warped[:,:-i,:-j,:] * x[:,i:,j:,:], axis = 3)); I+=1
                    cost[I] = tf.assign(cost[I][:,:-i,:-j],
                                        tf.reduce_sum(warped[:,i:,j:,:] * x[:,:-i,:-j,:], axis = 3)); I+=1
                    cost[I] = tf.assign(cost[I][:,i:,:-j],
                                        tf.reduce_sum(warped[:,:-i,j:,:] * x[:,i:,:-j,:], axis = 3)); I+=1
                    cost[I] = tf.assign(cost[I][:,:-i,j:],
                                        tf.reduce_sum(warped[:,i:,:-j,:] * x[:,:-i,j:,:], axis = 3)); I+=1
            
            # I = 1
            # for i in range(1, self.s_range+1):
            #     c_[I] = tf.assign(cost[:,i:,:,I],
            #                       tf.reduce_sum(warped[:,:-i,:,:] * x[:,i:,:,:], axis = 3))[:,:,:,I]; I+=1
            #     c_[I] = tf.assign(cost[:,:-i,:,I],
            #                       tf.reduce_sum(warped[:,i:,:,:] * x[:,:-i,:,:], axis = 3))[:,:,:,I]; I+=1
            #     c_[I] = tf.assign(cost[:,:,i:,I],
            #                       tf.reduce_sum(warped[:,:,:-i,:] * x[:,:,i:,:], axis = 3))[:,:,:,I]; I+=1
            #     c_[I] = tf.assign(cost[:,:,:-i,I],
            #                       tf.reduce_sum(warped[:,:,i:,:] * x[:,:,:-i,:], axis = 3))[:,:,:,I]; I+=1

            #     for j in range(1, self.s_range+1):
            #         c_[I] = tf.assign(cost[:,i:,j:,I],
            #                           tf.reduce_sum(warped[:,:-i,:-j,:] * x[:,i:,j:,:], axis = 3))[:,:,:,I]; I+=1
            #         c_[I] = tf.assign(cost[:,:-i,:-j,I],
            #                           tf.reduce_sum(warped[:,i:,j:,:] * x[:,:-i,:-j,:], axis = 3))[:,:,:,I]; I+=1
            #         c_[I] = tf.assign(cost[:,i:,:-j,I],
            #                           tf.reduce_sum(warped[:,:-i,j:,:] * x[:,i:,:-j,:], axis = 3))[:,:,:,I]; I+=1
            #         c_[I] = tf.assign(cost[:,:-i,j:,I],
            #                           tf.reduce_sum(warped[:,i:,:-j,:] * x[:,:-i,j:,:], axis = 3))[:,:,:,I]; I+=1

            # I = 1
            # for i in range(1, self.s_range+1):
            #     cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,:-i,:,:] * x[:,i:,:,:], axis = 3)); I+=1
            #     cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,i:,:,:] * x[:,:-i,:,:], axis = 3)); I+=1
            #     cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,:,:-i,:] * x[:,:,i:,:], axis = 3)); I+=1
            #     cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,:,i:,:] * x[:,:,:-i,:], axis = 3)); I+=1
                
            #     for j in range(1, self.s_range+1):
            #         cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,:-i,:-j,:] * x[:,i:,j:,:], axis = 3)); I+=1
            #         cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,i:,j:,:] * x[:,:-i,:-j,:], axis = 3)); I+=1
            #         cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,:-i,j:,:] * x[:,i:,:-j,:], axis = 3)); I+=1
            #         cost_volume[I] = tf.add(cost, tf.reduce_sum(warped[:,i:,:-j,:] * x[:,:-i,j:,:], axis = 3)); I+=1

            return tf.stack(cost, axis = 3) / cost_length
            
        

# class OpticalFlowEstimator(object):

#     def __init__(self, )
            
