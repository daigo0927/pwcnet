import tensorflow as tf
import pdb 

def L1loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.reduce_sum(tf.norm(x-y, ord = 1, axis = 3), axis = (1,2)))

def L2loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.reduce_sum(tf.norm(x-y, ord = 2, axis = 3), axis = (1,2)))

# end point error, each element is same as L2 loss
def EPE(flows_gt, flows):
    # Given ground truth and estimated flow must be unscaled
    return tf.reduce_mean(tf.norm(flows_gt-flows, ord = 2, axis = 3))

def multiscale_loss(flows_gt, flows_pyramid,
                    weights, name = 'multiscale_loss'):
    # Argument flows_gt must be unscaled, scaled inside of this loss function
    with tf.name_scope(name) as ns:
        # Scale the ground truth flow, stated Sec.4 in the original paper
        flows_gt_scaled = flows_gt/20.

        # Calculate mutiscale loss
        loss = 0.
        for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
            # Downsampling the scaled ground truth flow
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt_down = tf.image.resize_nearest_neighbor(flows_gt_scaled, (h, w))
            # Calculate l2 loss
            loss += weight*L2loss(fs_gt_down, fs)

        return loss

def multirobust_loss(flows_gt, flows_pyramid,
                     weights, epsilon = 0.01,
                     q = 0.4, name = 'multirobust_loss'):
    with tf.name_scope(name) as ns:
        flows_gt_scaled = flows_gt/20.
        loss = 0.
        for l, (weight, fs) in enumerate(zip(weights, flows_pyramid)):
            # Downsampling the scaled ground truth flow
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt_down = tf.image.resize_nearest_neighbor(flows_gt_scaled, (h, w))
            # Calculate l1 loss
            _l = L1loss(fs_gt_down, fs)
            loss += weight*(loss_level+epsilon)**q

        return loss
