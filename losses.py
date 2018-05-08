import tensorflow as tf

def L1loss(x, y):
    return tf.reduce_mean(tf.norm(x-y, ord = 1, axis = 3))

def L2loss(x, y):
    return tf.reduce_mean(tf.norm(x-y, ord = 2, axis = 3))

# end point error, same as L2 loss
def EPE(flows_gt, flows):
    return tf.reduce_mean(tf.norm(flows_gt-flows, ord = 2, axis = 3))

def multiscale_loss(flows_gt, flows_pyramid,
                    weights, name = 'multiscale_loss'):
    with tf.name_scope(name) as ns:
        _, H, W, _ = tf.unstack(tf.shape(flows_gt))
        loss, epe = 0, 0
        loss_levels, epe_levels = [], []

        # processing from coarce-to-fine level
        for l, (w, fs) in enumerate(zip(weights, flows_pyramid)):
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt = tf.image.resize_bilinear(flows_gt, (h, w))/(H/h)
            
            loss_level = L2loss(fs_gt, fs)
            loss += w * loss_level
            loss_levels.append(loss_level)

            epe = EPE(fs_gt, fs)
            epe_levels.append(epe)

        return loss, epe, loss_levels, epe_levels

def multirobust_loss(flows_gt, flows_pyramid,
                     weights, epsilon = 0.01,
                     q = 0.4, name = 'multirobust_loss'):
    with tf.name_scope(name) as ns:
        for l, (w, fs) in enumerate(zip(weights, flows_pyramid)):
            _, h, w, _ = tf.unstack(tf.shape(fs))
            fs_gt = tf.image.resize_bilinear(flows_gt, (h, w))/(H/h)
            
            loss_level = (L1loss(fs_gt, fs) + epsilon)**q
            loss += w * loss_level
            loss_levels.append(loss_level)

            epe = EPE(fs_gt, fs)
            epe_levels.append(epe)

        return loss, epe, loss_levels, epe_levels
