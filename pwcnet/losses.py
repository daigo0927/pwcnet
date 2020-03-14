import tensorflow as tf


def l1_loss(x, y):
    return tf.reduce_mean(tf.norm(x - y, ord=1, axis=-1))


def l2_loss(x, y):
    return tf.reduce_mean(tf.norm(x - y, ord=2, axis=-1))


def end_point_error(flow_true, flow_pred):
    return l2_loss(flow_true, flow_pred)


def multiscale_loss(flow_true,
                    flow_pred_pyramid,
                    weights=[0.32, 0.08, 0.02, 0.01, 0.005]):
    flow_true = flow_true / 20.0
    loss = tf.zeros([], dtype=tf.float32)
    for weight, flow_pred in zip(weights, flow_pred_pyramid):
        _, h, w, _ = tf.unstack(tf.shape(flow_pred))
        downflow_true = tf.image.resize(flow_true, (h, w))
        loss += weight * l2_loss(downflow_true, flow_pred)
    return loss


def multirobust_loss(flow_true,
                     flow_pred_pyramid,
                     weights=[0.32, 0.08, 0.02, 0.01, 0.005],
                     epsilon=0.01,
                     q=0.4):
    flow_true = flow_true / 20.0
    loss = 0.0
    for weight, flow_pred in zip(weights, flow_pred_pyramid):
        _, h, w, _ = tf.unstack(tf.shape(flow_pred))
        downflow_true = tf.image.resize(flow_true, (h, w))
        l = l1_loss(downflow_true, flow_pred)
        loss += weight * (l + epsilon)**q
    return loss
