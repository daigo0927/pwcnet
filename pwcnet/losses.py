import tensorflow as tf


def l1_loss(x, y):
    loss = tf.reduce_sum(tf.norm(x - y, ord=1, axis=-1), axis=(1, 2))
    return tf.reduce_mean(loss)


def l2_loss(x, y):
    loss = tf.reduce_sum(tf.norm(x - y, ord=2, axis=-1), axis=(1, 2))
    return tf.reduce_mean(loss)


def end_point_error(flow_true, flow_pred):
    return tf.reduce_mean(tf.norm(flow_true - flow_pred, ord=2, axis=-1))


def multiscale_loss(flow_true,
                    flow_pred_pyramid,
                    weights=[0.005, 0.01, 0.02, 0.08, 0.32]):
    """ Multiscale (L2) loss for hierarchical optical flow estimation.

    Args:
      flow_true: A tensor representing a batch of true flow.
      flow_pred_pyramid: A list of tensors representing a batch of
        predicted optical flow (fine to coarse order).
      weights: A list of weights for summing flow pyramid (fine to coarse order).

    Returns:
      A scalar loss tensor.
    """
    flow_true = flow_true / 20.0
    loss = 0.0
    for weight, flow_pred in zip(weights, flow_pred_pyramid):
        _, h, w, _ = tf.unstack(tf.shape(flow_pred))
        downflow_true = tf.image.resize(flow_true, (h, w))
        loss += weight * l2_loss(downflow_true, flow_pred)
    return loss


def multirobust_loss(flow_true,
                     flow_pred_pyramid,
                     weights=[0.005, 0.01, 0.02, 0.08, 0.32],
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
