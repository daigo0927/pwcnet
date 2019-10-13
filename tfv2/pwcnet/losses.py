import tensorflow as tf


def l1loss(x, y):
    norm = tf.norm(x-y, ord=1, axis=-1)
    return tf.reduce_mean(tf.reduce_sum(norm, axis=(1, 2)))

def l2loss(x, y):
    norm = tf.norm(x-y, ord=2, axis=-1)
    return tf.reduce_mean(tf.reduce_sum(norm, axis=(1, 2)))

def end_point_error(flow_true, flow_pred):
    """ Compute end point error between true/predicted flow
    Args:
      - flow_gt: ground truth optical flows
      - flow: predicted flows
    Returns:
      - loss value
    """
    return tf.reduce_mean(tf.norm(flow_true-flow_pred, ord=2, axis=-1))


def multiscale_loss(flow_true,
                    flow_pred_list,
                    weights=[0.32, 0.08, 0.02, 0.01, 0.005]):
    loss =  0.0
    flow_true_scaled = flow_true*0.05
    
    for l, (weight, flow_p) in enumerate(zip(weights, flow_pred_list)):
        _, h, w, _ = tf.unstack(tf.shape(flow_p))
        flow_t = tf.image.resize(flow_true_scaled, (h, w), method='bilinear')
        loss += weight*l2loss(flow_t, flow_p)

    return loss
        

def multirobust_loss(flow_true,
                     flow_pred_list,
                     weights=[0.32, 0.08, 0.02, 0.01, 0.005],
                     epsilon=0.01,
                     q=0.4):
    loss = 0.0
    flow_true_scaled = flow_true*0.05
    
    for l, (weight, flow_p) in enumerate(zip(weights, flow_pred_list)):
        _, h, w, _ = tf.unstack(tf.shape(flow_p))
        flow_t = tf.image.resize(flow_true_scaled, (h, w), method='bilinear')
        loss += weight*(l1loss(flow_t, flow)+epsilon)**q

    return loss
