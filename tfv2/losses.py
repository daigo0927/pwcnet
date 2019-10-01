import tensorflow as tf


def l1loss(x, y):
    return tf.reduce_mean(tf.abs(x-y))

def l2loss(x, y):
    return tf.reduce_mean(tf.norm(x-y, ord=2, axis=-1))

def end_point_error(flow_true, flow_pred):
    """ Compute end point error between true/predicted flow
    Args:
      - flow_gt: ground truth optical flows
      - flow: predicted flows
    Returns:
      - loss value
    """
    return tf.reduce_mean(tf.norm(flow_true-flow_pred, ord=2, axis=-1))


def avgpool(x, scale):
    return tf.nn.avg_pool2d(x, scale, scale, 'SAME')


def multiscale_loss(flow_true,
                    flow_pred_list,
                    weight=0.32):
    loss =  0
    flow_true = flow_true*0.05
    
    for l, flow in enumerate(flow_pred_list):
        w = weight / 2**l
        scale = 4*(2**l)
        flow_t = avgpool(flow_true, scale)
        loss += w*l2loss(flow_t, flow)

    return loss
        

def multirobust_loss(flow_true,
                     flow_pred_list,
                     weight=0.32,
                     epsilon=0.01,
                     q=0.4):
    loss = 0
    flow_true = flow_true*0.05
    
    for l, flow in enumerate(flow_pred_list):
        w = weight / 2**l
        scale = 4*(2**l)
        flow_t = avgpool(flow_true, scale)
        loss += w*(l1loss(flow_t, flow)+epsilon)**q

    return loss
    
