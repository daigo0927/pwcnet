import sys
import tensorflow as tf

def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def show_progress(epoch, batch, batch_total, loss, epe):
    sys.stdout.write(f'\r{epoch} epoch: [{batch}/{batch_total}, loss: {loss}, epe: {acc}]')
    sys.stdout.flush()
