import os
import sys
import imageio
import numpy as np
import tensorflow as tf
from pathlib import Path
from itertools import groupby, islice


def load_flow(uri):
    """ Function to load flow file.

    Args:
      uri: A string of path to the flow file.
    
    Returns:
      Output tensor
    """
    with open(uri, 'rb') as f:
        magic = float(np.fromfile(f, np.float32, count = 1)[0])
        if magic == 202021.25:
            w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
            data = np.fromfile(f, np.float32, count = h*w*2)
            data.resize((h, w, 2))
            return data
        return None


def load_flow_tf(uri):
    info = tf.io.decode_raw(tf.io.read_file(uri),
                            out_type=tf.int32, fixed_length=12)
    _, w, h = tf.unstack(info)
    data = tf.io.decode_raw(tf.io.read_file(uri), out_type=tf.float32)
    flow = tf.reshape(data[3:], [h, w, 2])
    return flow


def window(seq, n=2):
    """ Returns a sliding window (of width n) over data from the iterable.
    If n=2: s -> (s0,s1,...s[n-1]), (s1,s2,...,sn)

    Args:
      seq: A list containing a sequence.
      n: A scalar, the number of sliding steps
    
    Returns:
      A list containing a sequence of pairwise images.
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def _parse(sample):
    image1 = tf.io.decode_image(tf.io.read_file(sample[0]))
    image2 = tf.io.decode_image(tf.io.read_file(sample[1]))
    flow = load_flow_tf(sample[2])
    return [image1, image2, flow]


def normalize(image1, image2, flow, flow_scaler=20.0):
    image1 = tf.cast(image1, tf.float32)/255.0
    image2 = tf.cast(image2, tf.float32)/255.0
    flow = flow/flow_scaler
    return [image1, image2, flow]


def build_sintel_dataset(path, mode='clean'):
    """ Build a MPI-Sintel dataset from a directory.
    
    Args:
      path: A string path to a directory containing a dataset.
      mode: A string 'clean' or 'final' specifying difficulties.

    Returns:
      tf.data.Dataset object containing the dataset.
    """
    d = Path(path)
    d_image = d / 'training' / mode

    collections_of_scenes = sorted(map(str, d_image.glob('**/*.png')))
    collections = [list(g) for k, g
                   in groupby(collections_of_scenes, lambda x: x.split('/')[-2])]
    samples = [(*i, i[0].replace(mode, 'flow').replace('.png', '.flo'))
               for collection in collections for i in window(collection, 2)]
    
    dataset = tf.data.Dataset.from_tensor_slices(samples)
    dataset = dataset.map(_parse)
    return dataset


def random_crop(image1, image2, flow, target_size):
    inputs = tf.concat([image1, image2, flow], axis=-1)
    inputs = tf.image.random_crop(inputs, size=[*target_size, 8])
    image1, image2, flow = tf.split(inputs, [3, 3, 2], axis=-1)
    return [image1, image2, flow]


def hflip(image1, image2, flow):
    inputs = tf.concat([image1, image2, flow], axis=-1)
    inputs = tf.image.random_flip_left_right(inputs)
    image1, image2, fx, fy = tf.split(inputs, [3, 3, 1, 1], axis=-1)
    flow = tf.concat([-1*fx, fy], axis=-1)
    return [image1, image2, flow]


def random_horizontal_flip(image1, image2, flow):
    do_flip = tf.greater(tf.random.uniform([]), 0.5)
    outputs = tf.cond(do_flip,
                      lambda: hflip(image1, image2, flow),
                      lambda: [image1, image2, flow])
    return outputs


def vflip(image1, image2, flow):
    inputs = tf.concat([image1, image2, flow], axis=-1)
    inputs = tf.image.random_flip_up_down(inputs)
    image1, image2, fx, fy = tf.split(inputs, [3, 3, 1, 1], axis=-1)
    flow = tf.concat([fx, -1*fy], axis=-1)
    return [image1, image2, flow]


def random_vertical_flip(image1, image2, flow):
    do_flip = tf.greater(tf.random.uniform([]), 0.5)
    outputs = tf.cond(do_flip,
                      lambda: vflip(image1, image2, flow),
                      lambda: [image1, image2, flow])
    return outputs
