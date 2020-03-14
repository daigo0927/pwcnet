import os
import time
import yaml
import tempfile
import argparse
import tensorflow as tf
from shutil import copyfile
from functools import partial

from pwcnet.datasets import (build_sintel_dataset, scaling_image, random_crop,
                             random_horizontal_flip, random_vertical_flip,
                             concat_image)
from pwcnet.models import PWCNet
from pwcnet.losses import multiscale_loss, multirobust_loss, end_point_error


def train(config, logdir):
    sintel_dir = config['sintel']['directory']
    sintel_mode = config['sintel']['mode']

    crop_size = config['preprocess']['crop_size']

    model_params = config['model']

    epochs = config['train']['epochs']
    batch_size = config['train']['batch_size']
    optimizer_config = config['train']['optimizer']
    validation_split = config['train']['validation_split']

    dataset = build_sintel_dataset(sintel_dir, sintel_mode)

    dataset = dataset.map(scaling_image)\
        .map(partial(random_crop, target_size=crop_size))\
        .map(random_horizontal_flip)\
        .shuffle(500)\
        .batch(batch_size)\
        .repeat(epochs)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    model = PWCNet(**model_params)

    optimizer = tf.keras.optimizers.Adam(**optimizer_config)

    # model.compile(optimizer, multiscale_loss)
    # model.fit(dataset)

    @tf.function
    def train_step(inputs, flow_true):
        with tf.GradientTape() as tape:
            flow_pred_pyramid = model(inputs)
            loss = multiscale_loss(flow_true, flow_pred_pyramid)
        grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        return loss

    for i, (images_1, images_2, flows_true) in enumerate(dataset):
        loss = train_step([images_1, images_2], flows_true)
        print(f'{i+1}/10: loss: {loss}')
        if i == 0:
            break


def run(config_file, debug):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with tempfile.TemporaryDirectory() as tempdir:
        if debug:
            logdir = tempdir
        else:
            # Create output directory
            time_str = time.strftime('%Y%m%d_%H%M%S')
            logdir = os.path.join(config["output_dir"], time_str)
            if not os.path.exists(logdir):
                os.makedirs(logdir)

        # Save config file to output directory
        filename = config_file.split('/')[-1]
        copyfile(config_file, os.path.join(logdir, filename))

        print('--------------- Target config ------------------')
        print(yaml.dump(config))
        print('---------------------------------')
        print('Log directory: {}'.format(logdir))
        print('---------------------------------')

        # Run evaluation
        train(config, logdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',
                        type=str,
                        help='Experiment config file.')
    parser.add_argument('-d',
                        '--debug',
                        action='store_true',
                        help='Debug for script, clear all resulting files.')
    args = parser.parse_args()
    run(**vars(args))
