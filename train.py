import os
import time
import yaml
import tempfile
import argparse
import tensorflow as tf
from shutil import copyfile
from functools import partial

from pwcnet.datasets import (build_sintel_dataset, scaling, random_crop,
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
    gamma = config['train']['gamma']
    validation_split = config['train']['validation_split']

    dataset, samples = build_sintel_dataset(sintel_dir,
                                            sintel_mode,
                                            with_files=True)
    data_size = len(samples)

    dataset = dataset.map(scaling)\
        .map(partial(random_crop, target_size=crop_size))\
        .map(random_horizontal_flip)\
        .shuffle(data_size)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    model = PWCNet(**model_params)

    optimizer = tf.keras.optimizers.Adam(**optimizer_config)

    train_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    train_epe = tf.keras.metrics.Mean('epe', dtype=tf.float32)
    # val_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    # val_epe = tf.keras.metrics.Mean('epe', dtype=tf.float32)

    train_writer = tf.summary.create_file_writer(logdir + '/train')
    # val_writer = tf.summary.create_file_writer(logdir + '/val')

    @tf.function
    def train_step(inputs, flow_true):
        with tf.GradientTape() as tape:
            flow_pred_pyramid, flow_pred = model(inputs)
            loss = multiscale_loss(flow_true, flow_pred_pyramid)
            weights_norm = tf.reduce_sum(
                [tf.nn.l2_loss(w) for w in model.trainable_weights])
            loss += gamma * weights_norm
            epe = end_point_error(flow_true, flow_pred)
        grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        train_loss(loss)
        train_epe(epe)
        return loss, epe

    # @tf.function
    # def val_step(inputs, flow_true):
    #     flow_pred_pyramid, flow_pred = model(inputs)
    #     loss = multiscale_loss(flow_true, flow_pred_pyramid)
    #     epe = end_point_error(flow_true, flow_pred)
    #     val_loss(loss)
    #     val_epe(epe)

    for e in range(epochs):
        for b, (images_1, images_2, flows_true) in enumerate(dataset):
            loss, epe = train_step([images_1, images_2], flows_true)
            loss, epe = loss.numpy(), epe.numpy()
            print(f'Epoch: {e+1}/{epochs} {b}step: loss: {loss}, epe: {epe}')
        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=e)
            tf.summary.scalar('epe', train_epe.result(), step=e)
        train_loss.reset_states()
        train_epe.reset_states()


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
