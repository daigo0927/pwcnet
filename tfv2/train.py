import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tempfile

from datasets import MPISintel, Preprocess, Transform
from model import PWCNet
from losses import multiscale_loss, end_point_error
from utils import vis_flow, prepare_parser


def train(args):
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.debug:
            logdir = tmpdir
        else:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            logdir = f'logs/{datetime.now().strftime("%Y-%m-%dT%H:%M")}'
        _train(args, logdir)


def _train(args, logdir):
    preprocess = Preprocess()
    transform = Transform(original_size=(436, 1024),
                          random_scale=args.random_scale,
                          crop_shape=args.crop_shape,
                          horizontal_flip=args.horizontal_flip)
    
    dataset = MPISintel(dataset_dir=args.dataset_dir,
                        train_or_test='train',
                        mode=args.mode,
                        batch_size=args.batch_size,
                        validation_split=args.validation_split,
                        preprocess=preprocess,
                        transform=transform)

    model = PWCNet(name='pwcnet')

    optimizer = tf.keras.optimizers.Adam(lr=args.learning_rate)

    @tf.function
    def train_step(images1, images2, flows_true):
        with tf.GradientTape() as tape:
            flows_pred, flows_pred_list = model(images1, images2)
            loss = multiscale_loss(flows_true, flows_pred_list)
            l2decay = tf.reduce_sum([tf.nn.l2_loss(w) for w in model.trainable_weights])
            loss += args.gamma * l2decay
            epe = end_point_error(flows_true, flows_pred)
        grad = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))
        outputs = {'loss': loss, 'epe': epe}
        return outputs

    @tf.function
    def val_step(images1, images2, flows_true):
        flows_pred, _ = model(images1, images2)
        epe = end_point_error(flows_true, flows_pred)
        outputs = {'epe': epe,
                   'flows_true': flows_true,
                   'flows_pred': flows_pre}
        return outputs

    summary_writer = tf.summary.create_file_writer(logdir)

    n_batches = np.ceil(len(dataset)*(1-args.validation_split)/args.batch_size)
    for e in tqdm(range(args.epochs)):
        for i, (images1, images2, flows_true) in enumerate(dataset.train_loader):
            tout = train_step(images1, images2, flows_true)

        for i, (images1, images2, flows_true) in enumerate(dataset.val_loader):
            vout = train_step(images1, images2, flows_true)

        with summary_writer.as_default():
            tf.summary.scalar('train/loss', tout['loss'])
            tf.summary.scalar('train/epe', tout['epe'])
            tf.summary.scalar('val/epe', vout['epe'])
            ftrue_img = tf.numpy_function(vis_flow, [vout['flows_true'][0]], tf.uint8)
            fpred_img = tf.numpy_function(vis_flow, [vout['flows_pred'][0]], tf.uint8)
            tf.summary.image('val/flow_true', ftrue_img)
            tf.summary.image('val/flow_pred', fpred_img)
            summary_writer.flush()


if __name__ == '__main__':
    parser = prepare_parser()
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    train(args)
