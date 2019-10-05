import tensorflow as tf
import random
from glob import glob
from abc import abstractmethod, ABCMeta
from functools import partial
from itertools import islice, groupby


class Base(metaclass=ABCMeta):
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 batch_size=1,
                 validation_split=0.1,
                 preprocess=None,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - batch_size: int for batch size
          - validation_split: validation split ratio, should be in [0, 1]
          - preprocess: executed for all samples
          - transform: executed for only train samples
        """
        self.dataset_dir = dataset_dir
        self.train_or_test = train_or_test
        self.batch_size = batch_size
        self.validation_split = validation_split

        def nofn(*x): return x
        self.preprocess = preprocess if preprocess is not None else nofn
        self.transform = transform if transform is not None else nofn

        print('Building a dataset pipeline ...')
        self._get_filenames()
        print('Found {} images.'.format(len(self)))
        self._build()
        print('Done.')

    def __len__(self):
        return len(self.samples[0])

    @abstractmethod
    def _get_filenames(self):
        """ implement self.samples """; ...
        
    def read(self, imagefile1, imagefile2, flowfile=None):
        image1 = tf.io.decode_image(tf.io.read_file(imagefile1))
        image2 = tf.io.decode_image(tf.io.read_file(imagefile1))
        if flowfile is not None:
            flow_byte = tf.io.read_file(flowfile)
            _, w, h = tf.unstack(tf.io.decode_raw(flow_byte, tf.int32, fixed_length=12))
            flow_raw = tf.io.decode_raw(flow_byte, tf.float32)[3:]
            flow = tf.reshape(flow_raw, (h, w, 2))
            return image1, image2, flow
        else:
            return image1, image2

    def _build(self):
        if self.train_or_test == 'train':
            idx = int(len(self) * (1 - self.validation_split))
            dataset = tf.data.Dataset.from_tensor_slices(self.samples)
            dataset = dataset.shuffle(len(self))
            self.train_loader = dataset.take(idx)\
              .map(self.read, tf.data.experimental.AUTOTUNE)\
              .map(self.preprocess, tf.data.experimental.AUTOTUNE)\
              .map(self.transform, tf.data.experimental.AUTOTUNE)\
              .prefetch(self.batch_size)\
              .batch(self.batch_size)
            self.val_loader = dataset.skip(idx)\
              .map(self.read, tf.data.experimental.AUTOTUNE)\
              .map(self.preprocess, tf.data.experimental.AUTOTUNE)\
              .prefetch(self.batch_size)\
              .batch(1)
        else:
            self.test_loader = tf.data.Dataset.from_tensor_slices(self.samples)\
              .map(self.read, tf.data.experimental.AUTOTUNE)\
              .map(self.preprocess, tf.data.experimental.AUTOTUNE)\
              .prefetch(self.batch_size)\
              .batch(1)


class Preprocess:
    def __init__(self, base_shape):
        self.base_shape = base_shape

    def __call__(self, image1, image2, flow=None):
        image1 = tf.cast(image1, tf.float32)
        image2 = tf.cast(image2, tf.float32)
        image1 /= 255.0
        image2 /= 255.0
        image1 = tf.image.pad_to_bounding_box(image1, 0, 0, *self.base_shape)
        image2 = tf.image.pad_to_bounding_box(image2, 0, 0, *self.base_shape)
        if flow is not None:
            flow = tf.image.pad_to_bounding_box(flow, 0, 0, *self.base_shape)

            return image1, image2, flow
        else:
            return image1, image2

        
class Transform:
    def __init__(self,
                 crop_shape=None,
                 horizontal_flip=False):
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip

    def __call__(self, image1, image2, flow):
        
        if self.crop_shape:
            x = tf.concat([image1, image2, flow], axis=-1)
            x = tf.image.random_crop(x, [*self.crop_shape, 8])
            image1, image2, flow = tf.split(x, [3, 3, 2], axis=-1)

        if self.horizontal_flip:
            x = tf.concat([image1, image2, flow], axis=-1)
            x = tf.image.random_flip_left_right(x)
            image1, image2, fx, fy = tf.split(x, [3, 3, 1, 1], axis=-1)
            flow = tf.concat([-1*fx, fy], axis=-1)

        return image1, image2, flow


def window(seq, n = 2):
    """ Returns a sliding window """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for el in it:
        result = result[1:] + (el,)
        yield result


class MPISintel(Base):
    """
    TensorFlow dataset pipeline for MPI-Sintel dataset
    http://sintel.is.tue.mpg.de/
    """
    def __init__(self,
                 dataset_dir,
                 train_or_test,
                 mode,
                 batch_size=1,
                 validation_split=0.1,
                 preprocess=None,
                 transform=None):
        """
        Args:
          - dataset_dir: string of /path/to/dataset-directory
          - train_or_test: train or test argument
          - mode: target path specifing rendering format, clean/final
          - batch_size: int for batch size
          - validation_split: validation split ratio, should be in [0, 1]
          - preprocess: executed for all samples
          - transform: executed for only train samples
        """
        self.mode = mode
        super().__init__(dataset_dir=dataset_dir,
                         train_or_test=train_or_test,
                         batch_size=batch_size,
                         validation_split=validation_split,
                         preprocess=preprocess,
                         transform=transform)

    def _get_filenames(self):
        train_or_test = 'training' if self.train_or_test == 'train' else 'test'
        d = '/'.join([self.dataset_dir, train_or_test, self.mode])
        collections_of_scenes = sorted(glob(d+'/**/*.png'))
        collections = [list(g) for k, g in groupby(collections_of_scenes,
                                                   lambda x: x.split('/')[-2])]
        
        imagefiles1, imagefiles2, flowfiles = [], [], []
        for collection in collections:
            for i in window(collection, 2):
                imagefiles1.append(i[0])
                imagefiles2.append(i[1])
                f = i[0].replace(self.mode, 'flow').replace('.png', '.flo')
                flowfiles.append(f)

        if self.train_or_test == 'train':
            self.samples = (imagefiles1, imagefiles2, flowfiles)
        else:
            self.samples = (imagefiles1, imagefiles2)
