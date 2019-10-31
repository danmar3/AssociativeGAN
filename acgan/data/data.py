import tensorflow as tf
import tensorflow_datasets as tfds
from . import imagenet2012

DATA_DIR = None

def set_datadir(data_dir):
    global DATA_DIR
    DATA_DIR = data_dir

def load_mnist(batch_size, split=tfds.Split.TRAIN):
    dataset, info = tfds.load(
        'mnist', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .map(lambda x: (tf.cast(x['image'], tf.float32)-127.5)/127.5)\
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_mnist32(batch_size, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)
        batch = tf.image.resize_bilinear(
            batch, size=(32, 32), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch
    dataset, info = tfds.load(
        'mnist', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_fashion_mnist(batch_size, split=tfds.Split.TRAIN):
    dataset, info = tfds.load(
        'fashion_mnist', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .map(lambda x: (tf.cast(x['image'], tf.float32)-127.5)/127.5)\
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a(batch_size, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)
        batch = tf.image.central_crop(batch, central_fraction=0.7)
        batch = tf.image.resize_bilinear(
            batch, size=(64, 64), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch
    dataset, info = tfds.load(
        'celeb_a', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a_128_cropped(batch_size, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)
        batch = tf.image.central_crop(batch, central_fraction=0.7)
        batch = tf.image.resize_bilinear(
            batch, size=(128, 128), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch
    dataset, info = tfds.load(
        'celeb_a', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a_128(batch_size, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)
        # batch = tf.image.central_crop(batch, central_fraction=0.7)
        batch = tf.image.resize_bilinear(
            batch, size=(128, 128), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch
    dataset, info = tfds.load(
        'celeb_a', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_hd_512(batch_size, split=tfds.Split.TRAIN,
                      folder='datasets/celeb_hd'):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)
        batch = tf.image.resize_bilinear(
            batch, size=(512, 512), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch
    dataset, info = tfds.load(
        'celeb_a', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load(name, batch_size):
    def load_imagenet(batch_size, resolution):
        return imagenet2012.load_imagenet2012(
            data_dir=DATA_DIR,
            batch_size=batch_size,
            crop=True, resolution=resolution)
    loaders = {
        'celeb_a':
        load_celeb_a_128_cropped,
        'imagenet_512':
        lambda batch_size: load_imagenet(batch_size, 512),
        'imagenet_256':
        lambda batch_size: load_imagenet(batch_size, 256),
        'imagenet_128':
        lambda batch_size: load_imagenet(batch_size, 128)
    }
    return loaders[name](batch_size)
