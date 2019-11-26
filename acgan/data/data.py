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


def load_rockps(batch_size, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)
        batch = tf.image.resize_bilinear(
            batch, size=(128, 128), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch
    dataset, info = tfds.load(
        'rock_paper_scissors',
        with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_cats_vs_dogs(batch_size, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)[tf.newaxis, ...]
        batch = tf.image.resize_bilinear(
            batch, size=(128, 128), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch[0, ...]
    dataset, info = tfds.load(
        'cats_vs_dogs',
        with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .map(map_fn)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_stanford_dogs(batch_size, split=tfds.Split.TRAIN):
    '''load cropped stanford dogs'''
    def filter_fn(batch):
        shape = tf.cast(tf.shape(batch['image']), tf.float32)
        min_val = tf.math.minimum(shape[0], shape[1])
        max_val = tf.math.maximum(shape[0], shape[1])
        return (max_val/min_val) < 1.5

    def map_fn(batch):
        def crop(image, bbox):
            return tf.image.crop_to_bounding_box(
                image=image,
                offset_height=tf.cast(shape[0]*bbox[0], tf.int32),
                offset_width=tf.cast(shape[1]*bbox[1], tf.int32),
                target_height=tf.cast(shape[0]*(bbox[2]-bbox[0]), tf.int32),
                target_width=tf.cast(shape[1]*(bbox[3]-bbox[1]), tf.int32))
        bbox = batch['objects']['bbox']
        shape = tf.cast(tf.shape(batch['image']), tf.float32)
        image = tf.cast(batch['image'], tf.float32)
        image = crop(image, bbox[0, ...])
        # reshape
        image = tf.image.resize_bilinear(
            image[tf.newaxis, ...],
            size=(128, 128),
            align_corners=False)
        # normalize
        image = (image-127.5)/127.5
        return image[0, ...]

    dataset, info = tfds.load(
        'my_stanford_dogs', split=tfds.Split.TRAIN, with_info=True)
    dataset = dataset.filter(filter_fn).shuffle(1000).repeat()\
                     .map(map_fn)\
                     .batch(batch_size)\
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
        lambda batch_size: load_imagenet(batch_size, 128),
        'rockps': load_rockps,
        'cats_vs_dogs': load_cats_vs_dogs,
        'stanford_dogs': load_stanford_dogs
    }
    return loaders[name](batch_size)
