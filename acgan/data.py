import tensorflow as tf
import tensorflow_datasets as tfds

DATA_DIR = None


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
        # batch = tf.image.resize_bilinear(
        #    batch, size=(218//3, 178//3), align_corners=False)
        # batch = tf.image.crop_to_bounding_box(
        #    batch, offset_height=0, offset_width=0,
        #    target_height=65, target_width=65)
        # batch = tf.image.crop_to_bounding_box(
        #    batch, offset_height=0, offset_width=0,
        #    target_height=216, target_width=176)
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
