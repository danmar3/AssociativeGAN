import tensorflow as tf
import tensorflow_datasets as tfds


def load_mnist(batch_size, split=tfds.Split.TRAIN):
    dataset, info = tfds.load('mnist', with_info=True,
                              split=split)
    dataset = dataset.repeat()\
        .map(lambda x: (tf.cast(x['image'], tf.float32)-127.5)/127.5)\
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_fashion_mnist(batch_size, split=tfds.Split.TRAIN):
    dataset, info = tfds.load('fashion_mnist', with_info=True,
                              split=split)
    dataset = dataset.repeat()\
        .map(lambda x: (tf.cast(x['image'], tf.float32)-127.5)/127.5)\
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a(batch_size, split=tfds.Split.TRAIN):
    dataset, info = tfds.load('celeb_a', with_info=True,
                              split=split)
    dataset = dataset.repeat()\
        .map(lambda x: (tf.cast(x['image'], tf.float32)-127.5)/127.5)\
        .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
