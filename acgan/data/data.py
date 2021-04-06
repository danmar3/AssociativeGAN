import tensorflow as tf
import tensorflow_datasets as tfds
from . import imagenet2012
from . import augment

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
        batch = tf.compat.v1.image.resize_bilinear(
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


def load_mnist32_3c(
        batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        batch_img = tf.compat.v1.image.resize_bilinear(
            batch_img, size=(32, 32), align_corners=False)
        batch_img = (batch_img-127.5)/127.5
        batch_img = tf.tile(batch_img, [1, 1, 1, 3])
        if with_label:
            batch = {'label': batch['label'], 'image': batch_img}
        else:
            batch = batch_img
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


def load_fashion_mnist_3c(
        batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        batch_img = tf.compat.v1.image.resize_bilinear(
            batch_img, size=(32, 32), align_corners=False)
        batch_img = (batch_img-127.5)/127.5
        batch_img = tf.tile(batch_img, [1, 1, 1, 3])
        if with_label:
            batch = {'label': batch['label'], 'image': batch_img}
        else:
            batch = batch_img
        return batch

    dataset, info = tfds.load(
        'fashion_mnist', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a(
        batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        batch_img = tf.compat.v1.image.central_crop(
            batch_img, central_fraction=0.7)
        batch_img = tf.compat.v1.image.resize_bilinear(
            batch_img, size=(64, 64), align_corners=False)
        batch_img = (batch_img-127.5)/127.5
        if with_label:
            batch = {'label': batch['landmarks'], 'image': batch_img}
        else:
            batch = batch_img
        return batch
    dataset, info = tfds.load(
        'celeb_a', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a_128_cropped(
        batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        batch_img = tf.compat.v1.image.central_crop(
            batch_img, central_fraction=0.7)
        batch_img = tf.compat.v1.image.resize_bilinear(
            batch_img, size=(128, 128), align_corners=False)
        batch_img = (batch_img-127.5)/127.5
        if with_label:
            batch = {'label': batch['landmarks'], 'image': batch_img}
        else:
            batch = batch_img
        return batch
    dataset, info = tfds.load(
        'celeb_a', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_celeb_a_128(
        batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        # batch = tf.image.central_crop(batch, central_fraction=0.7)
        batch_img = tf.compat.v1.image.resize_bilinear(
            batch_img, size=(128, 128), align_corners=False)
        batch_img = (batch_img-127.5)/127.5
        if with_label:
            batch = {'label': batch['landmarks'], 'image': batch_img}
        else:
            batch = batch_img
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
        batch = tf.compat.v1.image.resize_bilinear(
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


def load_rockps(batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        batch_img = tf.compat.v1.image.resize_bilinear(
            batch_img, size=(128, 128), align_corners=False)
        batch_img = (batch_img-127.5)/127.5
        if with_label:
            batch = {'label': batch['landmarks'], 'image': batch_img}
        else:
            batch = batch_img
        return batch
    dataset, info = tfds.load(
        'rock_paper_scissors',
        with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_cats_vs_dogs(batch_size, split=tfds.Split.TRAIN, with_label=False):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)[tf.newaxis, ...]
        batch = tf.compat.v1.image.resize_bilinear(
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


def load_stanford_dogs(
        batch_size, size=128, split=tfds.Split.TRAIN, with_label=False):
    '''load cropped stanford dogs'''
    def filter_fn(batch):
        shape = tf.cast(tf.shape(batch['image']), tf.float32)
        min_val = tf.math.minimum(shape[0], shape[1])
        max_val = tf.math.maximum(shape[0], shape[1])
        return (max_val/min_val) < 1.5

    def map_crop(batch):
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
        if with_label:
            batch = {'label': batch['label'], 'image': image}
        else:
            batch = image
        return image

    def map_resize(batch):
        # reshape
        image = tf.compat.v1.image.resize_bilinear(
            batch['image'][tf.newaxis, ...],
            size=(size, size),
            align_corners=False)
        # normalize
        image = (image-127.5)/127.5
        image = image[0, ...]
        if with_label:
            batch = {'label': batch['label'], 'image': image}
        else:
            batch = image
        return

    dataset, info = tfds.load(
        'my_stanford_dogs', split=tfds.Split.TRAIN, with_info=True)
    dataset = dataset.shuffle(1000).repeat()\
                     .map(map_crop).filter(filter_fn).map(map_resize)\
                     .batch(batch_size)\
                     .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_cifar10(
        batch_size, split=tfds.Split.TRAIN, with_label=False,
        drop_remainder=False, augment_data=False, crop_method='resize',
        crop_kargs=None):
    if crop_kargs is None:
        if crop_method == 'resize':
            crop_kargs = {'resize_size': (40, 40), 'crop_size': (32, 32)}
        if crop_method == 'pad':
            crop_kargs = {'pad_size': (4, 4), 'crop_size': (32, 32)}

    def map_fn(batch):
        batch_img = tf.cast(batch['image'], tf.float32)
        # batch_img = (batch_img-127.5)/127.5
        import numpy as np
        print("using color normalize")
        batch_img = (batch_img - np.array([125.3, 123.0, 113.9]))/(
            np.array([63.0,  62.1,  66.7]))
        if with_label:
            batch = {'label': tf.one_hot(batch['label'], depth=10),
                     'image': batch_img}
        else:
            batch = batch_img
        return batch

    def random_crop(image):
        if crop_method == 'resize':
            return augment.random_resize_crop(image, **crop_kargs)
        elif crop_method == 'pad':
            return augment.random_pad_crop(image, **crop_kargs)
        else:
            raise ValueError(f'crop method {crop_method} not recognized')

    dataset, info = tfds.load(
        'cifar10', split=split, with_info=True)
    print('loading split {}'.format(split))
    if augment_data:
        print('using augmented dataset')
        dataset = dataset.shuffle(1000).repeat()\
            .map(augment.batch_wrapper(random_crop))\
            .batch(batch_size, drop_remainder=drop_remainder)\
            .map(map_fn).map(augment.batch_wrapper(augment.random_flip))\
            .prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.shuffle(1000).repeat()\
                         .batch(batch_size, drop_remainder=drop_remainder)\
                         .map(map_fn)\
                         .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_lsun(batch_size, category, size=128, split=tfds.Split.TRAIN):
    def map_fn(batch):
        batch = tf.cast(batch['image'], tf.float32)[tf.newaxis, ...]
        batch = tf.compat.v1.image.resize_bilinear(
            batch, size=(size, size), align_corners=False)
        batch = (batch-127.5)/127.5
        return batch[0, ...]

    dataset, info = tfds.load(
        'lsun_objects/{}'.format(category),
        with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .map(map_fn)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def load_stl10(batch_size, size=128, split='unlabelled', with_label=False):
    def map_fn(batch):
        image = tf.cast(batch['image'], tf.float32)
        image = tf.compat.v1.image.resize_bilinear(
            image, size=(size, size), align_corners=False)
        image = (image-127.5)/127.5
        if with_label:
            batch = {'label': batch['label'], 'image': image}
        else:
            batch = image
        return batch
    dataset, info = tfds.load(
        'stl10', with_info=True, split=split, data_dir=DATA_DIR)
    dataset = dataset.shuffle(1000).repeat()\
        .batch(batch_size)\
        .map(map_fn)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def to_one_hot(dataset, n_classes):
    return dataset.map(
        lambda batch: {'image': batch['image'],
                       'label': tf.one_hot(batch['label'], depth=n_classes)})


def load(name, batch_size, with_label=False, **kargs):
    def load_imagenet(batch_size, resolution):
        return imagenet2012.load_imagenet2012(
            data_dir=DATA_DIR,
            batch_size=batch_size,
            crop=True, resolution=resolution)
    loaders = {
        'celeb_a': lambda batch_size, **kargs: load_celeb_a_128_cropped(
            batch_size, with_label=with_label, **kargs),
        'imagenet_512': lambda batch_size, **kargs: load_imagenet(
            batch_size, 512, **kargs),
        'imagenet_256': lambda batch_size, **kargs: load_imagenet(
            batch_size, 256, **kargs),
        'imagenet_128': lambda batch_size, **kargs: load_imagenet(
            batch_size, 128, **kargs),
        'rockps': lambda batch_size, **kargs: load_rockps(
            batch_size, with_label=with_label, **kargs),
        'cats_vs_dogs': lambda batch_size, **kargs: load_cats_vs_dogs(
            batch_size, with_label=with_label, **kargs),
        'stanford_dogs': lambda batch_size, **kargs: load_stanford_dogs(
            batch_size, size=128, with_label=with_label, **kargs),
        'stanford_dogs64': lambda batch_size, **kargs: load_stanford_dogs(
            batch_size, size=64, with_label=with_label, **kargs),
        'cifar10': lambda batch_size, **kargs: load_cifar10(
            batch_size, with_label=with_label, **kargs),
        'lsun_dog': lambda batch_size, **kargs: load_lsun(
            batch_size, 'dog', size=128, **kargs),
        'lsun_cat': lambda batch_size, **kargs: load_lsun(
            batch_size, 'cat', size=128, **kargs),
        'lsun_cow': lambda batch_size, **kargs: load_lsun(
            batch_size, 'cow', size=128, **kargs),
        'lsun_sheep': lambda batch_size, **kargs: load_lsun(
            batch_size, 'sheep', size=128, **kargs),
        'lsun_dog64': lambda batch_size, **kargs: load_lsun(
            batch_size, 'dog', size=64, **kargs),
        'lsun_cat64': lambda batch_size, **kargs: load_lsun(
            batch_size, 'cat', size=64, **kargs),
        'lsun_cow64': lambda batch_size, **kargs: load_lsun(
            batch_size, 'cow', size=64, **kargs),
        'lsun_sheep64': lambda batch_size, **kargs: load_lsun(
            batch_size, 'sheep', size=64, **kargs),
        'mnist': lambda batch_size, **kargs: load_mnist32_3c(
            batch_size, with_label=with_label, **kargs),
        'stl10': lambda batch_size, **kargs: load_stl10(
            batch_size, size=128, with_label=with_label, **kargs),
        'stl10_64': lambda batch_size, **kargs: load_stl10(
            batch_size, size=64, with_label=with_label, **kargs),

    }
    return loaders[name](batch_size, **kargs)
