import tensorflow as tf


def random_resize_crop(input_img, resize_size=None, crop_size=None):
    ''' Randomly chooses if resize and crop image at a random position.
    '''
    if resize_size is None:
        resize_size = (36, 36)
    if crop_size is None:
        crop_size = (32, 32)

    def random_resize_crop(input):
        image = tf.image.resize(
            input[tf.newaxis, ...],
            size=resize_size,
            method=tf.image.ResizeMethod.BILINEAR)
        return tf.image.random_crop(image[0, ...], size=list(crop_size) + [3])

    image = tf.cast(input_img, tf.float32)
    image = tf.cond(
        tf.random.uniform(shape=[]) > 0.5,
        true_fn=lambda: random_resize_crop(image),
        false_fn=lambda: image)
    return image


def random_pad_crop(input_img, pad_size=None, crop_size=None):
    ''' Randomly chooses if resize and crop image at a random position.
    '''
    if pad_size is None:
        pad_size = (4, 4)
    if crop_size is None:
        crop_size = (32, 32)

    def random_resize_crop(input):
        image = tf.pad(
            input, [[pad_size[0], pad_size[0]],
                    [pad_size[1], pad_size[1]],
                    [0, 0]],
            "SYMMETRIC")
        return tf.image.random_crop(image, size=list(crop_size) + [3])

    image = tf.cast(input_img, tf.float32)
    image = tf.cond(
        tf.random.uniform(shape=[]) > 0.5,
        true_fn=lambda: random_resize_crop(image),
        false_fn=lambda: image)
    return image


def random_flip(images):
    """randomply chooses a set of images in a batch to apply a left-right flip
    """
    images = tf.image.random_flip_left_right(images)
    return images


def batch_wrapper(image_process_fn):
    """ wrapps an image-processing function to handle labels"""
    def wrap_fn(batch):
        if isinstance(batch, dict):
            image = image_process_fn(batch['image'])
            return {'label': batch['label'], 'image': image}
        else:
            return image_process_fn(batch)
    return wrap_fn
