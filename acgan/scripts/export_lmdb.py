import tensorflow as tf
tf.enable_eager_execution()
import os
import cv2
import shutil
import numpy as np
import twodlearn
import twodlearn.debug
import acgan
#from acgan.data import lsun_dogs
import tensorflow_datasets as tfds
import lmdb
from os.path import exists, join
import matplotlib.pyplot as plt


def lmdb_reader(filename):
    lmdb_env = lmdb.open(filename)
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:
                yield key, value


def export_lmdb(input_fn, output_fn, max_images=10000):
    if os.path.exists(output_fn):
        shutil.rmtree(output_fn)
    os.makedirs(output_fn)
    env_out = lmdb.open(output_fn, map_size=15000000000)

    with env_out.begin(write=True, buffers=True) as txn_out:
        for idx, (key, value) in enumerate(lmdb_reader(input_fn)):
            img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
            # img_bin = tfds.as_numpy(tf.image.encode_jpeg(img))
            img_bin = cv2.imencode('.jpg', img)[1].tobytes()
            txn_out.put(str(idx).encode('ascii'), img_bin)
            if idx % 100 == 0:
                print('processed {} images'.format(idx))
            if idx > max_images:
                break


def lmdb_image_reader(filename):
    for key, value in lmdb_reader(filename):
        yield tf.image.decode_image(value)


def main():
    input_fn = '/data/marinodl/datasets/lsun_cow/cow/'
    output_fn = 'tmp/cow'
    export_lmdb(input_fn=input_fn, output_fn=output_fn)
