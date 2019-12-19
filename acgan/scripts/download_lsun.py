import io
import os
import cv2
import tqdm
import lmdb
import shutil
import argparse
import collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
import acgan

tfds.download.add_checksums_dir(
    os.path.join(os.path.dirname(acgan.data.__file__),
                 'url_checksums'))

# LSUN_URL = "http://dl.yf.io/lsun/scenes/%s_%s_lmdb.zip"
LSUN_URL = "http://dl.yf.io/lsun/objects/%s.zip"
CATEGORIES = [
    'airplane'
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'dining_table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted_plant',
    'sheep',
    'sofa',
    'train',
    'tv-monitor',
    ]


def isgray(img):
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def lmdb_reader(filename):
    lmdb_env = lmdb.open(filename)
    with lmdb_env.begin() as lmdb_txn:
        with lmdb_txn.cursor() as lmdb_cursor:
            for key, value in lmdb_cursor:
                yield key, value


def prepare_lmdb(input_fn, output_fn, max_images=10000):
    if os.path.exists(output_fn):
        shutil.rmtree(output_fn)
    os.makedirs(output_fn)
    total_images = 0
    splits = collections.deque(['train', 'valid'])

    env_out, txn_out = None, None
    pbar = tqdm.tqdm(total=max_images*len(splits))
    for idx, (key, value) in enumerate(lmdb_reader(input_fn)):
        if total_images % max_images == 0:
            if env_out is not None:
                txn_out.commit()
                env_out.close()
            if len(splits) == 0:
                break
            env_out = lmdb.open(
                os.path.join(output_fn, splits.popleft()),
                map_size=15000000000)
            txn_out = env_out.begin(write=True, buffers=True)

        if value:
            img_mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), 1)
        else:
            continue
        if (isgray(img_mat)
                or img_mat.shape[0]/img_mat.shape[1] > 1.1
                or img_mat.shape[0]/img_mat.shape[1] < 0.5):
            continue
        # img_bin = tfds.as_numpy(tf.image.encode_jpeg(img))
        img_bin = cv2.imencode('.jpg', img_mat)[1].tobytes()
        txn_out.put(str(idx).encode('ascii'), img_bin)
        total_images += 1
        if idx % 100 == 0:
            pbar.update(100)
    pbar.close()


def download_and_prepare(data_dir='tmp/datasets',
                         output_dir='tmp/generated/',
                         category='cow',
                         samples_per_split=50000):
    if category not in CATEGORIES:
        raise ValueError('invalid category.')
    else:
        print('\n\nDownloading and preparing: {} \n\n'.format(category))
    download_dir = os.path.join(data_dir, 'download')
    extract_dir = os.path.join(download_dir, 'extract')
    download_mgr = tfds.download.DownloadManager(
        download_dir=download_dir,
        extract_dir=extract_dir)
    extracted_dir = download_mgr.download_and_extract(
        {'all': LSUN_URL % (category)})

    output_dir = os.path.join(output_dir, category)
    base_dir = os.path.join(extracted_dir['all'], category)
    # generated_dir = os.path.join(download_dir, '')
    prepare_lmdb(base_dir, output_dir, max_images=samples_per_split)
    shutil.rmtree(download_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='tmp/datasets',
                        help='datasets path')
    parser.add_argument('--output_dir', default='tmp/generated',
                        help='output path')
    parser.add_argument('--category', default=None,
                        help='object category')
    parser.add_argument('--force', default=None,
                        help='force redownload')

    FLAGS = parser.parse_args()
    if FLAGS.category is not None:
        download_and_prepare(
            data_dir=FLAGS.data_dir, output_dir=FLAGS.output_dir,
            category=FLAGS.category)
    else:
        for category in ['cow', 'sheep', 'dog', 'cat']:
            if (os.path.exists(os.path.join(FLAGS.output_dir, category))
                    and not FLAGS.force):
                continue
            download_and_prepare(
                data_dir=FLAGS.data_dir, output_dir=FLAGS.output_dir,
                category=category)

    print("move generated samples to '{datadir}/downloads/manual/lsun_objects'")


if __name__ == "__main__":
    main()
