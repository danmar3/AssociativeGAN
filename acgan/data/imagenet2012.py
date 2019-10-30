import os
import tarfile
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

import tensorflow as tf


def extract_bbox(class_folder):
    def extract_xml(file):
        path = os.path.join(class_folder, file)
        tree = ET.parse(path)
        data = [{'label': next(obj.iter('name')).text,
                 'xlim': (int(next(obj.iter('xmin')).text),
                          int(next(obj.iter('xmax')).text)),
                 'ylim': (int(next(obj.iter('ymin')).text),
                          int(next(obj.iter('ymax')).text)),
                 }
                for obj in tree.getroot().iter('object')]
        return data
    return {file: extract_xml(file)
            for file in os.listdir(class_folder)}


def extract_bbox_train(data_dir):
    bbox_dir = os.path.join(data_dir, 'ilsvrc2012',
                            'extracted', 'ILSVRC2012_bbox_train')
    bbox_data = {
        folder: extract_bbox(os.path.join(bbox_dir, folder))
        for folder in os.listdir(bbox_dir)}
    return bbox_data


def tar_generator(fileobj, bbox_data=None):
    data = tarfile.open(fileobj=fileobj)
    for member in data.getmembers():
        imgobj = data.extractfile(member)
        img_data = {
            'file_id': member.name,
            'label': member.name.split('_')[0],
            'image': np.array(Image.open(imgobj))}
        if bbox_data is not None:
            img_data['bbox'] = bbox_data[member.name.split('.')[0] + '.xml']
        yield img_data
    data.close()


def imagenet2012_train(data_dir, bbox_data=None):
    train_data = os.path.join(
        data_dir, 'ilsvrc2012/downloads/ILSVRC2012_img_train.tar')
    with tarfile.open(train_data) as train:
        classes_gen = [
            tar_generator(
                train.extractfile(member),
                None if bbox_data is None
                else bbox_data[member.name.split('/')[-1].split('.')[0]])
            for member in train.getmembers()
            if member.isfile()]
        done = False
        while not done:
            done = True
            for gen in classes_gen:
                try:
                    yield next(gen)
                    done = False
                except StopIteration:
                    continue


def load_imagenet2012(data_dir, batch_size, crop=False, resolution=512):
    def img_generator():
        bbox_data = extract_bbox_train(data_dir=data_dir)
        for item in imagenet2012_train(data_dir, bbox_data=bbox_data):
            if len(item['image'].shape) != 3:
                continue
            if item['image'].shape[-1] != 3:
                continue
            for bbox in item['bbox']:
                yield {'image': item['image'],
                       'label': bbox['label'],
                       'xlim': bbox['xlim'],
                       'ylim': bbox['ylim']}

    def map_fn(batch):
        image = tf.cast(batch['image'], tf.float32)[tf.newaxis, ...]
        if crop:
            image = tf.image.crop_to_bounding_box(
                image=image,
                offset_height=batch['ylim'][0],
                offset_width=batch['xlim'][0],
                target_height=batch['ylim'][1]-batch['ylim'][0],
                target_width=batch['xlim'][1]-batch['xlim'][0])
        image = tf.image.resize_bilinear(
            image, size=(resolution, resolution),
            align_corners=False)
        image = (image-127.5)/127.5
        return image[0, ...]

    dataset = tf.data.Dataset.from_generator(
                    img_generator,
                    {'image': tf.float32, 'label': tf.string,
                     'xlim': tf.int32, 'ylim': tf.int32},
                    {'image': tf.TensorShape([None, None, 3]),
                     'label': tf.TensorShape(()),
                     'xlim': tf.TensorShape((2,)),
                     'ylim': tf.TensorShape((2,))})\
                .shuffle(100).repeat()\
                .map(map_fn, num_parallel_calls=10)\
                .batch(batch_size)\
                .prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
