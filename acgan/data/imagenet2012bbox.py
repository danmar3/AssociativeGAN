# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Imagenet datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import tarfile

import xml.etree.ElementTree as ET

import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = '''\
ILSVRC 2012, aka ImageNet is an image dataset organized according to the
WordNet hierarchy. Each meaningful concept in WordNet, possibly described by
multiple words or word phrases, is called a "synonym set" or "synset". There are
more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). In
ImageNet, we aim to provide on average 1000 images to illustrate each synset.
Images of each concept are quality-controlled and human-annotated. In its
completion, we hope ImageNet will offer tens of millions of cleanly sorted
images for most of the concepts in the WordNet hierarchy.
Note that labels were never publicly released for the test set, so we only
include splits for the training and validation sets here.
'''

# Web-site is asking to cite paper from 2015.
# http://www.image-net.org/challenges/LSVRC/2012/index#cite
_CITATION = '''\
@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}
'''

_LABELS_FNAME = 'image_classification/imagenet2012_labels.txt'

# This file contains the validation labels, in the alphabetic order of
# corresponding image names (and not in the order they have been added to the
# tar file).
_VALIDATION_LABELS_FNAME = 'image_classification/imagenet2012_validation_labels.txt'


# From https://github.com/cytsai/ilsvrc-cmyk-image-list
CMYK_IMAGES = [
    'n01739381_1309.JPEG',
    'n02077923_14822.JPEG',
    'n02447366_23489.JPEG',
    'n02492035_15739.JPEG',
    'n02747177_10752.JPEG',
    'n03018349_4028.JPEG',
    'n03062245_4620.JPEG',
    'n03347037_9675.JPEG',
    'n03467068_12171.JPEG',
    'n03529860_11437.JPEG',
    'n03544143_17228.JPEG',
    'n03633091_5218.JPEG',
    'n03710637_5125.JPEG',
    'n03961711_5286.JPEG',
    'n04033995_2932.JPEG',
    'n04258138_17003.JPEG',
    'n04264628_27969.JPEG',
    'n04336792_7448.JPEG',
    'n04371774_5854.JPEG',
    'n04596742_4225.JPEG',
    'n07583066_647.JPEG',
    'n13037406_4650.JPEG',
]

PNG_IMAGES = ['n02105855_2933.JPEG']


def extract_class_bbox(class_folder):
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


def extract_bbox(bbox_dir):
    bbox_data = {
        folder: extract_class_bbox(os.path.join(bbox_dir, folder))
        for folder in os.listdir(bbox_dir)}
    return bbox_data


class Imagenet2012bbox(tfds.core.GeneratorBasedBuilder):
    """Imagenet 2012, aka ILSVRC 2012."""
    VERSION = tfds.core.Version(
        '5.0.0', 'New split API (https://tensorflow.org/datasets/splits)')

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain two files: ILSVRC2012_img_train.tar and
    ILSVRC2012_img_val.tar.
    You need to register on http://www.image-net.org/download-images in order
    to get the link to download the dataset.
    """

    def _info(self):
        names_file = tfds.core.get_tfds_path(_LABELS_FNAME)
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(encoding_format='jpeg'),
                'label': tfds.features.ClassLabel(names_file=names_file),
                'file_name': tfds.features.Text(),  # Eg: 'n15075141_54.JPEG'
                'objects': tfds.features.Sequence({
                    "bbox": tfds.features.FeaturesDict({
                        'xmin': tf.int32,
                        'xmax': tf.int32,
                        'ymin': tf.int32,
                        'ymax': tf.int32,
                    }),
                    "label": tfds.features.Text(),
                    }),
            }),
            supervised_keys=('image', 'label'),
            homepage='http://image-net.org/',
            citation=_CITATION,
            )

    @staticmethod
    def _get_validation_labels(val_path):
        """Returns labels for validation.
        Args:
          val_path: path to TAR file containing validation images.
          It is used to retrieve the name of pictures and associate them
          to labels.
        Returns:
          dict, mapping from image name (str) to label (str).
        """
        labels_path = tfds.core.get_tfds_path(_VALIDATION_LABELS_FNAME)
        with tf.io.gfile.GFile(labels_path) as labels_f:
            # `splitlines` to remove trailing `\r` in Windows
            labels = labels_f.read().strip().splitlines()
        with tf.io.gfile.GFile(val_path, 'rb') as tar_f_obj:
            tar = tarfile.open(mode='r:', fileobj=tar_f_obj)
            images = sorted(tar.getnames())
        return dict(zip(images, labels))

    def _split_generators(self, dl_manager):
        train_path = os.path.join(
            dl_manager.manual_dir, 'ILSVRC2012_img_train.tar')
        train_bbox = os.path.join(
            dl_manager.manual_dir, 'ILSVRC2012_bbox_train_v2.tar.gz')
        val_path = os.path.join(
            dl_manager.manual_dir, 'ILSVRC2012_img_val.tar')
        val_bbox = os.path.join(
            dl_manager.manual_dir, 'ILSVRC2012_bbox_val_v3.tgz')
        # We don't import the original test split, as it doesn't include
        # labels. These were never publicly released.
        if (not tf.io.gfile.exists(train_path)
                or not tf.io.gfile.exists(val_path)):
            raise AssertionError(
                'ImageNet requires manual download of the data. '
                'Please download the train and val set and place them into: '
                '{}, {}'.format(
                    train_path, val_path))

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'archive': dl_manager.iter_archive(train_path),
                    'bbox_folder': dl_manager.extract(train_bbox),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'archive': dl_manager.iter_archive(val_path),
                    'bbox_folder': dl_manager.extract(val_bbox),
                    'validation_labels': self._get_validation_labels(val_path),
                },
            ),
        ]

    def _fix_image(self, image_fname, image):
        """Fix image color system and format starting from v 3.0.0."""
        if self.version < '3.0.0':
            return image
        if image_fname in CMYK_IMAGES:
            image = io.BytesIO(tfds.core.utils.jpeg_cmyk_to_rgb(image.read()))
        elif image_fname in PNG_IMAGES:
            image = io.BytesIO(tfds.core.utils.png_to_jpeg(image.read()))
        return image

    @staticmethod
    def _get_objects(image_fname, bbox):
        if bbox is None:
            return list()

        xml_name = image_fname.split('.')[0] + '.xml'
        class_name = image_fname.split('_')[0]
        if class_name in bbox and xml_name in bbox[class_name]:
            return [
                {'bbox': {
                    'xmin': obj['xlim'][0],
                    'xmax': obj['xlim'][1],
                    'ymin': obj['ylim'][0],
                    'ymax': obj['ylim'][1],
                    },
                 'label': obj['label']}
                for obj in bbox[class_name][xml_name]
            ]
        else:
            return list()

    @staticmethod
    def _get_objects_valid(image_fname, bbox):
        if bbox is None:
            return list()

        xml_name = image_fname.split('.')[0] + '.xml'
        if xml_name in bbox:
            return [
                {'bbox': {
                    'xmin': obj['xlim'][0],
                    'xmax': obj['xlim'][1],
                    'ymin': obj['ylim'][0],
                    'ymax': obj['ylim'][1],
                    },
                 'label': obj['label']}
                for obj in bbox[xml_name]
            ]
        else:
            return list()

    def _generate_examples(self, archive, bbox_folder=None,
                           validation_labels=None):
        """Yields examples."""
        if validation_labels:  # Validation split
            for key, example in self._generate_examples_validation(
                    archive, bbox_folder, validation_labels):
                yield key, example
        # bbox
        if bbox_folder is not None:
            bbox = extract_bbox(bbox_folder)
        else:
            bbox = None
        # Training split. Main archive contains archives names after a
        # synset noun.
        # Each sub-archive contains pictures associated to that synset.
        for fname, fobj in archive:
            label = fname[:-4]  # fname is something like 'n01632458.tar'
            # TODO(b/117643231): in py3, the following lines trigger tarfile
            # module to call `fobj.seekable()`, which Gfile doesn't have.
            # We should find an alternative, as this loads ~150MB in RAM.
            fobj_mem = io.BytesIO(fobj.read())
            for image_fname, image in tfds.download.iter_archive(
                    fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
                image = self._fix_image(image_fname, image)
                record = {
                    'file_name': image_fname,
                    'image': image,
                    'label': label,
                    'objects': self._get_objects(image_fname, bbox)
                }
                yield image_fname, record

    def _generate_examples_validation(self, archive, bbox_folder, labels):
        if bbox_folder is not None:
            bbox = extract_class_bbox(os.path.join(bbox_folder, 'val'))
        else:
            bbox = None
        for fname, fobj in archive:
            record = {
                'file_name': fname,
                'image': fobj,
                'label': labels[fname],
                'objects': self._get_objects_valid(fname, bbox)
            }
            yield fname, record
