# coding=utf-8
# data at: http://dl.yf.io/lsun/objects/
# Copyright 2019 The TensorFlow Datasets Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""TODO(lsun_dogs): Add a description here."""

"""LSUN dataset.
Large scene understanding dataset.
"""
import io
import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

# LSUN_URL = "http://dl.yf.io/lsun/scenes/%s_%s_lmdb.zip"
LSUN_URL = "http://dl.yf.io/lsun/objects/%s.zip"

_CITATION = """\
@article{journals/corr/YuZSSX15,
    added-at = {2018-08-13T00:00:00.000+0200},
    author = {Yu, Fisher and Zhang, Yinda and Song, Shuran and Seff, Ari and Xiao, Jianxiong},
    biburl = {https://www.bibsonomy.org/bibtex/2446d4ffb99a5d7d2ab6e5417a12e195f/dblp},
    ee = {http://arxiv.org/abs/1506.03365},
    interhash = {3e9306c4ce2ead125f3b2ab0e25adc85},
    intrahash = {446d4ffb99a5d7d2ab6e5417a12e195f},
    journal = {CoRR},
    keywords = {dblp},
    timestamp = {2018-08-14T15:08:59.000+0200},
    title = {LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop.},
    url = {http://dblp.uni-trier.de/db/journals/corr/corr1506.html#YuZSSX15},
    volume = {abs/1506.03365},
    year = 2015
}
"""


# From http://dl.yf.io/lsun/categories.txt minus "test"
_CATEGORIES = [
    'cow',
    # 'airplane'
    # 'bicycle',
    # 'bird',
    # 'boat',
    # 'bottle',
    # 'bus',
    # 'car',
    'cat',
    # 'chair',
    # 'dining_table',
    'dog',
    # 'horse',
    # 'motorbike',
    # 'person',
    # 'potted_plant',
    'sheep',
    # 'sofa',
    # 'train',
    # 'tv-monitor',
]


class LsunObjects(tfds.core.GeneratorBasedBuilder):
    """Lsun dataset."""
    # Version history:
    # 3.0.0: S3 with new hashing function (different shuffle).
    # 2.0.0: S3 (new shuffling, sharding and slicing mechanism).
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(  # pylint: disable=g-complex-comprehension
            name=category,
            description="Images of category %s" % category,
            version=tfds.core.Version(
                "0.1.1", {tfds.core.Experiment.S3: False}),
            supported_versions=[
                tfds.core.Version("3.0.0"),
                tfds.core.Version("2.0.0"),
            ],
            ) for category in _CATEGORIES
        ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Large scale images showing different objects "
                         "from given categories like bedroom, tower etc."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(encoding_format="jpeg"),
                #"image": tfds.features.Tensor(shape=(None, None, 3), dtype=tf.uint8)
            }),
            urls=["https://www.yf.io/p/lsun"],
            citation=_CITATION,
            )

    def _split_generators(self, dl_manager):
        extracted_dirs = os.path.join(
            dl_manager.manual_dir, self.builder_config.name)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=40,
                gen_kwargs={
                    "extracted_dir": os.path.join(extracted_dirs, "train")
                    }),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=40,
                gen_kwargs={
                    "extracted_dir": os.path.join(extracted_dirs, "valid")
                    }),
                ]

    def _generate_examples(self, extracted_dir):
        with tf.Graph().as_default():
            dataset = tf.contrib.data.LMDBDataset(
                os.path.join(extracted_dir, "data.mdb"))
            for idx, (_, jpeg_image) in enumerate(tfds.as_numpy(dataset)):
                record = {"image": io.BytesIO(jpeg_image)}
                if self.version.implements(tfds.core.Experiment.S3):
                    yield idx, record
                else:
                    yield record
