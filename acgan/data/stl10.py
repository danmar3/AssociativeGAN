"""STL10 dataset.
Dataset extracted from imagenet.
"""

# sha256sum filepath
# ls -l filepath

import io
import os
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

_CITATION = """\
@inproceedings{coates2011analysis,
title={An analysis of single-layer networks in unsupervised feature learning},
author={Coates, Adam and Ng, Andrew and Lee, Honglak},
booktitle={Proceedings of the fourteenth international conference on
           artificial intelligence and statistics},
pages={215--223},
year={2011}
}
"""

_CATEGORIES = [
    'cow',
]

tfds.download.add_checksums_dir(
    str(pathlib.Path(__file__).parent.absolute().joinpath('url_checksums')))


# taken from https://github.com/mttk/STL10
def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


class Stl10(tfds.core.GeneratorBasedBuilder):
    """stl10 dataset."""
    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(  # pylint: disable=g-complex-comprehension
            name="unsupervised",
            description="Images of unsupervised set",
            version=tfds.core.Version(
                "0.1.1", {tfds.core.Experiment.S3: False}),
            supported_versions=[tfds.core.Version("3.0.0")],
            )
        ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The STL-10 dataset is an image recognition dataset "
                         "for developing unsupervised feature learning, deep "
                         "learning, self-taught learning algorithms. It is "
                         "inspired by the CIFAR-10 dataset but with some "
                         "modifications. In particular, each class has fewer "
                         "labeled training examples than in CIFAR-10, but a "
                         "very large set of unlabeled examples is provided to "
                         "learn image models prior to supervised training. "
                         "The primary challenge is to make use of the "
                         "unlabeled data (which comes from a similar but "
                         "different distribution from the labeled data) to "
                         "build a useful prior. We also expect that the "
                         "higher resolution of this dataset (96x96) will make "
                         "it a challenging benchmark for developing more "
                         "scalable unsupervised learning methods."),
            features=tfds.features.FeaturesDict({
                # "image": tfds.features.Image(encoding_format="jpeg"),
                "image": tfds.features.Tensor(shape=(96, 96, 3), dtype=tf.uint8)
            }),
            urls=["http://ai.stanford.edu/~acoates/stl10/"],
            citation=_CITATION,
            )

    def _split_generators(self, dl_manager):
        images_path = dl_manager.download_and_extract(_DATA_URL)
        data_dir = os.path.join(images_path, 'stl10_binary')
        if self.builder_config.name == 'unsupervised':
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    num_shards=40,
                    gen_kwargs={"extracted_dir":
                                os.path.join(data_dir, "unlabeled_X.bin")
                                })
                ]
        else:
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    num_shards=40,
                    gen_kwargs={"extracted_dir":
                                os.path.join(data_dir, "train_X.bin")
                                }),
                tfds.core.SplitGenerator(
                    name=tfds.Split.VALIDATION,
                    num_shards=40,
                    gen_kwargs={"extracted_dir":
                                os.path.join(data_dir, "test_X.bin")
                                }),
                    ]

    def _generate_examples(self, extracted_dir):
        data_X = read_all_images(extracted_dir)
        for idx in range(data_X.shape[0]):
            record = {"image": data_X[idx, ...]}
            if self.version.implements(tfds.core.Experiment.S3):
                yield idx, record
            else:
                yield record
