import argparse
import tensorflow_gan as tfgan
import acgan
import tensorflow as tf
from tensorflow_gan.examples.cifar import networks
from acgan.benchmark.dcgan import DCGAN

import matplotlib.pyplot as plt
import numpy as np

hparams = {
    'embedding_size': 64,
    'batch_size': 32,
    'generator_lr': 0.0002,
    'discriminator_lr': 0.0002,
    'master': '',
    'train_log_dir': '/tmp/tfgan_logdir/cifar/',
    'ps_replicas': 0,
    'task': 0
}

FLAGS = None


def load_dataset(name):
    if name == "cifar10":
        return acgan.data.data.load_cifar10(
            hparams['batch_size'], drop_remainder=True)


def main():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='cifar10', help='dataset to use')

    FLAGS = parser.parse_args()

    print('\n\n--------> loading dataset ...')
    dataset = load_dataset(FLAGS.dataset)

    print('\n\n--------> building the model ...')
    x_real = dataset.make_one_shot_iterator().get_next()
    experiment = DCGAN(images=x_real, params=hparams)

    print('\n\n--------> running training ...')
    experiment.run_training()


if __name__ == "__main__":
    main()
