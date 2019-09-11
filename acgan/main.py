import os
import tqdm
import json
import datetime
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import matplotlib.pyplot as plt
from . import data
from . import params
from .train import run_training
from .model.gmm_gan import GmmGan


class ExperimentGMM(object):
    def _load_params(self, dataset_name):
        self.params = params.PARAMS[dataset_name][self.name]
        filename = os.path.join(self.output_dir, 'params.json')
        with open(filename, 'w') as file_h:
            json.dump(self.params, file_h)

    def __init__(self, dataset_name='celeb_a'):
        self.name = 'gmmgan'
        self.session = tf.InteractiveSession()

        # init output_dir
        now = datetime.datetime.now()
        self.output_dir = 'tmp/{}/session_{}{:02d}{:02d}:{:02d}{:02d}'.format(
            type(self.model).__name__,
            now.year, now.month, now.day, now.hour, now.minute
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # init model
        self._load_params(dataset_name)
        dataset = data.load_celeb_a_128_cropped(
            self.params['generator_trainer']['batch_size'])
        self.model = GmmGan(**self.params['model'])
        iter = dataset.make_one_shot_iterator()
        xreal = iter.get_next()
        self.trainer = tdl.core.SimpleNamespace(
            gen=self.model.generator_trainer(
                **self.params['generator_trainer']),
            dis=self.model.discriminator_trainer(
                xreal=xreal, **self.params['discriminator_trainer']),
            enc=self.model.encoder_trainer(
                **self.params['encoder_trainer']))
        tdl.core.variables_initializer(self.trainer.gen.variables).run()
        tdl.core.variables_initializer(self.trainer.dis.variables).run()
        tdl.core.variables_initializer(self.trainer.enc.variables).run()
        # saver
        self.saver = tf.train.Saver(tdl.core.get_variables(self.model))

    def restore(self, filename):
        self.saver.restore(self.session, filename)

    def run(self, n_steps=100):
        for trial in tqdm.tqdm(range(n_steps)):
            run_training(
                dis=self.trainer.dis, gen=self.trainer.gen,
                n_steps=200, n_logging=10)
            for i in tqdm.tqdm(range(200)):
                self.session.run(self.trainer.enc.step['encoder'])
            for i in tqdm.tqdm(range(20)):
                self.session.run(self.trainer.enc.step['embedding'])

    def visualize(self, filename=None):
        if filename is None:
            folder = os.path.join(self.output_dir, 'images')
            if not os.path.exists(folder):
                os.makedirs(folder)
            now = datetime.datetime.now()
            filename = '{}/generated_{}{:02d}{:02d}:{:02d}{:02d}.pdf'.format(
                folder, now.year, now.month, now.day, now.hour, now.minute)

        def normalize(image):
            return ((image-image.min())/(image.max()-image.min()))
        fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        ax = np.reshape(ax, 4*4)
        # dis.sim_pyramid[-1].eval()
        xsim = self.session.run(self.trainer.gen.xsim)
        for i in range(4*4):
            image = (xsim[i][:, :, :]+1)*0.5
            ax[i].imshow(np.squeeze(normalize(image)),
                         interpolation='nearest')
            ax[i].axis('off')
        plt.savefig(filename)

    def save(self, folder=None):
        '''save model params'''
        if folder is None:
            folder = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(folder):
            os.makedirs(folder)
        now = datetime.datetime.now()
        filename = '{}/vars_{}{:02d}{:02d}:{:02d}{:02d}.ckpt'.format(
            folder,
            now.year, now.month, now.day, now.hour, now.minute)
        self.saver.save(self.session, filename)
