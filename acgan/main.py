import os
import tqdm
import json
import datetime
import numpy as np
import tensorflow as tf
import twodlearn as tdl
import matplotlib.pyplot as plt
from . import data
from . import params as acgan_params
from .train import run_training
from .model.gmm_gan import GmmGan


def normalize_image(image):
    return ((image-image.min())/(image.max()-image.min()))


class ExperimentGMM(object):
    @classmethod
    def restore_session(cls, session_path, dataset_name):
        with open(os.path.join(session_path, 'params.json'), "r") as file_h:
            params = json.load(file_h)
        experiment = cls(dataset_name=dataset_name, params=params)
        experiment.restore(session_path)
        return experiment

    def _init_params(self, params, dataset_name):
        if params is None:
            params = acgan_params.PARAMS[dataset_name][self.name]
        self.params = params
        filename = os.path.join(self.output_dir, 'params.json')
        with open(filename, 'w') as file_h:
            json.dump(self.params, file_h)

    def __init__(self, dataset_name='celeb_a', params=None):
        self.name = 'gmmgan'
        self.session = tf.InteractiveSession()

        # init output_dir
        now = datetime.datetime.now()
        self.output_dir = 'tmp/{}/session_{}{:02d}{:02d}:{:02d}{:02d}'.format(
            self.name,
            now.year, now.month, now.day, now.hour, now.minute
            )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # init model
        self._init_params(params, dataset_name)
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
                xreal=xreal, **self.params['encoder_trainer']))
        tdl.core.variables_initializer(self.trainer.gen.variables).run()
        tdl.core.variables_initializer(self.trainer.dis.variables).run()
        tdl.core.variables_initializer(self.trainer.enc.variables).run()
        # saver
        self.saver = tf.train.Saver(tdl.core.get_variables(self.model))

    def restore(self, pathname=None):
        '''Restore the weight values stored at pathname.

        Args:
            pathname: path where the variables checkpoints are stored.
                It could point to a session folder, a checkpoints folder or
                a specific checkpoint file. By default chooses the most recent
                checkpoint.
        '''
        def get_latest(pathname, filter=None):
            if filter is None:
                filter = lambda x: True
            folders = [folder for folder in os.listdir(pathname)
                       if os.path.isdir(os.path.join(pathname, folder))
                       and filter(folder)]
            if folders:
                return os.path.join(pathname, sorted(folders)[-1])
            files = [fi.split('.')[0] for fi in os.listdir(pathname)
                     if filter(fi)]
            if files:
                return os.path.join(pathname, sorted(files)[-1] + '.ckpt')
            else:
                raise ValueError('could not find any saved checkpoint in {}'
                                 ''.format(pathname))

        if pathname is None:
            pathname = self.output_dir
        if os.path.isdir(pathname):
            if any('session' in folder for folder in os.listdir(pathname)):
                pathname = get_latest(pathname, lambda x: 'session' in x)
            if 'checkpoints' in os.listdir(pathname):
                pathname = os.path.join(pathname, 'checkpoints')
            pathname = get_latest(pathname, lambda x: 'vars_' in x)
        print('-------------- Restoring: {} ------------------'
              ''.format(pathname))
        self.saver.restore(self.session, pathname)

    def run(self, n_steps=100):
        for trial in tqdm.tqdm(range(n_steps)):
            if not run_training(
                    dis=self.trainer.dis, gen=self.trainer.gen,
                    n_steps=200, n_logging=10):
                return False
            for i in tqdm.tqdm(range(200)):
                self.session.run(self.trainer.enc.step['encoder'])
            for i in tqdm.tqdm(range(20)):
                self.session.run(self.trainer.enc.step['embedding'])
        return True

    def visualize_clusters(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(4, 4, figsize=(20, 20))


    def visualize(self, filename=None):
        if filename is None:
            folder = os.path.join(self.output_dir, 'images')
            if not os.path.exists(folder):
                os.makedirs(folder)
            now = datetime.datetime.now()
            filename = '{}/generated_{}{:02d}{:02d}:{:02d}{:02d}.pdf'.format(
                folder, now.year, now.month, now.day, now.hour, now.minute)

        fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        ax = np.reshape(ax, 4*4)
        # dis.sim_pyramid[-1].eval()
        xsim = self.session.run(self.trainer.gen.xsim)
        for i in range(4*4):
            image = (xsim[i][:, :, :]+1)*0.5
            ax[i].imshow(np.squeeze(normalize_image(image)),
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
