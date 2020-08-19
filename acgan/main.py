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
from .utils import eager_function
from .train import run_training
from .model import msg_gan
from .model.gmm_gan import GmmGan
from .model.wacgan import WacGan, WacGanV2, WacGanDev
from .model.bigmmgan import BiGmmGan
import functools
from sklearn.manifold import TSNE


def normalize_image(image):
    return ((image-image.min())/(image.max()-image.min()))


def compat_params(params):
    if 'data' not in params:
        params['data'] = dict()
    return params


class ExperimentGMM(object):
    name = 'gmmgan'
    Model = GmmGan

    @classmethod
    def restore_session(cls, session_path, dataset_name, indicator=None):
        with open(os.path.join(session_path, 'params.json'), "r") as file_h:
            params = json.load(file_h)
        # remove n_start steps if loading a session
        if 'n_start' in params['run']:
            print('\n\n ---> resetting n_start to zero \n\n')
            params['run']['n_start'] = 0
        experiment = cls(dataset_name=dataset_name, params=params,
                         indicator=indicator)
        experiment.restore(session_path)
        return experiment

    def _init_params(self, params, dataset_name):
        if params is None:
            params = acgan_params.PARAMS[dataset_name][self.name]
        self.params = params
        self.params['indicator'] = self.indicator
        if 'global' in self.params:
            msg_gan.set_global(self.params['global'])
        filename = os.path.join(self.output_dir, 'params.json')
        with open(filename, 'w') as file_h:
            json.dump(self.params, file_h)

    def _init_trainer(self, xreal):
        trainer = tdl.core.SimpleNamespace(
            gen=self.model.generator_trainer(
                **self.params['generator_trainer']),
            dis=self.model.discriminator_trainer(
                xreal=xreal, **self.params['discriminator_trainer']),
            enc=self.model.encoder_trainer(
                xreal=xreal, **self.params['encoder_trainer']))
        tdl.core.variables_initializer(trainer.gen.variables).run()
        tdl.core.variables_initializer(trainer.dis.variables).run()
        tdl.core.variables_initializer(trainer.enc.variables).run()
        return trainer

    def __init__(self, dataset_name='celeb_a', params=None, indicator=None):
        self.indicator = indicator
        self.session = (tf.compat.v1.get_default_session()
                        if tf.compat.v1.get_default_session() is not None
                        else tf.compat.v1.InteractiveSession())

        # init output_dir
        now = datetime.datetime.now()
        self.output_dir = 'tmp/{}/session_{}{:02d}{:02d}_{:02d}{:02d}_{}'\
            ''.format(self.name,
                      now.year, now.month, now.day, now.hour, now.minute,
                      self.indicator
                      )
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # init model
        self._init_params(params, dataset_name)
        dataset = data.load(
            name=dataset_name,
            batch_size=self.params['generator_trainer']['batch_size'],
            **self.params['data'])
        self.model = self.Model(**self.params['model'])
        iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
        xreal = iter.get_next()
        self.trainer = self._init_trainer(xreal)

        # saver
        self.saver = tf.compat.v1.train.Saver(
            tdl.core.get_variables(self.model))

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

    def run(self, n_steps=100, **kwargs):
        if not kwargs:
            kwargs = self.params['run']
        if 'homogenize' not in kwargs:
            kwargs['homogenize'] = False
        if 'reset_embedding' not in kwargs:
            kwargs['reset_embedding'] = False
        if 'n_start' not in kwargs:
            kwargs['n_start'] = 0

        if not hasattr(self, '_global_steps'):
            self._global_steps = 0

        def train_gan():
            if kwargs['homogenize']:
                logits_h = self.model.embedding.logits
                _logits = logits_h.value().eval()
                logits = np.zeros(logits_h.shape.as_list(),
                                  dtype=logits_h.dtype.as_numpy_dtype)
                self._set_logits(logits=logits)
            if not run_training(
                    dis=self.trainer.dis, gen=self.trainer.gen,
                    n_steps=kwargs['gan_steps'], n_logging=10,
                    ind=self.indicator):
                return False
            if kwargs['homogenize']:
                self._set_logits(logits=_logits)
            return True

        def train_encoder():
            for i in tqdm.tqdm(range(kwargs['encoder_steps'])):
                self.session.run(self.trainer.enc.step['encoder'])

        def train_embedding():
            _n_steps = kwargs['embedding_steps']
            reset = False
            if kwargs['reset_embedding'] is not False:
                if self._global_steps % kwargs['reset_embedding'] == 0:
                    print('--> Resetting embedding.')
                    _n_steps = kwargs['reset_embedding']*_n_steps
                    reset = True

            def get_dict(sparsity):
                feed_dict = None
                if isinstance(self.trainer.enc.loss['embedding'],
                              tdl.core.SimpleNamespace):
                    feed_dict = {self.trainer.enc.loss['embedding'].sparsity:
                                 sparsity}
                return feed_dict

            if reset:
                self.session.run(self.model.embedding.init_op)
                feed_dict = get_dict(sparsity=0.0)
                for i in tqdm.tqdm(range(_n_steps)):
                    self.session.run(self.trainer.enc.step['embedding'],
                                     feed_dict=feed_dict)
                for i in tqdm.tqdm(range(_n_steps)):
                    self.session.run(self.trainer.enc.step['linear_disc'])
            else:
                for i in tqdm.tqdm(range(_n_steps)):
                    self.session.run(self.trainer.enc.step['linear_disc'])
                feed_dict = get_dict(sparsity=1.0)
                for i in tqdm.tqdm(range(_n_steps)):
                    self.session.run(self.trainer.enc.step['embedding'],
                                     feed_dict=feed_dict)

        # warm up trainin the gan
        if self._global_steps == 0:
            if not train_gan():
                return False
        # train for n_steps trials
        for trial in tqdm.tqdm(range(n_steps)):
            if self._global_steps >= kwargs['n_start']:
                train_encoder()
                train_embedding()
            if not train_gan():
                return False
            self._global_steps = self._global_steps + 1
        return True

    @eager_function
    def reconstruct(self, x_seed):
        encoded = self.model.encoder(x_seed)
        xrecon = self.model.generator(encoded.sample())
        return xrecon

    @eager_function
    def _set_logits(self, logits):
        logits_h = self.model.embedding.logits
        logits_op = logits_h.assign(logits)
        return logits_op

    def set_component(self, component):
        logits_h = self.model.embedding.logits
        logits = np.array(
            [30.0 if i == component else 0.0
             for i in range(logits_h.shape[-1].value)],
            dtype=logits_h.dtype.as_numpy_dtype)
        self._set_logits(logits=logits)

    def visualize_clusters(self, ax=None):
        '''visualize samples from gmm clusters.'''
        if ax is None:
            _, ax = plt.subplots(10, 10, figsize=(15, 15))
        _logits = self.model.embedding.logits.value().eval()
        n_components = _logits.shape[0]

        max_components = ax.shape[0]
        n_images = ax.shape[1]
        # comp_list = np.random.choice(
        #    n_components, max_components, replace=False)
        comp_sorted = np.argsort(_logits)[::-1]
        comp_list = np.concatenate([
            comp_sorted[:max_components//2],
            comp_sorted[-(max_components-max_components//2):]])
        assert len(comp_list) == max_components
        for component_idx, component in enumerate(comp_list):
            self.set_component(component)
            xsim = self.session.run(self.trainer.gen.xsim)
            while xsim.shape[0] < n_images:
                xsim = np.concatenate(
                    [xsim, self.session.run(self.trainer.gen.xsim)],
                    axis=0)
            for img_idx in range(n_images):
                image = normalize_image(xsim[img_idx, ...])
                ax[component_idx, img_idx].imshow(
                    np.squeeze(image),
                    interpolation='nearest')
                ax[component_idx, img_idx].axis('off')
        self._set_logits(logits=_logits)

    def visualize_reconstruction(self, ax=None):
        '''visualize the reconstruction of real images.'''
        if ax is None:
            fig, ax = plt.subplots(5, 8, figsize=(18, 10))
        n_real = ax.shape[1]
        xreal = self.trainer.dis.xreal.eval()
        while xreal.shape[0] < n_real:
            xreal = np.concatenate(
                [xreal, self.trainer.dis.xreal.eval()],
                axis=0)

        for i in range(ax.shape[1]):
            ax[0, i].imshow(np.squeeze(normalize_image(xreal[i, ...])),
                            interpolation='nearest')
            ax[0, i].axis('off')
        for j in range(1, ax.shape[0]):
            xrecon = self.reconstruct(x_seed=xreal)
            for i in range(ax.shape[1]):
                ax[j, i].imshow(np.squeeze(normalize_image(xrecon[i, ...])),
                                interpolation='nearest')
                ax[j, i].axis('off')

    def visualize_imgs(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        # dis.sim_pyramid[-1].eval()
        n_elements = functools.reduce(lambda x, y: x*y, ax.shape, 1)
        ax = np.reshape(ax, n_elements)
        xsim = self.session.run(self.trainer.gen.xsim)
        while xsim.shape[0] < n_elements:
            xsim = np.concatenate(
                [xsim, self.session.run(self.trainer.gen.xsim)],
                axis=0)
        for i in range(n_elements):
            image = (xsim[i][:, :, :]+1)*0.5
            ax[i].imshow(np.squeeze(normalize_image(image)),
                         interpolation='nearest')
            ax[i].axis('off')

    def visualize_manifold(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        Z = self.session.run(self.model.embedding.components.loc)
        Z_embedded = TSNE(n_components=2).fit_transform(Z)
        ax.scatter(Z_embedded[:, 0], Z_embedded[:, 1])

    def visualize(self, save=False, filename=None, visualize_manifold=False):
        if filename is None:
            folder = os.path.join(self.output_dir, 'images')
            if not os.path.exists(folder):
                os.makedirs(folder)
            now = datetime.datetime.now()
            filename = '{}/generated_{}{:02d}{:02d}_{:02d}{:02d}.pdf'.format(
                folder, now.year, now.month, now.day, now.hour, now.minute)

        fig = plt.figure(figsize=(13, 3*13))
        gs = fig.add_gridspec(30, 10)

        def reserve_ax(start, scale, shape):
            if isinstance(scale, int):
                scale = (scale, scale)
            ax = np.array(
                [fig.add_subplot(gs[
                    (start + i*scale[0]):(start + i*scale[0]+scale[0]),
                    j*scale[1]:j*scale[1]+scale[1]])
                 for i in range(shape[0]) for j in range(shape[1])])
            return np.reshape(ax, shape)

        ax = reserve_ax(start=0, scale=2, shape=(5, 5))
        self.visualize_imgs(ax=ax)

        ax = reserve_ax(start=10, scale=1, shape=(10, 10))
        self.visualize_clusters(ax=ax)

        ax = reserve_ax(start=20, scale=(2, 10), shape=(1, 1))
        ax1 = fig.add_subplot(gs[20:22, 0:8])
        ax2 = fig.add_subplot(gs[20:22, 8:10])
        probs = self.model.embedding.dist.cat.probs.eval()
        ax1.bar(x=range(probs.shape[0]), height=probs)
        if visualize_manifold:
            self.visualize_manifold(ax=ax2)

        ax = reserve_ax(start=22, scale=(1, 1), shape=(8, 10))
        self.visualize_reconstruction(ax=ax)

        if save:
            plt.savefig(filename)
            plt.close()

    def save(self, folder=None):
        '''save model params'''
        if folder is None:
            folder = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(folder):
            os.makedirs(folder)
        now = datetime.datetime.now()
        filename = '{}/vars_{}{:02d}{:02d}_{:02d}{:02d}.ckpt'.format(
            folder,
            now.year, now.month, now.day, now.hour, now.minute)
        self.saver.save(self.session, filename)


class ExperimentWACGAN(ExperimentGMM):
    name = 'wacgan'
    Model = WacGan


class ExperimentWACGAN_V2(ExperimentGMM):
    name = 'wacganV2'
    Model = WacGanV2


class ExperimentWACGAN_Dev(ExperimentGMM):
    name = 'wacganDev'
    Model = WacGanDev

    def _init_trainer(self, xreal):
        trainer = tdl.core.SimpleNamespace(
            gen=self.model.generator_trainer(
                **self.params['generator_trainer']),
            dis=self.model.discriminator_trainer(
                xreal=xreal, **self.params['discriminator_trainer']),
            encoder=self.model.encoder_trainer(
                xreal=xreal, **self.params['encoder_trainer']),
            embedding=self.model.embedding_trainer(
                xreal=xreal, **self.params['encoder_trainer']))
        tdl.core.variables_initializer(trainer.gen.variables).run()
        tdl.core.variables_initializer(trainer.dis.variables).run()
        tdl.core.variables_initializer(trainer.encoder.variables).run()
        tdl.core.variables_initializer(trainer.embedding.variables).run()
        return trainer

    def _train_encoder(self, encoder_steps):
        self.trainer.encoder.optim.run(n_steps=encoder_steps)

    def _train_embedding(self, embedding_steps, reset_embedding):
        _n_steps = embedding_steps
        reset = False
        if reset_embedding is not False:
            if self._global_steps % reset_embedding == 0:
                print('--> Resetting embedding.')
                _n_steps = reset_embedding*_n_steps
                reset = True

        def get_dict(sparsity):
            feed_dict = None
            sparsity_h = self.trainer.embedding.gmm['estimator']\
                             .loss.model.sparsity
            if isinstance(sparsity_h, tf.Tensor):
                feed_dict = {sparsity_h: sparsity}
            return feed_dict

        if reset:
            self.session.run(self.model.embedding.init_op)
            feed_dict = get_dict(sparsity=0.0)
            self.trainer.embedding.gmm['optim'].run(
                n_steps=_n_steps, feed_dict=feed_dict)

            self.trainer.embedding.linear['optim'].run(n_steps=_n_steps)
        else:

            self.trainer.embedding.linear['optim'].run(n_steps=_n_steps)
            feed_dict = get_dict(sparsity=1.0)
            self.trainer.embedding.gmm['optim'].run(
                n_steps=_n_steps, feed_dict=feed_dict)

    def run(self, n_steps=100, **kwargs):
        if not kwargs:
            kwargs = self.params['run']
        if 'homogenize' not in kwargs:
            kwargs['homogenize'] = False
        if 'reset_embedding' not in kwargs:
            kwargs['reset_embedding'] = False
        if 'n_start' not in kwargs:
            kwargs['n_start'] = 0

        if not hasattr(self, '_global_steps'):
            self._global_steps = 0

        def train_gan():
            if kwargs['homogenize']:
                logits_h = self.model.embedding.logits
                _logits = logits_h.value().eval()
                logits = np.zeros(logits_h.shape.as_list(),
                                  dtype=logits_h.dtype.as_numpy_dtype)
                self._set_logits(logits=logits)
            if not run_training(
                    dis=self.trainer.dis, gen=self.trainer.gen,
                    n_steps=kwargs['gan_steps'], n_logging=10,
                    ind=self.indicator):
                return False
            if kwargs['homogenize']:
                self._set_logits(logits=_logits)
            return True

        # warm up trainin the gan
        if self._global_steps == 0:
            if not train_gan():
                return False
        # train for n_steps trials
        for trial in tqdm.tqdm(range(n_steps)):
            if self._global_steps >= kwargs['n_start']:
                self._train_encoder(kwargs['encoder_steps'])
                self._train_embedding(
                    embedding_steps=kwargs['embedding_steps'],
                    reset_embedding=kwargs['reset_embedding'])
            if not train_gan():
                return False
            self._global_steps = self._global_steps + 1
        return True


class ExperimentBiGmmGan(ExperimentWACGAN_Dev):
    name = 'bigmmgan'
    Model = BiGmmGan

    def _init_trainer(self, xreal):
        trainer = tdl.core.SimpleNamespace(
            gen=self.model.generator_trainer(
                **self.params['generator_trainer']),
            dis=self.model.discriminator_trainer(
                xreal=xreal, **self.params['discriminator_trainer']),
            encoder=self.model.encoder_trainer(
                xreal=xreal, **self.params['encoder_trainer']),
            embedding=self.model.embedding_trainer(
                xreal=xreal, **self.params['embedding_trainer']))
        tdl.core.variables_initializer(trainer.gen.variables).run()
        tdl.core.variables_initializer(trainer.dis.variables).run()
        tdl.core.variables_initializer(trainer.encoder.variables).run()
        tdl.core.variables_initializer(trainer.embedding.variables).run()
        return trainer

    def _train_embedding(self, embedding_steps, reset_embedding):
        _n_steps = embedding_steps
        reset = False
        if reset_embedding is not False:
            if self._global_steps % reset_embedding == 0:
                print('--> Resetting embedding.')
                _n_steps = reset_embedding*_n_steps
                reset = True

        if reset:
            self.session.run(self.model.embedding.init_op)
        self.trainer.embedding.optim.run(n_steps=_n_steps)

    @eager_function
    def reconstruct(self, x_seed):
        encoded = self.model.encoder(x_seed)
        xrecon = self.model.generator(encoded.sample())
        return xrecon[-1]
