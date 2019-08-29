import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow_probability as tfp
import tensorflow.keras.layers as tf_layers
from .base import _add_noise
from .gmm import GMM
from .msg_gan import (MSG_GAN, MSG_DiscriminatorTrainer, MSG_GeneratorTrainer,
                      AffineLayer, Conv2DLayer, LEAKY_RATE)


def _loss_embedding():
    pass


class GmmDiscriminatorTrainer(MSG_DiscriminatorTrainer):
    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['batch_size'])
        dist = self.model.embedding(self.batch_size)
        return dist

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['real_pyramid', 'sim_pyramid', 'regularizer'])
        pred_real = self.discriminator(self.real_pyramid)
        pred_sim = self.discriminator(self.sim_pyramid)
        loss_real = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy()(
                tf.zeros_like(pred_sim), pred_sim)
        loss = (loss_real + loss_sim)/2.0
        return loss
        # Regularizer
        if self.regularizer is not None:
            layers = tdl.core.find_instances(
                self.discriminator,
                (tf.keras.layers.Conv2D, tdl.convnet.Conv2DLayer,
                 tdl.convnet.Conv1x1Proj))
            loss += self.regularizer.scale*tf.add_n(
                [self.regularizer.fn(layer.kernel)
                 for layer in layers]
            )
        return loss


class GmmGeneratorTrainer(MSG_GeneratorTrainer):
    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['batch_size'])
        dist = self.model.embedding(self.batch_size)
        return dist.mean()

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['batch_size', 'xsim', 'sim_pyramid', 'regularizer',
                           'pyramid_loss'])
        pred = self.discriminator(self.sim_pyramid)
        loss = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(pred), pred)
        # regularizer
        if self.regularizer is not None:
            layers = tdl.core.find_instances(
                self.generator,
                (tf.keras.layers.Conv2D, tdl.convnet.Conv2DLayer,
                 tdl.convnet.Conv1x1Proj))
            loss += self.regularizer.scale*tf.add_n(
                [self.regularizer.fn(layer.kernel)
                 for layer in layers]
            )
        if self.pyramid_loss is not None:
            loss += self.pyramid_loss.scale * self.pyramid_loss.fn()
        return loss

    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        var_list = list(set(tdl.core.get_trainable(self.generator)))
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=var_list)
        return step


@tdl.core.create_init_docstring
class GmmGan(MSG_GAN):
    EmbeddingModel = GMM

    @tdl.core.SubmodelInit
    def embedding(self, n_dims, n_components, init_loc=1e-5, init_scale=1.0):
        model = self.EmbeddingModel(
            n_dims=n_dims, n_components=n_components,
            components={'init_loc': init_loc,
                        'init_scale': init_scale})
        return model

    @tdl.core.SubmodelInit
    def encoder(self, units, kernels, strides, dropout=None, padding='same'):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = self.EncoderModel()
        for i in range(len(units)):
            model.add(self.EncoderHidden(
                filters=units[i], strides=strides[i],
                kernel_size=kernels[i], padding=padding[i]))
            model.add(tf_layers.LeakyReLU(LEAKY_RATE))
            if dropout is not None:
                model.add(tf_layers.Dropout(rate=dropout))

        model.add(tf_layers.Flatten())
        model.add(AffineLayer(units=self.embedding_size))
        return model

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None,
                              optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid', 'encoder'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.DiscriminatorTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer=optimizer,
            **kwargs)

    def generator_trainer(self, batch_size, optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'generator_trainer',
            ['generator', 'discriminator', 'pyramid', 'encoder'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.GeneratorTrainer(
            model=self, batch_size=2*batch_size,
            optimizer=optimizer,
            **kwargs)
