import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow_probability as tfp
import tensorflow.keras.layers as tf_layers
from .base import _add_noise
from .msg_gan import (MSG_GAN, MSG_DiscriminatorTrainer, MSG_GeneratorTrainer,
                      AffineLayer, Conv2DLayer, LEAKY_RATE)


class GeneratorTrainer(MSG_GeneratorTrainer):
    @tdl.core.InputArgument
    def xinput(self, value):
        tdl.core.assert_initialized(self, 'xinput', ['train_step'])
        if value is None:
            raise ValueError('xinput not provided')
        return value

    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['xinput'])
        dist = self.model.encoder(self.xinput)
        return tdl.core.SimpleNamespace(
            value=dist.sample(), distribution=dist)

    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        var_list = list(set(tdl.core.get_trainable(self.generator)) |
                        set(tdl.core.get_trainable(self.model.encoder)))
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=var_list)
        return step

    @property
    def variables(self):
        tdl.core.assert_initialized(
            self, 'variables', ['optimizer', 'train_step'])
        var_list = list(set(tdl.core.get_trainable(self.generator)) |
                        set(tdl.core.get_trainable(self.model.encoder)))
        return (self.optimizer.variables() + [self.train_step] + var_list)

    @tdl.core.SubmodelInit
    def prior(self, scale=1.0):
        return tfp.distributions.Normal(loc=0, scale=scale)

    @tdl.core.InputArgument
    def reg_factor(self, value):
        if value is None:
            value = 0.001
        return value

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['batch_size', 'xsim', 'sim_pyramid', 'regularizer',
                           'pyramid_loss', 'prior'])
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
            loss += (self.reg_factor*self.pyramid_loss.scale
                     * self.pyramid_loss.fn())
        # kl divergence
        kl_loss = tf.reduce_sum(
            tfp.distributions.kl_divergence(
                self.embedding.distribution, self.prior),
            axis=-1)
        loss = loss + self.reg_factor*tf.reduce_mean(kl_loss)
        return loss


class CyclicGeneratorTrainer(GeneratorTrainer):
    @tdl.core.Submodel
    def cyclic_embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['sim_pyramid'])
        dist = self.model.encoder(self.sim_pyramid[-1])
        return tdl.core.SimpleNamespace(
            value=dist.sample(), distribution=dist)

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['batch_size', 'xsim', 'sim_pyramid', 'regularizer',
                           'pyramid_loss', 'prior', 'cyclic_embedding'])
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
            loss += (self.reg_factor*self.pyramid_loss.scale
                     * self.pyramid_loss.fn())
        # kl divergence
        kl_loss = tf.reduce_sum(
            tfp.distributions.kl_divergence(
                self.embedding.distribution, self.prior),
            axis=-1)
        loss = loss + self.reg_factor*tf.reduce_mean(kl_loss)
        # cyclic divergence
        kl_cyclic = tf.reduce_sum(
            tfp.distributions.kl_divergence(
                self.cyclic_embedding.distribution,
                self.embedding.distribution),
            axis=-1)
        loss = loss + self.reg_factor*tf.reduce_mean(kl_cyclic)
        return loss


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


class CyclicGeneratorTrainerV2(CyclicGeneratorTrainer):
    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['xinput'])
        dist = self.model.encoder(self.xinput)
        return dist.mean()

    @tdl.core.Submodel
    def cyclic_embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['sim_pyramid'])
        dist = self.model.encoder(self.sim_pyramid[-1])
        return dist.mean()

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['batch_size', 'xsim', 'sim_pyramid', 'regularizer',
                           'pyramid_loss', 'prior', 'cyclic_embedding'])
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
            loss += (self.reg_factor*self.pyramid_loss.scale
                     * self.pyramid_loss.fn())
        # kl divergence
        embedding = self.embedding
        aggregated_dist = tfp.distributions.Normal(
            loc=tf.reduce_mean(embedding, 0),
            scale=reduce_std(embedding, 0))
        kl_loss = tf.reduce_sum(
            tfp.distributions.kl_divergence(
                aggregated_dist, self.prior),
            axis=-1)
        loss = loss + self.reg_factor*tf.reduce_mean(kl_loss)
        # cyclic divergence
        kl_cyclic = tf.reduce_sum(
            tf.square(self.cyclic_embedding - self.embedding),
            axis=-1)
        loss = loss + self.reg_factor*tf.reduce_mean(kl_cyclic)
        return loss


class DiscriminatorTrainer(MSG_DiscriminatorTrainer):
    @tdl.core.InputArgument
    def xreal(self, value):
        tdl.core.assert_initialized(self, 'xreal', ['train_step'])
        noise_rate = self.model.noise_rate(self.train_step)
        if noise_rate is not None:
            value = _add_noise(value, noise_rate)
        return value

    @tdl.core.InputArgument
    def xinput(self, value):
        tdl.core.assert_initialized(self, 'xinput', ['train_step'])
        if value is None:
            raise ValueError('xinput not provided')
        return value

    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['xinput'])
        dist = self.model.encoder(self.xinput)
        return tdl.core.SimpleNamespace(
            value=dist.sample(), distribution=dist)


class CyclicDiscriminatorTrainer(DiscriminatorTrainer):
    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['xinput'])
        dist = self.model.encoder(self.xinput)
        return dist.mean()


@tdl.core.create_init_docstring
class ACGAN(MSG_GAN):
    EncoderModel = tdl.stacked.StackedLayers
    EncoderHidden = Conv2DLayer
    GeneratorTrainer = CyclicGeneratorTrainerV2
    DiscriminatorTrainer = CyclicDiscriminatorTrainer

    @tdl.core.SubmodelInit
    def encoder(self, units, kernels, strides, padding='same'):
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

        model.add(tf_layers.Flatten())
        model.add(AffineLayer(units=self.embedding_size))
        model.add(twodlearn.bayesnet.NormalModel(
            loc=lambda x: x,
            batch_shape=self.embedding_size
        ))
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
