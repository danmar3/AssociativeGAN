import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow.keras.layers as tf_layers
from .base import _add_noise
from .msg_gan import (MSG_GAN, MSG_DiscriminatorTrainer, MSG_GeneratorTrainer,
                      AffineLayer)


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
            value=dist.sample(), dist=dist)

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
            value=dist.sample(), dist=dist)


@tdl.core.create_init_docstring
class ACGAN(MSG_GAN):
    EncoderModel = tdl.stacked.StackedLayers

    @tdl.core.SubmodelInit
    def encoder(self, units, kernels, strides, padding):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = self.EncoderModel()
        for i in range(len(units)):
            model.add(self.EncoderHidden(
                units=units[i], strides=strides[i],
                kernels=kernels[i], padding=padding[i]))

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
