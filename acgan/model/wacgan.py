import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow.keras.layers as tf_layers
import tensorflow_probability as tfp
from .gmm_gan import GmmGan, GmmGeneratorTrainer
from .encoder import ResConv


class WacGanGeneratorTrainer(GmmGeneratorTrainer):
    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        var_list = list(set(tdl.core.get_trainable(self.generator)) |
                        set(tdl.core.get_trainable(self.model.embedding)))
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=var_list)
        return step


@tdl.core.create_init_docstring
class WacGan(GmmGan):
    GeneratorTrainer = WacGanGeneratorTrainer

    @tdl.core.SubmodelInit
    def discriminator(self, units, kernels, strides, dropout=None,
                      padding='same'):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = self.DiscriminatorBaseModel()
        for i in range(len(units)):
            model.add(self.DiscriminatorHidden(
                units=units[i], kernels=kernels[i], strides=strides[i],
                padding=padding[i], dropout=dropout))
        model.add(self.DiscriminatorOutput())
        return model


class WacGanV2(WacGan):
    EncoderModel = ResConv
    @tdl.core.SubmodelInit
    def encoder(self, units, kernels, strides, dropout=None, padding='same',
                pooling=None, flatten='global_maxpool', res_layers=2):
        '''CNN for recovering the encodding of a given image'''
        return self.EncoderModel(
            units=units, kernels=kernels, strides=strides, dropout=dropout,
            padding=padding, pooling=pooling, flatten=flatten,
            res_layers=res_layers)
