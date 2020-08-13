from types import SimpleNamespace

import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers


class Conv2DLayer(tdl.convnet.Conv2DLayer):
    @tdl.core.InputArgument
    def filters(self, value):
        '''Number of filters (int), equal to the number of output maps.'''
        if value is None:
            tdl.core.assert_initialized(self, 'filters', ['input_shape'])
            value = self.input_shape[-1].value
        if not isinstance(value, int):
            raise TypeError('filters must be an integer')
        return value

    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['kernel_size', 'input_shape'])
        if initializer is None:
            initializer = tf.compat.v1.keras.initializers\
                            .RandomNormal(stddev=1.0)
        weight = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.kernel_size[0], self.kernel_size[1],
                   self.input_shape[-1].value, self.filters],
            trainable=trainable,
            **kargs)
        fan_in, fan_out = tdl.core.initializers.compute_fans(weight.shape)
        return weight * tf.sqrt(2.0/(fan_in.value + fan_out.value))


class Conv1x1Proj(tdl.convnet.Conv1x1Proj):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['units', 'input_shape'])
        if initializer is None:
            initializer = tf.compat.v1.keras.initializers\
                            .RandomNormal(stddev=1.0)
        kernel = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.input_shape[-1].value, self.units],
            trainable=trainable,
            **kargs)
        fan_in, fan_out = tdl.core.initializers.compute_fans(kernel.shape)
        return kernel * tf.sqrt(2.0/(fan_in.value + fan_out.value))


class Conv2DTranspose(tdl.convnet.Conv2DTranspose):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['kernel_size', 'input_shape'])
        if initializer is None:
            initializer = tf.compat.v1.keras.initializers\
                            .RandomNormal(stddev=1.0)
        kernel = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.kernel_size[0], self.kernel_size[1],
                   self.filters, self.input_shape[-1].value],
            trainable=trainable,
            **kargs)
        # see https://github.com/tkarras/progressive_growing_of_gans/blob/
        #   master/networks.py
        fan_in = max(self.kernel_size)
        return kernel * tf.sqrt(2.0/fan_in)


@tdl.core.create_init_docstring
class AffineLayer(tdl.dense.AffineLayer):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['units', 'input_shape'])
        if initializer is None:
            initializer = tf.compat.v1.keras.initializers\
                            .RandomNormal(stddev=1.0)
        kernel = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.input_shape[-1].value, self.units],
            trainable=trainable,
            **kargs)
        fan_in, fan_out = tdl.core.initializers.compute_fans(kernel.shape)
        return kernel * tf.sqrt(2.0/(fan_in.value + fan_out.value))


def get_layer_lib(lib_name='keras'):
    if lib_name == 'keras':
        return SimpleNamespace(
            Conv1x1Proj=tdl.convnet.Conv1x1Proj,
            Conv2D=tf_layers.Conv2D,
            Conv2DTranspose=tdl.convnet.Conv2DTranspose,
            Dense=tf_layers.Dense)
    elif lib_name == 'equalized':
        return SimpleNamespace(
            Conv1x1Proj=Conv1x1Proj,
            Conv2D=Conv2DLayer,
            Conv2DTranspose=Conv2DTranspose,
            Dense=AffineLayer)
    else:
        raise ValueError(f'unrecognized lib_name {lib_name}')
