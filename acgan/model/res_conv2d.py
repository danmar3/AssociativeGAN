import twodlearn as tdl
from twodlearn.resnet import ResConv2D, tf_compat_dim, convnet
from twodlearn.core import SimpleNamespace

import tensorflow as tf
import tensorflow.keras.layers as tf_layers


from . import equalized
from .base import VectorNormalizer
from ..utils import replicate_to_list


BATCHNORM_TYPES = {
    'batchnorm': tf_layers.BatchNormalization,
    'pixelwise': VectorNormalizer
}


class Conv2DBlock(tdl.core.Layer):
    ''' follows residual architectures from
        - https://arxiv.org/pdf/1603.05027.pdf
        - https://arxiv.org/pdf/1809.11096.pdf
    '''
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)

    leaky_rate = tdl.core.InputArgument.optional(
        'leaky_rate', doc='leaky rate', default=0.2)

    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    @tdl.core.Submodel
    def Conv2DClass(self, value):
        tdl.core.assert_initialized(self, 'Conv2DClass', ['equalized'])
        if value is None:
            value = (equalized.Conv2DLayer if self.equalized
                     else tf_layers.Conv2D)
        return value

    @tdl.core.SubmodelInit
    def BatchNormClass(self, method='batchnorm'):
        if isinstance(method, str):
            method = BATCHNORM_TYPES[method]
        return method

    @tdl.core.SubmodelInit(lazzy=True)
    def upsample(self, method='bilinear', layer=None, size=2, data_format=None):
        if layer is None:
            method = None
        elif method in ('nearest, bilinear'):
            method = tf_layers.UpSampling2D(
                size=size, data_format=data_format, interpolation=method)
        return SimpleNamespace(method=method, layer=layer)

    @tdl.core.SubmodelInit(lazzy=True)
    def conv(self, filters=None, kernels=3, padding='same'):
        tdl.core.assert_initialized(
            self, 'conv', ['use_bias', 'leaky_rate',
                           'Conv2DClass', 'BatchNormClass'])
        if filters is None:
            filters = [None, None]
        kernels = replicate_to_list(kernels, len(filters))
        layers = tdl.stacked.StackedLayers()
        for idx in range(len(filters)):
            if self.BatchNormClass:
                layers.add(self.BatchNormClass())

            layers.add(
                tf_layers.LeakyReLU(self.leaky_rate))

            if self.upsample and idx in self.upsample.layers:
                layers.add(self.upsample.method())

            kargs = {'filters': filters[idx]} if filters[idx] else {}
            layers.add(
                self.Conv2DClass(
                    kernel_size=kernels[idx], strides=[1, 1],
                    padding=padding,
                    use_bias=self.use_bias,
                    **kargs))
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def pooling(self, size=None, method="average"):
        if size is not None:
            if method == "average":
                return tf.keras.layers.AveragePooling2D(pool_size=size)
            else:
                ValueError(f'pooling method {method} not recognized.')
        else:
            return None

    @tdl.core.SubmodelInit(lazzy=True)
    def dropout(self, rate=None):
        if rate is not None:
            return tf_layers.Dropout(rate)
        else:
            return None

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(
            self, 'compute_output_shape', ['conv', 'pooling', 'dropout'])
        output_shape = self.conv.compute_output_shape(input_shape)
        if self.pooling:
            output_shape = self.pooling.compute_output_shape(output_shape)
        if self.dropout:
            output_shape = self.dropout.compute_output_shape(output_shape)
        return output_shape

    def call(self, inputs, **kargs):
        value = self.conv(inputs)
        if self.pooling:
            value = self.pooling(value)
        if self.dropout:
            value = self.dropout(value)
        return value


class ResBlock(ResConv2D):
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)

    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    DEFAULT_RESIZE = {'upsample': 'bilinear',
                      'downsample': 'avg_pool',
                      'resize': 'bilinear'}

    @tdl.core.SubmodelInit(lazzy=True)
    def projection(self, units=None, use_bias=False, channel_dim=-1):
        tdl.core.assert_initialized(
            self, 'projection', ['input_shape', 'residual', 'equalized'])
        input_shape = self.input_shape
        input_units = tf_compat_dim(input_shape[channel_dim])
        if units is None:
            output_shape = self.residual.compute_output_shape(input_shape)
            output_units = tf_compat_dim(output_shape[channel_dim])
        else:
            output_units = units
        if input_units == output_units:
            return tf_layers.Activation(activation=tf.identity)
        else:
            if self.equalized:
                return equalized.Conv1x1Proj(units=output_units)
            else:
                return convnet.Conv1x1Proj(units=output_units)

    @tdl.core.SubmodelInit(lazzy=True)
    def residual(self, conv=None, pooling=None, upsample=None, **kargs):
        tdl.core.assert_initialized(
            self, 'residual', ['use_bias', 'equalized'])
        if conv is None:
            conv = {}

        return Conv2DBlock(
            conv=conv, pooling=pooling, upsample=upsample,
            use_bias=self.use_bias,
            equalized=self.equalized,
            **kargs)

    def _resize_operation(self):
        output_shape = self.residual.compute_output_shape(
                input_shape=self.input_shape)
        input_size = self.input_shape[1:-1].as_list()
        output_size = output_shape[1:-1].as_list()
        if all(oi >= ii for ii, oi in zip(input_size, output_size)):
            return 'upsample'
        elif all(ii >= oi == 0 for ii, oi in zip(input_size, output_size)):
            return 'downsample'
        else:
            return None

    def call(self, inputs):
        residual = self.residual(inputs)

        if self.upsample is not None:
            resized = self.upsample(inputs)
            shortcut = self.projection(resized)
        elif self.downsample is not None:
            projection = self.projection(inputs)
            shortcut = self.downsample(projection)
        elif self.resize is not None:
            if self._resize_operation() == 'upsample':
                resized = self.resize(inputs)
                shortcut = self.projection(resized)
            elif self._resize_operation() == 'downsample':
                projection = self.projection(inputs)
                shortcut = self.resize(projection)
            else:
                raise ValueError('resize operation is a mix of upsampling and '
                                 'downsampling.')
        else:
            shortcut = inputs

        return shortcut + residual


class ResNet(tdl.core.Layer):
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)

    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    leaky_rate = tdl.core.InputArgument.optional(
        'leaky_rate', doc='leaky rate', default=0.2)

    units = tdl.core.InputArgument.required()

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, kernels, strides, dropout=None, padding='same', batchnorm=None):
        tdl.core.assert_initialized(
            self, 'hidden', ['units', 'equalized', 'use_bias', 'leaky_rate'])
        n_layers = len(self.units)
        kernels = replicate_to_list(kernels, n_layers)
        strides = replicate_to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = tdl.staked.StackedLayers()
        for i in range(len(self.units)):
            model.add(ResBlock(
                residual={
                    'conv': {
                        'filters': [self.units[i], self.units[i]],
                        'kernels': kernels[i],
                        'padding': padding[i]},
                    'BatchNormClass': {'method': batchnorm},
                    'leaky_rate': self.leaky_rate,
                    'pooling': {'size': strides[i]},
                    'dropout': {'rate': dropout},
                    },
                use_bias=self.use_bias,
                equalized=self.equalized))
        return model

    @tdl.core.SubmodelInit(lazzy=True)
    def flatten(self, method='global_maxpool', activation=True):
        layers = tdl.staked.StackedLayers()
        if activation:
            layers.add(tf_layers.LeakyReLU(self.leaky_rate))
        if method == 'flatten':
            layers.add(tf_layers.Flatten())
        elif method == 'global_maxpool':
            layers.add(tf_layers.GlobalMaxPooling2D())
        else:
            raise ValueError(f'flatten option {method} not recognized')
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def dense(self, units, batchnorm=None):
        tdl.core.assert_initialized(self, 'dense', ['equalized', 'leaky_rate'])
        if isinstance(units, int):
            units = [units]
        if isinstance(batchnorm, str):
            batchnorm = BATCHNORM_TYPES[batchnorm]
        dense_layer = (equalized.AffineLayer if self.equalized
                       else tf_layers.Dense)

        layers = tdl.staked.StackedLayers()
        for ui in units[:-1]:
            if batchnorm is not None:
                layers.add(batchnorm())
            layers.add(tf_layers.LeakyReLU(self.leaky_rate))
            layers.add(dense_layer(units=ui))
        if batchnorm is not None:
            layers.add(batchnorm())
        layers.add(dense_layer(units=units[-1]))

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(
            self, 'compute_output_shape', ['hidden', 'flatten', 'dense'])
        output_shape = self.hidden.compute_output_shape(input_shape)
        if self.flatten:
            output_shape = self.flatten.compute_output_shape(output_shape)
        if self.dense:
            output_shape = self.dense.compute_output_shape(output_shape)
        return output_shape

    def call(self, inputs):
        pass
