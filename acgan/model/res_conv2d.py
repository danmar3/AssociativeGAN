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
    def batchnorm(self, method='batchnorm'):
        if isinstance(method, str):
            method = BATCHNORM_TYPES[method]
        return method

    @tdl.core.SubmodelInit(lazzy=True)
    def upsample(self, method='bilinear', layers=None, size=2, data_format=None):
        def upsampling():
            return tf_layers.UpSampling2D(
                size=size, data_format=data_format, interpolation=method)

        if layers is None:
            method = None
        elif method in ('nearest, bilinear'):
            method = upsampling
        if isinstance(layers, int):
            layers = [layers]
        return SimpleNamespace(method=method, layers=layers)

    @tdl.core.SubmodelInit(lazzy=True)
    def conv(self, filters=None, kernels=3, padding='same'):
        tdl.core.assert_initialized(
            self, 'conv', ['use_bias', 'leaky_rate',
                           'Conv2DClass', 'batchnorm'])
        if filters is None:
            filters = [None, None]
        kernels = replicate_to_list(kernels, len(filters))
        layers = tdl.stacked.StackedLayers()
        for idx in range(len(filters)):
            if self.batchnorm:
                layers.add(self.batchnorm())

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

    PROJECTION_TYPE = {
        'conv1x1',
        'drop'
        'concat'
    }

    @tdl.core.InputArgument
    def batchnorm(self, value):
        """batch normalization method. Defaults to batchnorm."""
        if value is None:
            return None
        elif isinstance(value, dict):
            return value['method']
        return value

    DEFAULT_RESIZE = {'upsample': 'bilinear',
                      'downsample': 'avg_pool',
                      'resize': 'bilinear'}

    @tdl.core.SubmodelInit(lazzy=True)
    def projection(self, units=None,
                   on_increment='conv1x1', on_decrement='conv1x1',
                   use_bias=False, channel_dim=-1):
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
    def residual(self, conv=None, pooling=None, upsample=None,
                 batchnorm='batchnorm', **kargs):
        tdl.core.assert_initialized(
            self, 'residual', ['use_bias', 'equalized'])
        # conv arg
        if conv is None:
            conv = {}
        # batchnorm arg
        if self.batchnorm is not None:
            batchnorm = self.batchnorm

        # build residual
        return Conv2DBlock(
            conv=conv, pooling=pooling, upsample=upsample,
            use_bias=self.use_bias,
            equalized=self.equalized,
            batchnorm=(batchnorm if isinstance(batchnorm, dict)
                       else {'method': batchnorm}),
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


class ResBlockPreAct(ResBlock):
    @tdl.core.SubmodelInit(lazzy=True)
    def pre_activation(self, enable=True, batchnorm='batchnorm',
                       leaky_rate=0.2):
        tdl.core.assert_initialized(self, 'pre_activation', ['batchnorm'])
        # batchnorm arg
        if self.batchnorm is not None:
            batchnorm = self.batchnorm
        if isinstance(batchnorm, dict):
            batchnorm = batchnorm['method']
        if isinstance(batchnorm, str):
            batchnorm = BATCHNORM_TYPES[batchnorm]

        if enable:
            layers = tdl.stacked.StackedLayers()
            if batchnorm:
                layers.add(batchnorm())
            layers.add(tf_layers.LeakyReLU(leaky_rate))
            return layers
        else:
            return None

    @tdl.core.SubmodelInit(lazzy=True)
    def residual(self, conv=None, pooling=None, upsample=None,
                 batchnorm='batchnorm', **kargs):
        """residual network.
        Args:
            conv: dict{'filters': list(int), 'kernels': list(int)}
            pooling: dict{size: int, method: str}
            upsample: dict{method: str, layers: list(int)}
            batchnorm: str | dict{method: str}
        """
        tdl.core.assert_initialized(
            self, 'residual', ['use_bias', 'equalized', 'pre_activation'])
        # conv arg
        if conv is None:
            conv = {'filters': [None, None]}
        conv['kernels'] = replicate_to_list(
            conv['kernels'], len(conv['filters']))
        # batchnorm arg
        if self.batchnorm is not None:
            batchnorm = self.batchnorm

        # build residual
        if self.pre_activation:
            layers = tdl.stacked.StackedLayers()

            conv2d_type = (equalized.Conv2DLayer if self.equalized
                           else tf_layers.Conv2D)
            layers.add(conv2d_type(
                filters=conv['filters'][0],
                kernel_size=conv['kernels'][0],
                padding='same',
                use_bias=self.use_bias
            ))

            if isinstance(upsample, dict) and ('layers' in upsample):
                if isinstance(upsample['layers'], int):
                    upsample['layers'] = [upsample['layers']]
                upsample['layers'] = [
                    max(0, li-1) for li in upsample['layers']]

            if conv['filters'][1:]:
                layers.add(Conv2DBlock(
                    conv={'filters': conv['filters'][1:],
                          'kernels': conv['kernels'][1:]},
                    pooling=pooling, upsample=upsample,
                    use_bias=self.use_bias,
                    equalized=self.equalized,
                    batchnorm=(batchnorm if isinstance(batchnorm, dict)
                               else {'method': batchnorm}),
                    **kargs))
            return layers

        return Conv2DBlock(
            conv=conv, pooling=pooling, upsample=upsample,
            use_bias=self.use_bias,
            equalized=self.equalized,
            batchnorm=(batchnorm if isinstance(batchnorm, dict)
                       else {'method': batchnorm}),
            **kargs)

    def call(self, inputs):
        if self.pre_activation:
            inputs = self.pre_activation(inputs)

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


class ResStages(tdl.stacked.StackedLayers):
    stages = tdl.core.InputArgument.optional(
        'stages', doc='number of stages.', default=1)
    units = tdl.core.InputArgument.required(
        'units', doc='number of output units')
    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.', default=None)
    kernel_size = tdl.core.InputArgument.optional(
        'kernel_size', doc='kernel size.', default=3)
    leaky_rate = tdl.core.InputArgument.optional(
        'leaky_rate', doc='leaky rate', default=0.2)
    dropout = tdl.core.InputArgument.optional(
        'dropout', doc='dropout rate', default=None)
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)
    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    @tdl.core.InputArgument
    def pooling(self, value):
        tdl.core.assert_initialized(self, 'pooling', ['stages'])
        if value is None:
            return [{'size': None} for _ in range(self.stages)]
        if isinstance(value, int):
            return [{'size': value if idx == 0 else None}
                    for idx in range(self.stages)]
        if isinstance(value, dict):
            pooling_size = value['size']
            del value['size']
            return [{**value, **{'size': pooling_size if idx == 0 else None}}
                    for idx in range(len(self.units))]
        raise ValueError('pooling should be either an int or a dict.')

    @tdl.core.SubmodelInit(lazzy=True)
    def layers(self, **kargs):
        tdl.core.assert_initialized(
            self, 'layers', ['stages', 'units', 'kernel_size', 'leaky_rate',
                             'pooling', 'dropout', 'batchnorm', 'use_bias',
                             'equalized'])
        layers = list()
        for idx in range(self.stages):
            layers.append(ResBlockPreAct(
                pre_activation={'enable': True if idx == 0 else False},
                residual={
                    'conv': {
                        'filters': [self.units, self.units],
                        'kernels': self.kernel_size},
                    'leaky_rate': self.leaky_rate,
                    'pooling': self.pooling[idx],
                    'dropout': {'rate': self.dropout},
                    },
                batchnorm={'method': self.batchnorm},
                use_bias=self.use_bias,
                equalized=self.equalized,
                **kargs))
        return layers


class ResStages101(ResStages):
    """
    https://arxiv.org/pdf/1603.05027.pdf
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    """
    stages = tdl.core.InputArgument.optional(
        'stages', doc='number of stages.', default=2)

    @tdl.core.SubmodelInit(lazzy=True)
    def layers(self, **kargs):
        tdl.core.assert_initialized(
            self, 'layers',
            ['stages', 'units', 'kernel_size', 'leaky_rate', 'pooling',
             'dropout', 'batchnorm', 'use_bias', 'equalized'])
        layers = list()
        for idx in range(self.stages):
            layers.append(ResBlockPreAct(
                pre_activation={'enable': True if idx == 0 else False},
                residual={
                    'conv': {
                        'filters': [self.units//4, self.units//4, self.units],
                        'kernels': [1, self.kernel_size, 1]},
                    'leaky_rate': self.leaky_rate,
                    'pooling': self.pooling[idx],
                    'dropout': {'rate': self.dropout},
                    },
                batchnorm={'method': self.batchnorm},
                use_bias=self.use_bias,
                equalized=self.equalized,
                **kargs))
        return layers


class ResNet(tdl.core.Layer):
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)

    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    leaky_rate = tdl.core.InputArgument.optional(
        'leaky_rate', doc='leaky rate', default=0.2)

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, units, pooling=None, kernels=3,
               dropout=None, batchnorm=None, stages=3):
        """hidden layers
        Args:
            units: list(int) with n_layers elements.
            pooling: int | None | list(int|None) with n_layers elements.
            kernels: int | list(int) with n_layers elements.
            batchnorm: str | None.
        """
        tdl.core.assert_initialized(
            self, 'hidden', ['equalized', 'use_bias', 'leaky_rate'])
        n_layers = len(units)
        # kernels arg
        assert isinstance(kernels, (int, list, tuple))
        if isinstance(kernels, int):
            kernels = [kernels for _ in range(n_layers)]
        assert all(isinstance(ki, int) for ki in kernels)
        # pooling arg
        assert (isinstance(pooling, (int, list, tuple)) or (pooling is None))
        if isinstance(pooling, (int, dict)):
            pooling = [pooling for _ in range(n_layers)]
        assert all(isinstance(pi, (int, dict)) or (pi is None)
                   for pi in pooling)

        model = tdl.stacked.StackedLayers()
        for i in range(len(units)):
            model.add(ResStages101(
                stages=stages,
                units=units[i],
                kernel_size=kernels[i],
                batchnorm=batchnorm,
                leaky_rate=self.leaky_rate,
                pooling=pooling[i],
                dropout=dropout,
                use_bias=self.use_bias,
                equalized=self.equalized
                ))
        return model

    @tdl.core.SubmodelInit(lazzy=True)
    def flatten(self, method='global_maxpool', batchnorm=None, activation=True):
        layers = tdl.stacked.StackedLayers()
        if batchnorm:
            if isinstance(batchnorm, str):
                batchnorm = BATCHNORM_TYPES[batchnorm]
            layers.add(batchnorm())
        if activation:
            layers.add(tf_layers.LeakyReLU(self.leaky_rate))
        if method == 'flatten':
            layers.add(tf_layers.Flatten())
        elif method == 'global_maxpool':
            layers.add(tf_layers.GlobalMaxPooling2D())
        elif method == 'global_avgpool':
            layers.add(tf_layers.GlobalAveragePooling2D())
        else:
            raise ValueError(f'flatten option {method} not recognized')
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def dense(self, units, batchnorm=None):
        tdl.core.assert_initialized(self, 'dense', ['equalized', 'leaky_rate'])
        if not units:
            return None
        if isinstance(units, int):
            units = [units]
        if isinstance(batchnorm, str):
            batchnorm = BATCHNORM_TYPES[batchnorm]
        dense_layer = (equalized.AffineLayer if self.equalized
                       else tf_layers.Dense)

        layers = tdl.stacked.StackedLayers()
        for ui in units[:-1]:
            if batchnorm is not None:
                layers.add(batchnorm())
            layers.add(tf_layers.LeakyReLU(self.leaky_rate))
            layers.add(dense_layer(units=ui))
        if batchnorm is not None:
            layers.add(batchnorm())
        layers.add(dense_layer(units=units[-1]))
        return layers

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(
            self, 'compute_output_shape', ['hidden', 'flatten', 'dense'])
        output_shape = self.hidden.compute_output_shape(input_shape)
        if self.flatten:
            output_shape = self.flatten.compute_output_shape(output_shape)
        if self.dense:
            output_shape = self.dense.compute_output_shape(output_shape)
        return output_shape

    def call(self, inputs, output='dense'):
        out = self.hidden(inputs)
        if output == 'hidden':
            return out
        if self.flatten:
            out = self.flatten(out)
        if output == 'flatten':
            return out
        if self.dense:
            out = self.dense(out)
        if output == 'dense':
            return out
        raise ValueError(f'option {output} not recognized as valid output.')
