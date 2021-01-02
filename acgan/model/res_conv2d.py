import twodlearn as tdl
from twodlearn.resnet import ResConv2D, tf_compat_dim, convnet
from twodlearn.core import SimpleNamespace

import tensorflow as tf
import tensorflow.keras.layers as tf_layers


from . import equalized
from .base import VectorNormalizer
from .non_local import NonLocalBlock
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
    def upsample(self, size=None, method='bilinear', layers=0,
                 data_format=None):
        def upsampling():
            return tf_layers.UpSampling2D(
                size=size, data_format=data_format, interpolation=method)

        if (layers is None) or (size is None):
            method_fn = None
        elif method in ('nearest, bilinear'):
            method_fn = upsampling
        else:
            method_fn = method
        if isinstance(layers, int):
            layers = [layers]
        return SimpleNamespace(method=method_fn, layers=layers)

    @tdl.core.SubmodelInit(lazzy=True)
    def conv(self, filters=None, kernels=3, padding='same',
             batchnorm=True, activation=True, strides=None):
        tdl.core.assert_initialized(
            self, 'conv', ['use_bias', 'leaky_rate',
                           'Conv2DClass', 'batchnorm', 'upsample'])
        if filters is None:
            filters = [None, None]
        # kernels
        kernels = replicate_to_list(kernels, len(filters))
        # batchnorm
        if batchnorm in (True, False):
            batchnorm = [batchnorm for _ in range(len(filters))]
        # activation
        if activation in (True, False):
            activation = [activation for _ in range(len(filters))]
        # strides
        if strides is None:
            strides = [1, 1]
        # layers
        layers = tdl.stacked.StackedLayers()
        for idx in range(len(filters)):
            if batchnorm[idx] and self.batchnorm:
                layers.add(self.batchnorm())

            if activation[idx]:
                layers.add(tf_layers.LeakyReLU(self.leaky_rate))

            if self.upsample and self.upsample.method and idx in self.upsample.layers:
                layers.add(self.upsample.method())

            kargs = {'filters': filters[idx]} if filters[idx] else {}
            layers.add(
                self.Conv2DClass(
                    kernel_size=kernels[idx],
                    strides=(strides if idx == 0 else [1, 1]),
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


class ZeroPad(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of output units')

    def call(self, inputs):
        tdl.core.assert_initialized(self, 'call', ['input_shape', 'units'])
        n_dims = self.input_shape.ndims
        input_units = self.input_shape.as_list()[-1]
        assert input_units < self.units
        # no padding in leftmost dimensions
        paddings = [[0, 0] for _ in range(n_dims-1)]
        paddings.append([0, self.units-input_units])
        return tf.pad(inputs, paddings, "CONSTANT")


class DropLayer(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of output units')

    def call(self, inputs):
        tdl.core.assert_initialized(self, 'call', ['input_shape', 'units'])
        input_units = self.input_shape.as_list()[-1]
        assert input_units > self.units
        return inputs[..., :self.units]


class ConcatConv1x1(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of output units')

    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    @tdl.core.SubmodelInit(lazzy=True)
    def conv(self, use_bias=True):
        tdl.core.assert_initialized(
            self, 'conv', ['units', 'input_shape', 'equalized'])
        if self.equalized:
            Conv1x1 = equalized.Conv1x1Proj
        else:
            Conv1x1 = convnet.Conv1x1Proj
        input_units = self.input_shape.as_list()[-1]
        assert input_units < self.units
        return Conv1x1(units=self.units-input_units, use_bias=use_bias)

    def call(self, inputs):
        return tf.concat([inputs, self.conv(inputs)], -1)


class ResBlock(ResConv2D):
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)

    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=False)

    PROJECTION_TYPE = {
        'linear',
        'pad',
        'drop',
        'concat',
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
                      'downsample': 'nearest',
                      'resize': 'bilinear'}

    @tdl.core.SubmodelInit(lazzy=True)
    def projection(self, units=None,
                   on_increment='pad', on_decrement='drop',
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
        elif output_units > input_units:
            method = on_increment
        elif output_units < input_units:
            method = on_decrement
        # method
        assert method in self.PROJECTION_TYPE
        # equalized
        if self.equalized:
            Conv1x1 = equalized.Conv1x1Proj
        else:
            Conv1x1 = convnet.Conv1x1Proj
        #
        if method == 'linear':
            return Conv1x1(units=output_units, use_bias=self.use_bias)
        elif method == 'pad':
            return ZeroPad(units=output_units)
        elif method == 'drop':
            return DropLayer(units=output_units)
        elif method == 'concat':
            return ConcatConv1x1(
                units=output_units, equalized=self.equalized,
                conv={'use_bias': self.use_bias})

    @tdl.core.SubmodelInit(lazzy=True)
    def residual(self, conv=None, pooling=None, upsample=None,
                 batchnorm='batchnorm', **kargs):
        """Residual model
        Args:
            conv: {filters: list(int), kernels: int | list(int),
                   strides: list(int)},
            pooling: dict{size: int, method: str}
            upsample: dict{method: str, layers: list(int)}
            batchnorm: str | dict{method: str}
            """
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
        if all(oi == ii for ii, oi in zip(input_size, output_size)):
            return None
        elif all(oi >= ii for ii, oi in zip(input_size, output_size)):
            return 'upsample'
        elif all(oi <= ii for ii, oi in zip(input_size, output_size)):
            return 'downsample'
        else:
            return 'resize'

    def call(self, inputs):
        residual = self.residual(inputs)
        resize_type = self._resize_operation()
        assert resize_type in (None, 'upsample', 'downsample')

        if resize_type:
            resize_operation = self._get_resize_operation()
            if resize_type == 'upsample':
                resized = resize_operation(inputs)
                shortcut = self.projection(resized)
            elif resize_type == 'downsample':
                projection = self.projection(inputs)
                resized = resize_operation(projection)
        else:
            shortcut = self.projection(inputs)

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
            conv['batchnorm'] = [False] + [True]*(len(conv['filters'])-1)
            conv['activation'] = [False] + [True]*(len(conv['filters'])-1)

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
            shortcut = self.projection(inputs)

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
    pre_activation = tdl.core.InputArgument.optional(
        'pre_activation', doc="use preactivation in the first stage.",
        default=True)

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

    @tdl.core.InputArgument
    def strides(self, value):
        tdl.core.assert_initialized(self, 'pooling', ['stages'])
        if value is None:
            value = None
        if value is None or isinstance(value, int):
            value = [value if idx == 0 else None
                     for idx in range(self.stages)]
        return value

    @tdl.core.SubmodelInit
    def upsample(self, size=None, method='nearest', **kargs):
        return {'size': size, 'method': method, **kargs}

    @tdl.core.SubmodelInit(lazzy=True)
    def layers(self, **kargs):
        tdl.core.assert_initialized(
            self, 'layers',
            ['stages', 'units', 'kernel_size', 'leaky_rate', 'pooling',
             'upsample', 'dropout', 'batchnorm', 'use_bias', 'equalized',
             'pre_activation', 'strides'])
        layers = list()
        for idx in range(self.stages):
            layers.append(ResBlockPreAct(
                pre_activation={
                    'enable': self.pre_activation if idx == 0 else False},
                residual={
                    'conv': {
                        'filters': [self.units, self.units],
                        'kernels': self.kernel_size,
                        'strides': self.strides[idx]},
                    'leaky_rate': self.leaky_rate,
                    'pooling': self.pooling[idx],
                    'upsample': (self.upsample if idx == 0 else dict()),
                    'dropout': {'rate': self.dropout},
                    },
                batchnorm={'method': self.batchnorm},
                use_bias=self.use_bias,
                equalized=self.equalized,
                **kargs))
        return layers


class ResBottleneck(ResStages):
    """
    https://arxiv.org/pdf/1603.05027.pdf
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
    """
    stages = tdl.core.InputArgument.optional(
        'stages', doc='number of stages.', default=3)

    @tdl.core.SubmodelInit(lazzy=True)
    def layers(self, **kargs):
        tdl.core.assert_initialized(
            self, 'layers',
            ['stages', 'units', 'kernel_size', 'leaky_rate', 'pooling',
             'upsample', 'dropout', 'batchnorm', 'use_bias', 'equalized',
             'pre_activation', 'strides'])
        layers = list()
        for idx in range(self.stages):
            layers.append(ResBlockPreAct(
                pre_activation={
                    'enable': self.pre_activation if idx == 0 else False},
                residual={
                    'conv': {
                        'filters': [self.units//4, self.units//4, self.units],
                        'kernels': [1, self.kernel_size, 1],
                        'strides': self.strides[idx]
                        },
                    'leaky_rate': self.leaky_rate,
                    'pooling': self.pooling[idx],
                    'upsample': (self.upsample if idx == 0 else dict()),
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
    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.',
        default=None)
    leaky_rate = tdl.core.InputArgument.optional(
        'leaky_rate', doc='leaky rate', default=0.2)

    LAYER_TYPE = {'plain': ResStages, 'bottleneck': ResBottleneck}

    @tdl.core.SimpleParameter
    def layer_lib(self, _):
        tdl.core.assert_initialized(self, 'layer_lib', ['equalized'])
        if self.equalized:
            return equalized.get_layer_lib('equalized')
        else:
            return equalized.get_layer_lib('keras')

    @tdl.core.SubmodelInit(lazzy=True)
    def input_layer(self, units=None, kernels=3, enable=True):
        tdl.core.assert_initialized(
            self, 'input_layer', ['use_bias', 'layer_lib', 'hidden'])
        if not enable:
            return None
        if units is None:
            units = self.hidden.layers[0].units
        return self.layer_lib.Conv2D(
            filters=units,
            kernel_size=[kernels, kernels],
            padding='same',
            use_bias=self.use_bias)

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, units, pooling=None, kernels=3,
               dropout=None, batchnorm=None, stages=3, strides=None,
               layer_type="bottleneck", pre_activation=True,
               upsample=None,
               non_local=None,
               **kargs):
        """hidden layers
        Args:
            units: list(int) with n_layers elements.
            pooling: int | None | list(int|None) with n_layers elements.
            kernels: int | list(int) with n_layers elements.
            batchnorm: str | None.
            stages: int,
            layer_type: str (bottleneck | plain)
            pre_activation: bool
        """
        tdl.core.assert_initialized(
            self, 'hidden',
            ['equalized', 'use_bias', 'leaky_rate', 'layer_lib', 'batchnorm'])
        n_layers = len(units)
        # kernels arg
        assert isinstance(kernels, (int, list, tuple))
        if isinstance(kernels, int):
            kernels = [kernels for _ in range(n_layers)]
        assert all(isinstance(ki, int) for ki in kernels)
        # pooling arg
        assert (isinstance(pooling, (int, list, tuple)) or (pooling is None))
        if isinstance(pooling, (int, dict)) or pooling is None:
            pooling = [pooling for _ in range(n_layers)]
        assert all(isinstance(pi, (int, dict)) or (pi is None)
                   for pi in pooling)
        # stages arg
        assert isinstance(stages, (int, list, tuple))
        if isinstance(stages, int):
            stages = [stages for _ in range(n_layers)]
        assert all(isinstance(si, int) for si in stages)
        # strides arg
        assert (isinstance(strides, (int, list, tuple)) or (strides is None))
        if isinstance(strides, (int, dict)) or strides is None:
            strides = [strides for _ in range(n_layers)]
        # upsample arg
        assert (isinstance(upsample, (int, dict, list, tuple))
                or upsample is None)
        if isinstance(upsample, (int, dict)) or upsample is None:
            upsample = [upsample for _ in range(n_layers)]
        if all(isinstance(ui, int) or (ui is None) for ui in upsample):
            upsample = [{'size': ui} for ui in upsample]
        # batchnorm
        if (self.batchnorm is not None) and (batchnorm is None):
            batchnorm = self.batchnorm
        # nonlocal
        if non_local is None:
            non_local = [False for _ in range(n_layers)]
        # build
        model = tdl.stacked.StackedLayers()
        for i in range(len(units)):
            res_block = self.LAYER_TYPE[layer_type](
                stages=stages[i],
                units=units[i],
                kernel_size=kernels[i],
                batchnorm=batchnorm,
                leaky_rate=self.leaky_rate,
                strides=strides[i],
                pooling=pooling[i],
                upsample=upsample[i],
                dropout=dropout,
                use_bias=self.use_bias,
                equalized=self.equalized,
                pre_activation=pre_activation,
                **kargs
                )
            if non_local[i]:
                model.add(tdl.stacked.StackedLayers(
                    layers=[res_block, NonLocalBlock()]
                    ))
            else:
                model.add(res_block)

        return model

    @tdl.core.SubmodelInit(lazzy=True)
    def flatten(self, method='global_maxpool', batchnorm=None, activation=True):
        layers = tdl.stacked.StackedLayers()
        # batchnorm
        if (self.batchnorm is not None) and (batchnorm is None):
            batchnorm = self.batchnorm
        if batchnorm:
            if isinstance(batchnorm, str):
                batchnorm = BATCHNORM_TYPES[batchnorm]
            layers.add(batchnorm())
        # flatten
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
        tdl.core.assert_initialized(
            self, 'dense', ['equalized', 'leaky_rate', 'layer_lib'])
        # units arg
        if not units:
            return None
        if isinstance(units, int):
            units = [units]
        # batchnorm arg
        if isinstance(batchnorm, str):
            batchnorm = BATCHNORM_TYPES[batchnorm]
        # build
        layers = tdl.stacked.StackedLayers()
        for ui in units[:-1]:
            if batchnorm is not None:
                layers.add(batchnorm())
            layers.add(tf_layers.LeakyReLU(self.leaky_rate))
            layers.add(self.layer_lib.Dense(units=ui))
        if batchnorm is not None:
            layers.add(batchnorm())
        layers.add(self.layer_lib.Dense(units=units[-1]))
        return layers

    def internal_shapes(self, input_shape=None):
        tdl.core.assert_initialized(
            self, 'hidden_shapes',
            ['input_shape', 'input_layer', 'hidden', 'flatten', 'dense'])
        if input_shape is None:
            input_shape = self.input_shape

        output_shape = input_shape
        output = list()
        if self.input_layer:
            output_shape = self.input_layer.compute_output_shape(output_shape)
            output.append(output_shape)
        output_shape = self.hidden.compute_output_shape(output_shape)
        output.append(output_shape)
        if self.flatten:
            output_shape = self.flatten.compute_output_shape(output_shape)
            output.append(output_shape)
        if self.dense:
            output_shape = self.dense.compute_output_shape(output_shape)
            output.append(output_shape)
        return output

    def compute_output_shape(self, input_shape):
        return self.internal_shapes(input_shape)[-1]

    def hidden_shapes(self):
        output_shape = self.input_shape
        if self.input_layer:
            output_shape = self.input_layer.compute_output_shape(output_shape)
        output = list()
        for layer in self.hidden.layers:
            output_shape = layer.compute_output_shape(output_shape)
            output.append(output_shape)
        return output

    def call(self, inputs, output='dense'):
        out = inputs
        if self.input_layer:
            out = self.input_layer(out)
        out = self.hidden(out)
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
