from types import SimpleNamespace

import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
from twodlearn.resnet import convnet as tdl_convnet

from . import equalized


class Discriminator(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of units in each hidden block of the network')

    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.',
        default='pixelwise')
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)
    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=True)

    @tdl.core.SimpleParameter
    def layer_lib(self, _):
        tdl.core.assert_initialized(self, 'layer_lib', ['equalized'])
        if self.equalized:
            return equalized.get_layer_lib('equalized')
        else:
            return equalized.get_layer_lib('keras')

    @tdl.core.LazzyProperty
    def hidden_shapes(self):
        tdl.core.assert_initialized(
            self, 'hidden_shapes', ['input_shape', 'projections'])
        pyramid_shape = self.input_shape
        input_shapes = pyramid_shape[::-1]
        output_list = list()
        proj_shape = self.projections[0].compute_output_shape(input_shapes[0])
        output_shape = self.hidden[0].compute_output_shape(proj_shape)
        output_list.append(output_shape)
        for x_shape, projection, layer in\
                zip(input_shapes[1:], self.projections[1:], self.hidden[1:]):
            proj_shape = projection.compute_output_shape(x_shape)
            extended = proj_shape[:-1].concatenate(
                proj_shape[-1].value + output_shape[-1].value)
            output_shape = layer.compute_output_shape(extended)
            output_list.append(output_shape)
        return output_list

    @tdl.core.SubmodelInit(lazzy=True)
    def projections(self, init_units=None, activation=None):
        tdl.core.assert_initialized(
            self, 'projections',
            ['units', 'hidden', 'input_shape', 'use_bias', 'layer_lib'])
        projections = list()
        if init_units is None:
            init_units = self.units[0]//2
        units_list = [init_units] + [ui for ui in self.units]
        for units in units_list:
            projections.append(
                self.layer_lib.Conv1x1Proj(
                    units=units,
                    activation=activation,
                    use_bias=self.use_bias
                ))
        return projections

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, stages=3, batchnorm=None, dropout=None,
               kernel_size=3, pooling=2,
               leaky_rate=0.2):
        tdl.core.assert_initialized(
            self, 'hidden', ['units', 'use_bias', 'equalized'])
        n_layers = len(self.units)
        # kernels arg
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernels = [kernel_size for _ in range(n_layers)]
        assert all(isinstance(ki, int) for ki in kernels)
        # pooling arg
        assert (isinstance(pooling, (int, list, tuple)) or (pooling is None))
        if isinstance(pooling, (int, dict)):
            pooling = [pooling for _ in range(n_layers)]
        assert all(isinstance(pi, (int, dict)) or (pi is None)
                   for pi in pooling)

        layers = list()
        for i in range(len(self.units)):
            layers.append(ResBottleneck(
                stages=stages,
                units=self.units[i],
                kernel_size=kernels[i],
                batchnorm=batchnorm,
                leaky_rate=leaky_rate,
                pooling=pooling[i],
                dropout=dropout,
                use_bias=self.use_bias,
                equalized=self.equalized
                ))
        layers.append(MinibatchStddev())
        layers.append(ResBottleneck(
            stages=stages,
            units=self.units[-1],
            kernel_size=3,
            batchnorm=batchnorm,
            leaky_rate=leaky_rate,
            pooling=None,
            dropout=dropout,
            use_bias=self.use_bias,
            equalized=self.equalized
            ))
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def flatten(self, method='global_maxpool'):
        layers = tdl.stacked.StackedLayers()
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
    def dense(self, **kargs):
        return tdl.stacked.StackedLayers(layers=[
            self.layer_lib.Dense(units=1)
            ])

    def call(self, inputs, output='logits'):
        '''the call expects a full pyramid.'''
        tdl.core.assert_initialized(self, 'call', ['projections'])
        depth = len(inputs)
        inputs = inputs[::-1]
        assert depth == len(self.projections),\
            'pyramid size does not match the number of projections'
        proj = self.projections[0](inputs[0])
        out = self.hidden[0](proj)
        for x_i, projection, layer in\
                zip(inputs[1:], self.projections[1:], self.hidden[1:]):
            proj = projection(x_i)
            extended = tf.concat([proj, out], axis=-1)
            out = layer(extended)
        if output == 'hidden':
            return out
        if self.flatten:
            out = self.flatten(out)
        if output == 'flatten':
            return out
        elif output == 'logits':
            return self.dense(out)
        elif output == 'prob':
            return tf.nn.sigmoid(self.dense(out))
        else:
            raise ValueError(f'output {output} not a valid option.')


class NewAxis(tdl.core.Layer):
    @tdl.core.InputArgument
    def axis(self, value):
        if value is None:
            raise ValueError('axis must be specified')
        assert isinstance(value, (list, tuple))
        assert all(ai in (..., tf.newaxis) for ai in value)
        return value

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(self, 'compute_output_shape', ['axis'])
        output_shape = list()
        for ai in self.axis:
            if ai is tf.newshape:
                output_shape.append(1)
            elif ai is ...:
                output_shape = output_shape + input_shape.as_list()
            else:
                raise ValueError('')
        return tf.TensorShape(output_shape)

    def call(self, inputs):
        return inputs[self.axis]


class TransposeLayer(tdl.core.Layer):
    @tdl.core.InputArgument
    def axis(self, value):
        if value is None:
            raise ValueError('axes must be specified')
        assert isinstance(value, (list, tuple))
        assert all(isinstance(ai, int) for ai in value)
        return value

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(self, 'compute_output_shape', ['axis'])
        return tf.TensorShape(input_shape[ai] for ai in self.axis)

    def call(self, inputs):
        return tf.transpose(inputs, self.axis)


class Generator(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of units in each hidden block of the network')

    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.',
        default='pixelwise')
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)
    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=True)
    output_activation = tdl.core.InputArgument.optional(
        'output_activation', doc="activation function at the output.",
        default=None)
    output_channels = tdl.core.InputArgument.optional(
        'output_channels', doc="number of output channels"
    )

    @tdl.core.SimpleParameter
    def layer_lib(self, _):
        tdl.core.assert_initialized(self, 'layer_lib', ['equalized'])
        if self.equalized:
            return equalized.get_layer_lib('equalized')
        else:
            return equalized.get_layer_lib('keras')

    @tdl.core.InputArgument
    def embedding_size(self, value: typing.Union[int, None]):
        if value is None:
            tdl.core.assert_initialized(
                self, 'embedding_size', ['input_shape'])
            value = self.input_shape.as_list()[-1]
        return value

    def hidden_shapes(self):
        tdl.core.assert_initialized(
            self, 'hidden_shapes', ['embedding_size', 'layers'])
        _input_shape = tf.TensorShape([None, self.embedding_size])
        hidden_shapes = list()
        for layer in self.layers[:-1]:
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(_input_shape)
        return hidden_shapes

    def output_shape(self):
        return self.projections[-1].compute_output_shape(
            self.hidden_shapes()[-1])

    def pyramid_shapes(self):
        tdl.core.assert_initialized(
            self, 'pyramid_shapes', ['projections'])
        shapes = list()
        for proj, hidden_shape in zip(self.projections, self.hidden_shapes()):
            shapes.append(proj.compute_output_shape(hidden_shape))
        return shapes

    @tdl.core.Submodel
    def projections(self, value):
        if value is not None:
            return value
        tdl.core.assert_initialized(
            self, 'projections',
            ['output_activation', 'output_channels', 'use_bias', 'hidden',
             'layer_lib'])
        projections = [
            self.layer_lib.Conv1x1Proj(
                units=self.output_channels,
                activation=(self.output_activation() if self.output_activation
                            else None),
                use_bias=self.use_bias)
            for i in range(len(self.hidden))]
        return projections

    @tdl.core.SubmodelInit(lazzy=True)
    def input_layer(self, target_shape, batchnorm=None, kernel_size=3):
        tdl.core.assert_initialized(
            self, 'input_layer', ['layer_lib', 'use_bias'])
        # batchnorm
        if self.batchnorm is not None:
            batchnorm = self.batchnorm
        # project
        layers = tdl.stacked.StackedLayers()
        layers.add(NewAxis(axis=[..., tf.newaxis, tf.newaxis]))
        layers.add(TransposeLayer(axis=[0, 3, 2, 1]))

        target_shape = target_shape.as_list()
        layers.add(self.layer_lib.Conv2DTranspose(
            filters=target_shape[-1],
            kernel_size=[target_shape[0], target_shape[1]]
        ))
        if batchnorm:
            layers.add(batchnorm())
        # conv
        layers.add(self.layer_lib.Conv2D(
            filters=target_shape[-1],
            kernel_size=kernel_size,
            padding='same',
            use_bias=self.use_bias
        ))
        # layers.add(tf_layers.LeakyReLU(self.leaky_rate))
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, stages=3, batchnorm=None, dropout=None,
               kernel_size=3, upsample=2,
               leaky_rate=0.2):
        tdl.core.assert_initialized(
            self, 'hidden', ['units', 'use_bias', 'equalized'])
        n_layers = len(self.units)
        # kernels arg
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernels = [kernel_size for _ in range(n_layers)]
        assert all(isinstance(ki, int) for ki in kernels)
        # upsample arg
        assert (isinstance(upsample, (int, list, tuple)) or (upsample is None))
        if isinstance(upsample, (int, dict)):
            upsample = [upsample for _ in range(n_layers)]
        assert all(isinstance(pi, (int, dict)) or (pi is None)
                   for pi in upsample)
        # batchnorm
        if self.batchnorm is not None:
            batchnorm = self.batchnorm

        layers = tdl.stacked.StackedLayers()
        for i in range(len(self.units)):
            layers.add(ResBottleneck(
                stages=stages,
                units=self.units[i],
                kernel_size=kernels[i],
                batchnorm=batchnorm,
                leaky_rate=leaky_rate,
                upsample=upsample[i],
                dropout=dropout,
                use_bias=self.use_bias,
                equalized=self.equalized
                ))
        return layers

    def call(self, inputs, output='image'):
        '''the call expects a full pyramid.'''
        out = inputs
        if self.input_layer:
            out = self.input_layer(out)
        hidden = list()
        for layer_i in self.hidden.layers:
            out = layer_i(out)
            hidden.append(out)

        if output == 'hidden':
            return hidden
        elif output == 'image':
            return self.projections[-1](out)
        elif output == 'pyramid':
            return [pi(hi) for hi, pi in zip(hidden, self.projections)]

    def pyramid(self, inputs):
        return self(inputs, output='pyramid')
