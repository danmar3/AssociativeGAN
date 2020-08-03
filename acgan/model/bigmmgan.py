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
            layers.append(ResStages101(
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
        layers.append(ResStages101(
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


class Generator(tdl.core.Layer):
    pass
