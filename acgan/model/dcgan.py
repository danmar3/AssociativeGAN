import operator
import functools
import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
from .base import (BaseGAN, TransposeLayer, BatchNormalization,
                   _compute_output_shape)
from .msg_gan import MSGProjection


@tdl.core.create_init_docstring
class DCGANHiddenGen(TransposeLayer, BatchNormalization):
    @tdl.core.Submodel
    def activation(self, value):
        if value is None:
            value = tf_layers.LeakyReLU(0.2)
        return value

    def compute_output_shape(self, input_shape=None):
        chain = [self.conv, self.batch_normalization, self.activation]
        return _compute_output_shape(chain, input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        if self.batch_normalization is not None:
            output = self.batch_normalization(output)
        if self.activation is not None:
            output = self.activation(output)
        return output


@tdl.core.create_init_docstring
class DCGANOutputGen(TransposeLayer):
    @tdl.core.Submodel
    def activation(self, value):
        if value is None:
            value = tf_layers.Activation(tf.keras.activations.tanh)
        return value


@tdl.core.create_init_docstring
class DCGANProjection(tdl.stacked.StackedLayers):
    @tdl.core.InputArgument
    def projected_shape(self, value):
        value = tf.TensorShape(value)
        return value

    @tdl.core.Submodel
    def layers(self, _):
        tdl.core.assert_initialized(self, 'projection', ['projected_shape'])
        units = functools.reduce(
            operator.mul, self.projected_shape.as_list(), 1)
        layers = [tdl.dense.LinearLayer(units=units),
                  tf_layers.Reshape(self.projected_shape),
                  tf.keras.layers.BatchNormalization(),
                  tf_layers.LeakyReLU(0.2)]
        return layers


@tdl.core.create_init_docstring
class DCGAN(BaseGAN):
    InputProjection = MSGProjection
    HiddenGenLayer = DCGANHiddenGen
    OutputGenLayer = DCGANOutputGen