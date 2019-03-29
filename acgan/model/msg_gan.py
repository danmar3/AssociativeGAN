import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers


@tdl.core.create_init_docstring
class MSGProjection(tdl.core.Layer):
    @tdl.core.InputArgument
    def projected_shape(self, value):
        value = tf.TensorShape(value)
        return value

    @tdl.core.Submodel
    def projection(self, _):
        tdl.core.assert_initialized(self, 'projection', ['projected_shape'])
        model = tdl.stacked.StackedLayers()
        projected_shape = self.projected_shape.as_list()
        model.add(tf_layers.Conv2DTranspose(
            projected_shape[-1],
            kernel_size=[projected_shape[0], projected_shape[1]],
            strides=(1, 1),
            use_bias=False))
        model.add(tf_layers.LeakyReLU(0.2))
        return model

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape([input_shape[0], 1, 1, input_shape[-1]])
        return self.projection.compute_output_shape(input_shape)

    def call(self, inputs):
        assert inputs.shape.ndims == 2, \
            'input to MSG projection should be batched vectors'
        inputs = tf.transpose(inputs[..., tf.newaxis, tf.newaxis],
                              [0, 2, 3, 1])
        return self.projection(inputs)


@tdl.core.create_init_docstring
class MSGProjectionV2(tdl.core.Layer):
    @tdl.core.Submodel
    def projection(self, _):
        tdl.core.assert_initialized(self, 'projection', ['projected_shape'])
        model = tdl.stacked.StackedLayers()
        projected_shape = self.projected_shape.as_list()
        model.add(tf_layers.Conv2DTranspose(
            projected_shape[-1],
            kernel_size=[projected_shape[0], projected_shape[1]],
            strides=(1, 1),
            use_bias=False))
        model.add(tf_layers.LeakyReLU(0.2))
        model.add(tf_layers.Conv2D(
            units=projected_shape[-1], kernel_size=(3, 3),
            strides=(1, 1), padding='same'))
        model.add(tf_layers.LeakyReLU(0.2))
        return model


@tdl.core.create_init_docstring
class MSGHiddenGen(tdl.core.Layer):
    @tdl.core.InputArgument
    def units(self, value):
        '''Number of output filters.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided()
        return value

    @tdl.core.InputArgument
    def upsampling(self, value):
        '''Upsampling ratio.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided()
        return value

    @tdl.core.SubmodelInit
    def conv(self, kernels, padding='same'):
        tdl.core.assert_initialized(self, 'conv', ['units', 'upsampling'])
        return tf_layers.Conv2DTranspose(
                self.units, kernel_size=kernels,
                strides=(1, 1), padding=padding,
                use_bias=False)
