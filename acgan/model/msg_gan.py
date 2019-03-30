from .base import (BaseGAN, compute_output_shape, DiscriminatorTrainer)
import typing
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
class MSGProjectionV2(MSGProjection):
    @tdl.core.Submodel
    def projection(self, _):
        tdl.core.assert_initialized(self, 'projection', ['projected_shape'])
        model = tdl.stacked.StackedLayers()
        projected_shape = self.projected_shape.as_list()
        model.add(tf_layers.Conv2DTranspose(
            filters=projected_shape[-1],
            kernel_size=[projected_shape[0], projected_shape[1]],
            strides=(1, 1),
            use_bias=False))
        model.add(tf_layers.LeakyReLU(0.2))
        model.add(tf_layers.Conv2D(
            filters=projected_shape[-1], kernel_size=(3, 3),
            strides=(1, 1), padding='same'))
        model.add(tf_layers.LeakyReLU(0.2))
        return model


@tdl.core.create_init_docstring
class Upsample2D(tdl.core.Layer):
    @tdl.core.InputArgument
    def scale(self, value):
        '''scale factor.'''
        if value is None:
            raise tdl.core.exceptions.ArgumentNotProvided()
        return value

    @tdl.core.InputArgument
    def interpolation(self, value):
        '''Type of interpolation. Either bilinear or neirest (Default) '''
        if value is None:
            value = 'nearest'
        elif value not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('interpolation should be nearest, bilinear or '
                             'bicubic')
        return value

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(
            self, 'compute_output_shape', ['scale'])
        input_shape = input_shape.as_list()
        assert len(input_shape) == 4, 'upsample accepts only 4D tensors'
        output_shape = [input_shape[0], input_shape[1]*self.scale[0],
                        input_shape[2]*self.scale[1], input_shape[3]]
        return tf.TensorShape(output_shape)

    def call(self, inputs):
        input_shape = self.input_shape.as_list()
        size = [input_shape[1] * self.scale[0],
                input_shape[2] * self.scale[1]]
        if self.interpolation == 'nearest':
            output = tf.image.resize_nearest_neighbor(inputs, size=size)
        elif self.interpolation == 'bilinear':
            output = tf.image.resize_bilinear(inputs, size=size)
        elif self.interpolation == 'bicubic':
            output = tf.image.resize_bicubic(inputs, size=size)
        else:
            raise ValueError('interpolation should be nearest, bilinear or '
                             'bicubic')
        return output


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
    def conv(self, kernels=None, padding='same', interpolation='bilinear'):
        if kernels is None:
            kernels = (3, 3)
        tdl.core.assert_initialized(self, 'conv', ['units', 'upsampling'])
        model = tdl.stacked.StackedLayers()
        # model.add(tf_layers.UpSampling2D(
        #    size=tuple(self.upsampling),
        #    interpolation=interpolation))
        model.add(Upsample2D(scale=self.upsampling,
                             interpolation=interpolation))
        model.add(tf_layers.Conv2D(
                self.units, kernel_size=kernels,
                strides=(1, 1), padding=padding,
                use_bias=False))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf_layers.LeakyReLU(0.2))
        model.add(tf_layers.Conv2D(
                self.units, kernel_size=kernels,
                strides=(1, 1), padding=padding,
                use_bias=True))
        model.add(tf.keras.layers.BatchNormalization())
        return model

    # @tdl.core.SubmodelInit
    def conv_old(self, kernels, padding):
        tdl.core.assert_initialized(self, 'conv', ['units', 'upsampling'])
        return tf_layers.Conv2DTranspose(
                self.units, kernel_size=kernels,
                strides=self.upsampling, padding=padding,
                use_bias=False)

    @tdl.core.Submodel
    def activation(self, value):
        if value is None:
            value = tf_layers.LeakyReLU(0.2)
        return value

    def compute_output_shape(self, input_shape):
        chain = [self.conv]
        if self.activation is not None:
            chain.append(self.activation)
        return compute_output_shape(chain, input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        if self.activation is not None:
            output = self.activation(output)
        return output


@tdl.core.create_init_docstring
class MSGOutputGen(MSGHiddenGen):
    @tdl.core.Submodel
    def activation(self, value):
        if value is None:
            value = tf_layers.Activation(tf.keras.activations.tanh)
        return value


@tdl.core.create_init_docstring
class ImagePyramid(tdl.core.Layer):
    @tdl.core.InputArgument
    def scales(self, value: typing.Union[int, typing.List[int]]):
        '''scaling factor.'''
        return value

    @tdl.core.InputArgument
    def depth(self, value: typing.Union[None, int]):
        '''number of scaling operations performed'''
        tdl.core.assert_initialized(self, 'depth', ['scales'])
        if value is None:
            if isinstance(value, int):
                raise ValueError('depth of ImagePyramid not specified')
            value = len(self.scales)
        return value

    @tdl.core.InputArgument
    def interpolation(self, value: typing.Union[None, str]):
        '''Type of interpolation. Either bilinear or neirest (Default) '''
        if value is None:
            value = 'nearest'
        elif value not in ('nearest', 'bilinear', 'bicubic'):
            raise ValueError('interpolation should be nearest, bilinear or '
                             'bicubic')
        return value

    @tdl.core.Submodel
    def _resize_fn(self, _):
        tdl.core.assert_initialized(self, '_resize_fn', ['interpolation'])
        if self.interpolation == 'nearest':
            value = tf.image.resize_nearest_neighbor
        elif self.interpolation == 'bilinear':
            value = tf.image.resize_bilinear
        elif self.interpolation == 'bicubic':
            value = tf.image.resize_bicubic
        return value

    def compute_output_shape(self, input_shape):
        input_shape = input_shape.as_list()
        scales = ([self.scales]*self.depth if isinstance(self.scales, int)
                  else self.scales)
        output_shapes = list()
        current_size = input_shape[1:3]
        for scale in scales:
            if isinstance(scale, int):
                scale = (scale, scale)
            output_shapes.append(tf.TensorShape(
                (input_shape[0],
                 scale[0]*current_size[0], scale[1]*current_size[1],
                 input_shape[-1])
            ))
            current_size = output_shapes[-1].to_list()[1:3]
        return output_shapes

    def call(self, inputs):
        assert self.input_shape.ndims == 4, \
            'provided input is not 4 dimensional'
        scales = ([self.scales]*self.depth if isinstance(self.scales, int)
                  else self.scales)

        output = list([inputs])
        for scale in scales:
            if isinstance(scale, int):
                scale = (scale, scale)
            input_size = tf.convert_to_tensor(inputs).shape.as_list()[1:3]
            size = (input_size[0]//scale[0], input_size[1]//scale[1])
            output.append(self._resize_fn(inputs, size=size))
            inputs = output[-1]
        return output[::-1]


@tdl.core.create_init_docstring
class MSG_GeneratorModel(tdl.stacked.StackedLayers):
    @tdl.core.LazzyProperty
    def embedding_size(self):
        tdl.core.assert_initialized(self, 'embedding_size', ['input_shape'])
        return self.input_shape[-1].value

    @tdl.core.LazzyProperty
    def pyramid_shapes(self):
        tdl.core.assert_initialized(
            self, 'pyramid_shapes', ['embedding_size', 'layers'])
        _input_shape = tf.TensorShape([None, self.embedding_size])
        hidden_shapes = list()
        for layer in self.layers:
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(_input_shape)
        return hidden_shapes

    @tdl.core.Submodel
    def projections(self, _):
        tdl.core.assert_initialized(
            self, 'projections', ['layers', 'pyramid_shapes'])
        output_channels = self.pyramid_shapes[-1][-1].value
        projections = [
            tdl.stacked.StackedLayers(layers=[
                tf_layers.Conv2D(
                    output_channels, kernel_size=(1, 1),
                    strides=(1, 1), padding='same',
                    use_bias=False),
                tf_layers.Activation(tf.keras.activations.tanh)])
            for i in range(len(self.pyramid_shapes)-1)]
        return projections

    def pyramid(self, inputs):
        if self.built is False:
            self.build(input.shape)
        pyramid = list()
        hidden = inputs
        for layer, projection in zip(self.layers[:-1], self.projections):
            hidden = layer(hidden)
            pyramid.append(projection(hidden))
        pyramid.append(self.layers[-1](hidden))
        self._update_variables()
        return pyramid


@tdl.core.create_init_docstring
class MSG_GAN(BaseGAN):
    GeneratorBaseModel = MSG_GeneratorModel
    InputProjection = MSGProjectionV2
    HiddenGenLayer = MSGHiddenGen
    OutputGenLayer = MSGOutputGen

    @tdl.core.SubmodelInit
    def pyramid(self, interpolation='bilinear'):
        tdl.core.assert_initialized(
            self, 'pyramid', ['target_shape', 'generator'])
        target_size = self.target_shape[0:2].as_list()
        _input_shape = tf.TensorShape([None, self.embedding_size])
        hidden_shapes = list()
        for layer in self.generator.layers:
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(_input_shape)
        hidden_sizes = [shape.as_list()[1:3] for shape in hidden_shapes[::-1]]
        assert hidden_sizes[0] == target_size,\
            'generator and target sizes do not match'

        scaling = list()
        _input_size = hidden_sizes[0]
        for target_size in hidden_sizes[1:]:
            scale = [_input_size[0]//target_size[0],
                     _input_size[1]//target_size[1]]
            assert target_size[0]*scale[0] == _input_size[0],\
                'scales of image pyramid and generator do not match'
            assert target_size[1]*scale[1] == _input_size[1],\
                'scales of image pyramid and generator do not match'
            scaling.append(scale)
            _input_size = target_size

        return ImagePyramid(scales=scaling)

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None,
                              learning_rate=0.0002):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid'])
        return MSG_DiscriminatorTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer={'learning_rate': learning_rate})


class MSG_DiscriminatorTrainer(DiscriminatorTrainer):
    @tdl.core.OutputValue
    def pyramid(self, _):
        tdl.core.assert_initialized(self, 'pyramid', ['xreal'])
        return self.model.pyramid(self.xreal)
