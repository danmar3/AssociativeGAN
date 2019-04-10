from .base import (BaseGAN, compute_output_shape, DiscriminatorTrainer,
                   GeneratorTrainer, MinibatchStddev, VectorNormalizer)
import typing
import operator
import functools
import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers


USE_BIAS = {'generator': True, 'discriminator': True}
LEAKY_RATE = 0.2
GeneratorFeatureNorm = VectorNormalizer  # tf.keras.layers.BatchNormalization()


def OutputActivation():
    # return tf_layers.Activation(tf.keras.activations.tanh)
    return None


def regularizer_fn(weights):
    loss = tf.reduce_mean(tf.abs(weights))
    size = functools.reduce(
        operator.mul, weights.shape[-2:].as_list(), 1)
    # size = max(weights.shape[-2:].as_list())
    loss = loss*size
    # if weight.shape.ndims == 4:
    #     shape = weight.shape.as_list()
    #     loss = loss/(shape[0] * shape[1])
    return loss


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
            initializer = tf.keras.initializers.RandomNormal(stddev=1.0)
        weight = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.kernel_size[0], self.kernel_size[1],
                   self.input_shape[-1].value, self.filters],
            trainable=trainable,
            **kargs)
        fan_in, fan_out = tdl.core.initializers.compute_fans(weight.shape)
        return weight * tf.sqrt(2.0/fan_in.value)


class Conv1x1Proj(tdl.convnet.Conv1x1Proj):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['units', 'input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.RandomNormal(stddev=1.0)
        kernel = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.input_shape[-1].value, self.units],
            trainable=trainable,
            **kargs)
        fan_in, fan_out = tdl.core.initializers.compute_fans(kernel.shape)
        return kernel * tf.sqrt(2.0/fan_in.value)


class Conv2DTranspose(tdl.convnet.Conv2DTranspose):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['kernel_size', 'input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.RandomNormal(stddev=1.0)
        kernel = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.kernel_size[0], self.kernel_size[1],
                   self.filters, self.input_shape[-1].value],
            trainable=trainable,
            **kargs)
        # see https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py
        fan_in = max(self.kernel_size)
        return kernel * tf.sqrt(2.0/fan_in)


@tdl.core.create_init_docstring
class AffineLayer(tdl.dense.AffineLayer):
    @tdl.core.ParameterInit
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(
            self, 'kernel', ['units', 'input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.RandomNormal(stddev=1.0)
        kernel = self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.input_shape[-1].value, self.units],
            trainable=trainable,
            **kargs)
        fan_in, fan_out = tdl.core.initializers.compute_fans(kernel.shape)
        return kernel * tf.sqrt(2.0/fan_in.value)


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
        model.add(Conv2DTranspose(
            filters=projected_shape[-1],
            kernel_size=[projected_shape[0], projected_shape[1]],
            strides=(1, 1),
            use_bias=USE_BIAS['generator']))
        model.add(tf_layers.LeakyReLU(LEAKY_RATE))
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
        model.add(Conv2DTranspose(
            filters=projected_shape[-1],
            kernel_size=[projected_shape[0], projected_shape[1]],
            strides=[1, 1],
            use_bias=USE_BIAS['generator']))
        model.add(tf_layers.LeakyReLU(LEAKY_RATE))
        if GeneratorFeatureNorm is not None:
            model.add(GeneratorFeatureNorm())
        model.add(Conv2DLayer(
            filters=projected_shape[-1],
            kernel_size=[3, 3],
            strides=[1, 1], padding='same',
            use_bias=USE_BIAS['generator']))
        model.add(tf_layers.LeakyReLU(LEAKY_RATE))
        if GeneratorFeatureNorm is not None:
            model.add(GeneratorFeatureNorm())
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
class MSG_GeneratorHidden(tdl.core.Layer):
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
    def conv(self, kernels=None, padding='same', interpolation='nearest'):
        if kernels is None:
            kernels = (3, 3)
        tdl.core.assert_initialized(self, 'conv', ['units', 'upsampling'])
        model = tdl.stacked.StackedLayers()
        model.add(Upsample2D(scale=self.upsampling,
                             interpolation=interpolation))
        model.add(Conv2DLayer(
                filters=self.units, kernel_size=list(kernels),
                strides=[1, 1], padding=padding,
                use_bias=USE_BIAS['generator']))
        model.add(tf_layers.LeakyReLU(LEAKY_RATE))
        if GeneratorFeatureNorm is not None:
            model.add(GeneratorFeatureNorm())
        model.add(Conv2DLayer(
                filters=self.units, kernel_size=list(kernels),
                strides=[1, 1], padding=padding,
                use_bias=USE_BIAS['generator']))
        return model

    @tdl.core.Submodel
    def activation(self, value):
        model = tdl.stacked.StackedLayers()
        if value is None:
            model.add(tf_layers.LeakyReLU(LEAKY_RATE))
        if GeneratorFeatureNorm is not None:
            model.add(GeneratorFeatureNorm())
        return model

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
class MSG_GeneratorOutput(Conv1x1Proj):
    @tdl.core.Submodel
    def activation(self, value):
        if value is None:
            value = OutputActivation()
        return value

    @tdl.core.ParameterInit
    def bias(self, initializer=None, trainable=True, **kargs):
        if not USE_BIAS:
            return None
        tdl.core.assert_initialized(self, 'bias', ['units'])
        if initializer is None:
            initializer = tf.keras.initializers.zeros()
        return self.add_weight(
            name='bias',
            initializer=initializer,
            shape=[self.units],
            trainable=trainable,
            **kargs)


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
    @tdl.core.InputArgument
    def input_shape(self, value):
        if value is None:
            tdl.core.assert_initialized_if_available(
                self, 'input_shape', ['embedding_size'])
            if tdl.core.is_property_initialized(self, 'embedding_size'):
                value = tf.TensorShape([None, self.embedding_size])
            else:
                raise tdl.core.exceptions.ArgumentNotProvided(self)
        return tf.TensorShape(value)

    @tdl.core.InputArgument
    def embedding_size(self, value: typing.Union[int, None]):
        if value is None:
            tdl.core.assert_initialized(
                self, 'embedding_size', ['input_shape'])
            value = self.input_shape[-1].value
        return value

    @tdl.core.LazzyProperty
    def hidden_shapes(self):
        tdl.core.assert_initialized(
            self, 'hidden_shapes', ['embedding_size', 'layers'])
        _input_shape = tf.TensorShape([None, self.embedding_size])
        hidden_shapes = list()
        for layer in self.layers[:-1]:
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(_input_shape)
        return hidden_shapes

    @tdl.core.LazzyProperty
    def output_shape(self):
        tdl.core.assert_initialized(self, 'output_shape', ['hidden_shapes'])
        return self.layers[-1].compute_output_shape(self.hidden_shapes[-1])

    @tdl.core.LazzyProperty
    def pyramid_shapes(self):
        tdl.core.assert_initialized(
            self, 'pyramid_shapes', ['hidden_shapes', 'projections',
                                     'output_shape'])
        shapes = list()
        for proj, hidden_shape in zip(self.projections, self.hidden_shapes):
            shapes.append(proj.compute_output_shape(hidden_shape))
        # shapes.append(self.output_shape)
        return shapes

    @tdl.core.Submodel
    def projections(self, value):
        if value is not None:
            return value
        tdl.core.assert_initialized(self, 'projections', ['layers'])
        kargs = [{} for i in range(len(self.hidden_shapes))]
        if tdl.core.any_initialized(self, ['input_shape', 'embedding_size']):
            tdl.core.assert_initialized(self, 'projections', ['hidden_shapes'])
            for k_i, shape in zip(kargs, self.hidden_shapes):
                k_i['input_shape'] = shape
            # set the input_shape of last layer
            self.layers[-1].input_shape = self.hidden_shapes[-1]

        output_channels = self.layers[-1].units
        projections = [
            Conv1x1Proj(
                units=output_channels,
                activation=OutputActivation(),
                use_bias=USE_BIAS['generator'],
                **kargs[i])
            for i in range(len(self.layers)-2)]
        # output layer is a projection
        projections.append(self.layers[-1])
        return projections

    def pyramid(self, inputs):
        if self.built is False:
            self.build(inputs.shape)
        pyramid = list()
        hidden = inputs
        for layer, projection in zip(self.layers[:-1], self.projections):
            hidden = layer(hidden)
            pyramid.append(projection(hidden))
        self._update_variables()
        return pyramid


class MSG_DiscriminatorModel(tdl.stacked.StackedLayers):
    @tdl.core.LazzyProperty
    def hidden_shapes(self):
        tdl.core.assert_initialized(self, 'hidden_shapes', ['input_shape'])
        _input_shape = self.input_shape
        hidden_shapes = list()
        for layer in self.layers[:-1]:
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(_input_shape)
        return hidden_shapes

    @tdl.core.SubmodelInit
    def projections(self, init_units=None, activation=None):
        tdl.core.assert_initialized(
            self, 'projections', ['layers', 'input_shape'])
        projections = list()
        if init_units is None:
            init_units = self.layers[0].units//2
        units_list = [init_units] + [layer.units for layer in self.layers[:-1]]
        for units in units_list:
            projections.append(
                Conv1x1Proj(
                    units=units,
                    activation=activation,
                    use_bias=USE_BIAS['discriminator']
                ))
        return projections

    @staticmethod
    def _size_matches(shape1, shape2):
        '''check if the shapes match one another'''
        return shape1[1:3].as_list() == shape2[1:3].as_list()

    @staticmethod
    def _eval_chain(chain, inputs):
        hidden = inputs
        for layer in chain:
            hidden = layer(hidden)
        return hidden

    def call(self, inputs, depth=None, **kargs):
        '''the call handles inputs of any size in the pyramid.'''
        tdl.core.assert_initialized(self, 'call', ['projections'])
        if depth is not None:
            # TODO(daniel): implement depth?
            pass
        if self._size_matches(inputs.shape, self.input_shape):
            chain = [self.projections[0]] + self.layers
            return self._eval_chain(chain, inputs)
        for idx, (shape, proj) in enumerate(
                zip(self.hidden_shapes, self.projections[1:])):
            if self._size_matches(inputs.shape, shape):
                chain = [proj] + self.layers[idx+1:]
                return self._eval_chain(chain, inputs)
        # raise error if no size matches
        raise ValueError('invalid input shape {}'.format(inputs.shape))


class MSG_GeneratorTrainer(GeneratorTrainer):
    @tdl.core.OutputValue
    def sim_pyramid(self, _):
        tdl.core.assert_initialized(self, 'sim_pyramid', ['embedding'])
        return self.generator.pyramid(self.embedding)

    @tdl.core.OutputValue
    def xsim(self, _):
        tdl.core.assert_initialized(self, 'xsim', ['sim_pyramid'])
        return self.sim_pyramid[-1]

    @tdl.core.SubmodelInit
    def regularizer(self, scale=None):
        if scale is None:
            return None
        return tdl.core.SimpleNamespace(scale=scale, fn=regularizer_fn)

    @tdl.core.SubmodelInit
    def pyramid_loss(self, scale=None):
        def loss_fn():
            mirror = self.model.pyramid(self.xsim)
            value = tf.add_n([tf.nn.l2_loss(sim_i - mirror_i)
                              for sim_i, mirror_i
                              in zip(self.sim_pyramid, mirror)])
            return value
        return (None if scale is None
                else tdl.core.SimpleNamespace(fn=loss_fn, scale=scale))

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['batch_size', 'xsim', 'sim_pyramid', 'regularizer',
                           'pyramid_loss'])
        losses = list()
        for xsim in self.sim_pyramid[::-1]:
            pred = self.discriminator(xsim, training=True)
            loss_i = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(pred), pred)
            losses.append(loss_i)
        loss = tf.add_n(losses)/len(losses)
        # regularizer
        if self.regularizer is not None:
            layers = tdl.core.find_instances(
                self.generator,
                (tf.keras.layers.Conv2D, tdl.convnet.Conv2DLayer,
                 tdl.convnet.Conv1x1Proj))
            loss += self.regularizer.scale*tf.add_n(
                [self.regularizer.fn(layer.kernel)
                 for layer in layers]
            )
        if self.pyramid_loss is not None:
            loss += self.pyramid_loss.scale * self.pyramid_loss.fn()
        return loss


class MSG_DiscriminatorHidden(tdl.stacked.StackedLayers):
    @tdl.core.Submodel
    def layers(self, _):
        value = [Conv2DLayer(
                    filters=self.units,
                    kernel_size=list(self.kernels), strides=[1, 1],
                    padding=self.padding,
                    use_bias=USE_BIAS['discriminator']),
                 # tf.keras.layers.BatchNormalization(),
                 tf_layers.LeakyReLU(LEAKY_RATE),
                 Conv2DLayer(
                    filters=self.units,
                    kernel_size=list(self.kernels), strides=[1, 1],
                    padding=self.padding,
                    use_bias=USE_BIAS['discriminator']),
                 # tf.keras.layers.BatchNormalization(),
                 tf_layers.LeakyReLU(LEAKY_RATE),
                 tf.keras.layers.AveragePooling2D(pool_size=self.strides)
                 ]
        if self.dropout is not None:
            value.append(tf_layers.Dropout(self.dropout))
        return value

    def __init__(self, units, kernels, strides, padding, dropout, **kargs):
        self.units = units
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.dropout = dropout
        super(MSG_DiscriminatorHidden, self).__init__(**kargs)


class MSG_DiscriminatorOutput(tdl.stacked.StackedLayers):
    @tdl.core.Submodel
    def layers(self, _):
        value = [
            MinibatchStddev(),
            Conv2DLayer(kernel_size=[3, 3], padding='same',
                        use_bias=USE_BIAS['discriminator']),
            tf_layers.LeakyReLU(LEAKY_RATE),
            Conv2DLayer(kernel_size=[4, 4], padding='same',
                        use_bias=USE_BIAS['discriminator']),
            tf_layers.LeakyReLU(LEAKY_RATE),
            tf_layers.Flatten(),
            AffineLayer(units=1),
            tf_layers.Activation(tf.keras.activations.sigmoid)]
        return value


class MSG_DiscriminatorTrainer(DiscriminatorTrainer):
    @tdl.core.OutputValue
    def real_pyramid(self, _):
        tdl.core.assert_initialized(self, 'real_pyramid', ['xreal'])
        return self.model.pyramid(self.xreal)

    @tdl.core.OutputValue
    def sim_pyramid(self, _):
        tdl.core.assert_initialized(self, 'sim_pyramid', ['embedding'])
        return self.generator.pyramid(self.embedding)

    @tdl.core.SubmodelInit
    def regularizer(self, scale=None):
        if scale is None:
            return None
        return tdl.core.SimpleNamespace(scale=scale, fn=regularizer_fn)

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['real_pyramid', 'sim_pyramid', 'regularizer'])
        # real losses
        real_losses = list()
        for xreal in self.real_pyramid:
            pred_real = self.discriminator(xreal, training=True)
            loss_i = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(pred_real), pred_real)
            real_losses.append(loss_i)
        # sim losses
        sim_losses = list()
        for xsim in self.sim_pyramid:
            pred_sim = self.discriminator(xsim, training=True)
            loss_i = tf.keras.losses.BinaryCrossentropy()(
                tf.zeros_like(pred_sim), pred_sim)
            sim_losses.append(loss_i)
        losses = real_losses + sim_losses
        loss = tf.add_n(losses)/len(losses)
        # Regularizer
        if self.regularizer is not None:
            layers = tdl.core.find_instances(
                self.discriminator,
                (tf.keras.layers.Conv2D, tdl.convnet.Conv2DLayer,
                 tdl.convnet.Conv1x1Proj))
            loss += self.regularizer.scale*tf.add_n(
                [self.regularizer.fn(layer.kernel)
                 for layer in layers]
            )
        return loss


@tdl.core.create_init_docstring
class MSG_GAN(BaseGAN):
    GeneratorBaseModel = MSG_GeneratorModel
    InputProjection = MSGProjectionV2
    GeneratorHidden = MSG_GeneratorHidden
    GeneratorOutput = MSG_GeneratorOutput
    GeneratorTrainer = MSG_GeneratorTrainer

    def DiscriminatorBaseModel(self, **kargs):
        tdl.core.assert_initialized(self, 'discriminator', ['generator'])
        tdl.core.assert_initialized(
            self.generator, 'discriminator', ['projections'])
        projections = self.generator.projections
        return MSG_DiscriminatorModel(
            projections=[proj.get_transpose(
                            trainable=False,
                            activation=tf_layers.LeakyReLU(LEAKY_RATE))
                         for proj in projections[::-1]])
    DiscriminatorHidden = MSG_DiscriminatorHidden
    DiscriminatorTrainer = MSG_DiscriminatorTrainer

    @tdl.core.SubmodelInit
    def pyramid(self, interpolation='bilinear'):
        '''layer that creates an image pyramid by performing several
           sub-sampling operations.
        '''
        tdl.core.assert_initialized(
            self, 'pyramid', ['target_shape', 'generator'])
        target_size = self.target_shape[0:2].as_list()
        _input_shape = tf.TensorShape([None, self.embedding_size])
        hidden_shapes = list()
        for layer, proj in zip(self.generator.layers[:-1],
                               self.generator.projections):
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(proj.compute_output_shape(_input_shape))
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

        return ImagePyramid(scales=scaling, interpolation=interpolation)

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None,
                              optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.DiscriminatorTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer=optimizer,
            **kwargs)

    def generator_trainer(self, batch_size, optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'generator_trainer',
            ['generator', 'discriminator', 'pyramid'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.GeneratorTrainer(
            model=self, batch_size=batch_size,
            optimizer=optimizer,
            **kwargs)


class MSG_DiscriminatorModel_V2(MSG_DiscriminatorModel):
    @tdl.core.LazzyProperty
    def hidden_shapes(self):
        tdl.core.assert_initialized(self, 'hidden_shapes', ['input_shape'])
        pyramid_shape = self.input_shape
        input_shapes = pyramid_shape[::-1]
        output_list = list()
        proj_shape = self.projections[0].compute_output_shape(input_shapes[0])
        output_shape = self.layers[0].compute_output_shape(proj_shape)
        output_list.append(output_shape)
        for x_shape, projection, layer in\
                zip(input_shapes[1:], self.projections[1:], self.layers[1:]):
            proj_shape = projection.compute_output_shape(x_shape)
            extended = proj_shape[:-1].concatenate(
                proj_shape[-1].value + output_shape[-1].value)
            output_shape = layer.compute_output_shape(extended)
            output_list.append(output_shape)
        return output_list

    def call(self, inputs, **kargs):
        '''the call expects a full pyramid.'''
        tdl.core.assert_initialized(self, 'call', ['projections'])
        depth = len(inputs)
        inputs = inputs[::-1]
        assert depth == len(self.projections),\
            'pyramid size does not match the number of projections'
        proj = self.projections[0](inputs[0])
        out = self.layers[0](proj)
        for x_i, projection, layer in\
                zip(inputs[1:], self.projections[1:], self.layers[1:]):
            proj = projection(x_i)
            extended = tf.concat([proj, out], axis=-1)
            out = layer(extended)
        return out


class MSG_DiscriminatorTrainer_V2(MSG_DiscriminatorTrainer):
    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['real_pyramid', 'sim_pyramid', 'regularizer'])
        pred_real = self.discriminator(self.real_pyramid)
        pred_sim = self.discriminator(self.sim_pyramid)
        loss_real = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy()(
            tf.zeros_like(pred_sim), pred_sim)
        loss = (loss_real + loss_sim)/2.0
        return loss


class MSG_GeneratorTrainer_V2(MSG_GeneratorTrainer):
    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(
            self, 'loss', ['batch_size', 'xsim', 'sim_pyramid', 'regularizer',
                           'pyramid_loss'])
        pred = self.discriminator(self.sim_pyramid)
        loss = tf.keras.losses.BinaryCrossentropy()(
                tf.ones_like(pred), pred)
        # regularizer
        if self.regularizer is not None:
            layers = tdl.core.find_instances(
                self.generator,
                (tf.keras.layers.Conv2D, tdl.convnet.Conv2DLayer,
                 tdl.convnet.Conv1x1Proj))
            loss += self.regularizer.scale*tf.add_n(
                [self.regularizer.fn(layer.kernel)
                 for layer in layers]
            )
        if self.pyramid_loss is not None:
            loss += self.pyramid_loss.scale * self.pyramid_loss.fn()
        return loss


class MSG_GAN_V2(MSG_GAN):
    GeneratorTrainer = MSG_GeneratorTrainer_V2

    def DiscriminatorBaseModel(self, **kargs):
        tdl.core.assert_initialized(self, 'discriminator', ['generator'])
        tdl.core.assert_initialized(
            self.generator, 'discriminator', ['projections'])
        return MSG_DiscriminatorModel_V2()
    DiscriminatorHidden = MSG_DiscriminatorHidden
    DiscriminatorTrainer = MSG_DiscriminatorTrainer_V2
