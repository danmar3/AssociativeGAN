import tensorflow as tf
import tensorflow.keras.layers as tf_layers
import twodlearn as tdl
import typing


def compute_output_shape(chain, input_shape):
    output_shape = input_shape
    for fn in chain:
        if fn is not None:
            output_shape = fn.compute_output_shape(output_shape)
    return output_shape


@tdl.core.create_init_docstring
class TransposeLayer(tdl.core.Layer):
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
    def conv(self, kernels, padding):
        tdl.core.assert_initialized(self, 'conv', ['units', 'upsampling'])
        return tf_layers.Conv2DTranspose(
                self.units, kernel_size=kernels,
                strides=self.upsampling, padding=padding,
                use_bias=False)

    @tdl.core.Submodel
    def activation(self, value):
        return value

    def compute_output_shape(self, input_shape=None):
        chain = [self.conv, self.activation]
        return compute_output_shape(chain, input_shape)

    def call(self, inputs):
        output = self.conv(inputs)
        if self.activation is not None:
            output = self.activation(output)
        return output


@tdl.core.create_init_docstring
class BatchNormalization(tdl.core.Layer):
    @tdl.core.Submodel
    def batch_normalization(self, value: typing.Union[bool, None]):
        if value is True or value is None:
            return tf.keras.layers.BatchNormalization()
        return None


@tdl.core.create_init_docstring
class MinibatchStddev(tdl.core.Layer):
    @tdl.core.InputArgument
    def tolerance(self, value):
        if value is None:
            value = 1e-8
        return value

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape[:-1].concatenate(input_shape[-1].value + 1)

    def call(self, inputs):
        assert inputs.shape.ndims == 4, 'input must be a 4D tensor'
        inputs = tf.convert_to_tensor(inputs)
        diff = inputs - tf.reduce_mean(inputs, axis=0)[tf.newaxis, ...]
        # [1, H, W, D]
        stddev = tf.sqrt(tf.reduce_mean(diff**2.0, axis=0)
                         + self.tolerance)[tf.newaxis, ...]
        c_value = tf.fill(tf.shape(inputs)[:-1],
                          tf.reduce_mean(stddev))[..., tf.newaxis]
        return tf.concat([inputs, c_value], axis=-1)


@tdl.core.create_init_docstring
class VectorNormalizer(tdl.core.Layer):
    @tdl.core.InputArgument
    def tolerance(self, value):
        if value is None:
            value = 1e-8
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        moment2 = tf.reduce_mean(tf.square(inputs), axis=-1) + self.tolerance
        return inputs*tf.rsqrt(moment2[..., tf.newaxis])


@tdl.core.PropertyShortcuts({'model': ['discriminator', 'generator',
                                       'embedding_size']})
@tdl.core.create_init_docstring
class BaseTrainer(tdl.core.TdlModel):
    @tdl.core.InputModel
    def model(self, value):
        return value

    @tdl.core.InputArgument
    def batch_size(self, value):
        '''Number of images extracted/generated for training.'''
        return value

    @tdl.core.SubmodelInit
    def optimizer(self, learning_rate, beta1=0.0):
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1)
        # return tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    @tdl.core.LazzyProperty
    def train_step(self):
        '''Number of training steps executed.'''
        return tf.Variable(0, dtype=tf.int32, name='train_step')


class GeneratorTrainer(BaseTrainer):
    @tdl.core.OutputValue
    def embedding(self, _):
        '''Random samples in embedded space used to generate images.'''
        return tf.random.normal([self.batch_size, self.embedding_size])

    @tdl.core.OutputValue
    def xsim(self, _):
        tdl.core.assert_initialized(self, 'xsim', ['embedding'])
        xsim = self.generator(self.embedding, training=True)
        return xsim

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(self, 'loss', ['batch_size', 'xsim'])
        pred = self.discriminator(self.xsim, training=True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(pred), pred)
        return loss

    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=tdl.core.get_trainable(self.generator))
        return step

    @property
    def variables(self):
        tdl.core.assert_initialized(
            self, 'variables', ['optimizer', 'train_step'])
        return (self.optimizer.variables() + [self.train_step] +
                tdl.core.get_variables(self.generator))


def _add_noise(samples, noise_rate):
    uniform_noise = tf.random.uniform(
        shape=tf.shape(samples), minval=0, maxval=1)
    output = tf.where(uniform_noise > noise_rate, samples,
                      2*uniform_noise-1)
    return output


class DiscriminatorTrainer(BaseTrainer):
    @tdl.core.InputArgument
    def xreal(self, value):
        '''real images obtained from the dataset'''
        tdl.core.assert_initialized(self, 'xreal', ['train_step'])
        noise_rate = self.model.noise_rate(self.train_step)
        if noise_rate is not None:
            value = _add_noise(value, noise_rate)
        return value

    @tdl.core.OutputValue
    def embedding(self, _):
        return tf.random.normal([self.batch_size, self.embedding_size])

    @tdl.core.OutputValue
    def xsim(self, _):
        '''generated images from the genrator model'''
        tdl.core.assert_initialized(self, 'xsim', ['embedding'])
        xsim = self.generator(self.embedding, training=True)
        noise_rate = self.model.noise_rate(self.train_step)
        if noise_rate is not None:
            xsim = _add_noise(xsim, noise_rate)
        return xsim

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(self, 'loss', ['xreal', 'xsim'])
        pred_real = self.discriminator(self.xreal, training=True)
        pred_sim = self.discriminator(self.xsim, training=True)
        pred = tf.concat([pred_real, pred_sim], axis=0)
        loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.zeros_like(pred_sim), pred_sim)
        loss = (loss_real + loss_sim)/2.0
        return loss

    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=tdl.core.get_trainable(self.discriminator))
        return step

    @property
    def variables(self):
        tdl.core.assert_initialized(
            self, 'variables', ['optimizer', 'train_step'])
        return (self.optimizer.variables() + [self.train_step] +
                tdl.core.get_variables(self.discriminator))


class DiscriminatorHidden(tdl.stacked.StackedLayers):
    @tdl.core.Submodel
    def layers(self, _):
        value = [tf_layers.Conv2D(
                    self.units, self.kernels, strides=self.strides,
                    padding=self.padding),
                 tf.keras.layers.BatchNormalization(),
                 tf_layers.LeakyReLU(0.2)]
        if self.dropout is not None:
            value.append(tf_layers.Dropout(self.dropout))
        return value

    def __init__(self, units, kernels, strides, padding, dropout, **kargs):
        self.units = units
        self.kernels = kernels
        self.strides = strides
        self.padding = padding
        self.dropout = dropout
        super(DiscriminatorHidden, self).__init__(**kargs)


class DiscriminatorOutputBase(tdl.stacked.StackedLayers):
    @tdl.core.Submodel
    def layers(self, _):
        value = [tf_layers.Flatten(),
                 tf_layers.Dense(1),
                 # tf_layers.Activation(tf.keras.activations.sigmoid)
                 ]
        return value


@tdl.core.create_init_docstring
class BaseGAN(tdl.core.TdlModel):
    GeneratorBaseModel = tdl.stacked.StackedLayers
    InputProjection = None
    GeneratorHidden = None
    GeneratorOutput = None
    GeneratorTrainer = GeneratorTrainer

    DiscriminatorBaseModel = tdl.stacked.StackedLayers
    DiscriminatorHidden = DiscriminatorHidden
    DiscriminatorOutput = DiscriminatorOutputBase
    DiscriminatorTrainer = DiscriminatorTrainer

    @staticmethod
    def _to_list(value, n_elements):
        '''check if value is a list. If not, return a list with n_elements.

        If value is an integer, each element is a 2-dim tuple (value, value).
        If value is an iterable, each element of the new list is a tuple
        with duplicated elements (value[i], value[i]) '''
        if isinstance(value, int):
            value = [value]*n_elements
        if isinstance(value[0], int) or (value[0] is None):
            value = [[vi, vi] for vi in value]
        assert len(value) == n_elements, \
            'list does not have the expected number of elements'
        assert all([len(vi) == 2 for vi in value]), \
            'list does not have the expected number of elements'
        return value

    @tdl.core.LazzyProperty
    def target_shape(self):
        '''Shape of the generated images.'''
        tdl.core.assert_initialized(self, 'target_shape', ['generator'])
        return self.generator.compute_output_shape(
            [None, self.embedding_size])[1:]

    @tdl.core.SubmodelInit
    def generator(self, init_shape, units, outputs, output_kargs=None,
                  kernels=5, strides=2, add_noise=False,
                  padding='same'):
        n_layers = len(units) + 1
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        add_noise = (add_noise if isinstance(add_noise, (list, tuple))
                     else [add_noise]*n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)
        model = self.GeneratorBaseModel(
            input_shape=[None, self.embedding_size])
        model.add(self.InputProjection(projected_shape=init_shape))
        for i in range(len(units)):
            model.add(self.GeneratorHidden(
                units=units[i], upsampling=strides[i], add_noise=add_noise[i],
                conv={'kernels': kernels[i], 'padding': padding[i]}))
        if output_kargs is None:
            output_kargs = dict()
        model.add(self.GeneratorOutput(units=outputs, **output_kargs))
        return model

    @tdl.core.SubmodelInit
    def discriminator(self, units, kernels, strides, dropout=None,
                      padding='same'):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = self.DiscriminatorBaseModel()
        for i in range(len(units)):
            model.add(self.DiscriminatorHidden(
                units=units[i], kernels=kernels[i], strides=strides[i],
                padding=padding[i], dropout=dropout))
        model.add(self.DiscriminatorOutput())
        return model

    def generator_trainer(self, batch_size, optimizer=None,
                          **kwargs):
        tdl.core.assert_initialized(
            self, 'generator_trainer', ['generator', 'discriminator'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.GeneratorTrainer(
            model=self, batch_size=2*batch_size, optimizer=optimizer,
            **kwargs)

    @tdl.core.MethodInit
    def noise_rate(self, local, rate=None):
        local.rate = rate

    @noise_rate.eval
    def noise_rate(self, local, step):
        return (None if local.rate is None
                else tf.exp(-local.rate * tf.cast(step, tf.float32)))

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None,
                              optimizer=None,
                              **kwargs):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.DiscriminatorTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer=optimizer,
            **kwargs)

    def __init__(self, embedding_size, name=None, **kargs):
        self.embedding_size = embedding_size
        super(BaseGAN, self).__init__(name=None, **kargs)
