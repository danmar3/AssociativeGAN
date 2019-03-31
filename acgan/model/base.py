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


@tdl.core.PropertyShortcuts({'model': ['discriminator', 'generator',
                                       'embedding_size']})
@tdl.core.create_init_docstring
class BaseTrainer(tdl.core.TdlModel):
    @tdl.core.InputModel
    def model(self, value):
        return value

    @tdl.core.InputArgument
    def batch_size(self, value):
        return value

    @tdl.core.SubmodelInit
    def optimizer(self, learning_rate, beta1=0.5):
        return tf.train.AdamOptimizer(learning_rate, beta1=beta1)

    @tdl.core.LazzyProperty
    def train_step(self):
        return tf.Variable(0, dtype=tf.int32, name='train_step')


class GeneratorTrainer(BaseTrainer):
    @tdl.core.OutputValue
    def embedding(self, _):
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
        loss = tf.keras.losses.BinaryCrossentropy()(
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


class DiscriminatorTrainer(BaseTrainer):
    @staticmethod
    def _add_noise(samples, noise_rate):
        uniform_noise = tf.random.uniform(
            shape=tf.shape(samples), minval=0, maxval=1)
        output = tf.where(uniform_noise > noise_rate, samples,
                          2*uniform_noise-1)
        return output

    @tdl.core.InputArgument
    def xreal(self, value):
        tdl.core.assert_initialized(self, 'xreal', ['train_step'])
        noise_rate = self.model.noise_rate(self.train_step)
        if noise_rate is not None:
            value = self._add_noise(value, noise_rate)
        return value

    @tdl.core.OutputValue
    def embedding(self, _):
        return tf.random.normal([self.batch_size, self.embedding_size])

    @tdl.core.OutputValue
    def xsim(self, _):
        tdl.core.assert_initialized(self, 'xsim', ['embedding'])
        xsim = self.generator(self.embedding, training=True)
        return xsim

    @tdl.core.OutputValue
    def loss(self, _):
        tdl.core.assert_initialized(self, 'loss', ['xreal', 'xsim'])
        pred_real = self.discriminator(self.xreal, training=True)
        pred_sim = self.discriminator(self.xsim, training=True)
        pred = tf.concat([pred_real, pred_sim], axis=0)
        loss_real = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy()(
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


class DiscriminatorOutput(tdl.stacked.StackedLayers):
    @tdl.core.Submodel
    def layers(self, _):
        value = [tf_layers.Flatten(),
                 tf_layers.Dense(1),
                 tf_layers.Activation(tf.keras.activations.sigmoid)]
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
    DiscriminatorOutput = DiscriminatorOutput
    DiscriminatorTrainer = DiscriminatorTrainer

    @staticmethod
    def _to_list(value, n_elements):
        if isinstance(value, int):
            value = [value]*n_elements
        if isinstance(value[0], int):
            value = [[vi, vi] for vi in value]
        assert len(value) == n_elements, \
            'list does not have the expected number of elements'
        assert all([len(vi) == 2 for vi in value]), \
            'list does not have the expected number of elements'
        return value

    @tdl.core.LazzyProperty
    def target_shape(self):
        tdl.core.assert_initialized(self, 'target_shape', ['generator'])
        return self.generator.compute_output_shape(
            [None, self.embedding_size])[1:]

    @tdl.core.SubmodelInit
    def generator(self, init_shape, units, kernels=5, strides=2,
                  padding='same'):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)
        model = self.GeneratorBaseModel()
        model.add(self.InputProjection(projected_shape=init_shape))
        for i in range(len(units)-1):
            model.add(self.GeneratorHidden(
                units=units[i], upsampling=strides[i],
                conv={'kernels': kernels[i], 'padding': padding[i]}))
        model.add(self.GeneratorOutput(
                units=units[-1], upsampling=strides[-1],
                conv={'kernels': kernels[-1], 'padding': padding[-1]}))
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

    def generator_trainer(self, batch_size, learning_rate=0.0002):
        tdl.core.assert_initialized(
            self, 'generator_trainer', ['generator', 'discriminator'])
        return self.GeneratorTrainer(
            model=self, batch_size=batch_size,
            optimizer={'learning_rate': learning_rate})

    @tdl.core.MethodInit
    def noise_rate(self, local, rate=None):
        local.rate = rate

    @noise_rate.eval
    def noise_rate(self, local, step):
        return (None if local.rate is None
                else tf.exp(-local.rate * tf.cast(step, tf.float32)))

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None,
                              learning_rate=0.0002):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate'])
        return self.DiscriminatorTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer={'learning_rate': learning_rate})

    def __init__(self, embedding_size, name=None, **kargs):
        self.embedding_size = embedding_size
        super(BaseGAN, self).__init__(name=None, **kargs)
