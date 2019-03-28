import tensorflow as tf
import tensorflow.keras.layers as tf_layers
import twodlearn as tdl
import operator
import functools


@tdl.core.create_init_docstring
class DCGAN(tdl.core.TdlModel):
    @staticmethod
    def _to_list(value, n_elements):
        if isinstance(value, int):
            value = [value]*n_elements
        if isinstance(value[0], int):
            value = [(vi, vi) for vi in value]
        assert len(value) == n_elements, \
            'list does not have the expected number of elements'
        assert all([len(vi) == 2 for vi in value]), \
            'list does not have the expected number of elements'
        return value

    @tdl.core.SubmodelInit
    def generator(self, init_shape, units, kernels=5, strides=2,
                  padding='same'):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)
        model = tdl.stacked.StackedLayers()

        model.add(tdl.dense.LinearLayer(
            units=functools.reduce(operator.mul, init_shape, 1)))
        model.add(tf_layers.Reshape(init_shape))

        for i in range(len(units)):
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf_layers.ReLU())
            # model.add(tf_layers.Activation(tf.keras.activations.softplus))
            model.add(tf_layers.Conv2DTranspose(
                units[i], kernels[i], strides=strides[i],
                padding=padding[i],
                use_bias=False))
        model.add(tf_layers.Activation(tf.keras.activations.tanh))
        return model

    @tdl.core.SubmodelInit
    def discriminator(self, units, kernels, strides, dropout=None,
                      padding='same'):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)
        model = tdl.stacked.StackedLayers()

        for i in range(len(units)):
            model.add(tf_layers.Conv2D(
                units[i], kernels[i], strides=strides[i],
                padding=padding[i]))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf_layers.LeakyReLU(0.2))
            # model.add(tf_layers.Activation(tf.keras.activations.softplus))
            if dropout is not None:
                model.add(tf_layers.Dropout(dropout))
        model.add(tf_layers.Flatten())
        model.add(tf_layers.Dense(1))
        model.add(tf_layers.Activation(tf.keras.activations.sigmoid))
        return model

    def generator_trainer(self, batch_size, learning_rate=0.0002):
        tdl.core.assert_initialized(self, 'generator_trainer',
                                    ['generator', 'discriminator'])
        noise = tf.random.normal([batch_size, self.embedding_size])
        xsim = self.generator(noise, training=True)
        pred = self.discriminator(xsim, trainint=True)
        loss = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(pred), pred)

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        step = optimizer.minimize(
            loss, var_list=tdl.core.get_trainable(self.generator))
        return tdl.core.SimpleNamespace(
            loss=loss, xsim=xsim, pred=pred, step=step,
            optimizer=optimizer,
            variables=(optimizer.variables() +
                       tdl.core.get_variables(self.generator))
        )

    @tdl.core.MethodInit
    def noise_rate(self, local, rate=None):
        local.rate = rate

    @noise_rate.eval
    def noise_rate(self, local, step):
        if local.rate is None:
            return None
        else:
            return tf.exp(-local.rate * tf.cast(step, tf.float32))

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None,
                              learning_rate=0.0002):
        tdl.core.assert_initialized(self, 'discriminator_trainer',
                                    ['generator', 'discriminator',
                                     'noise_rate'])
        if xreal is None:
            xreal = tf.keras.Input(shape=input_shape)
        noise = tf.random.normal([batch_size, self.embedding_size])
        xsim = self.generator(noise, trainint=True)

        train_step = tf.Variable(0, dtype=tf.int32, name='disc_train_step')
        noise_rate = self.noise_rate(train_step)
        if noise_rate is not None:
            uniform_noise = tf.random.uniform(shape=tf.shape(xreal),
                                              minval=0, maxval=1)
            xreal = tf.where(uniform_noise > noise_rate, xreal,
                             2*uniform_noise-1)
            # uniform_noise = tf.random.uniform(shape=tf.shape(xreal),
            #                                   minval=0, maxval=1)
            # xsim = tf.where(uniform_noise > noise_rate, xsim,
            #                 2*uniform_noise-1)

        pred_real = self.discriminator(xreal, training=True)
        pred_sim = self.discriminator(xsim, trainint=True)
        pred = tf.concat([pred_real, pred_sim], axis=0)
        loss_real = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy()(
            tf.zeros_like(pred_sim), pred_sim)
        loss = (loss_real + loss_sim)/2.0

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
        with tf.control_dependencies([train_step.assign_add(1)]):
            step = optimizer.minimize(
                loss, var_list=tdl.core.get_trainable(self.discriminator))
        return tdl.core.SimpleNamespace(
            loss=loss, xreal=xreal, xsim=xsim,
            output=pred, step=step, optimizer=optimizer,
            variables=(optimizer.variables() + [train_step] +
                       tdl.core.get_variables(self.discriminator)),
            train_step=train_step,
            noise_rate=noise_rate
        )

    def __init__(self, embedding_size, name=None, **kargs):
        self.embedding_size = embedding_size
        super(DCGAN, self).__init__(name=None, **kargs)
