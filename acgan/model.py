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
    def generator(self, init_shape, units, kernels=5, strides=2):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        model = tdl.stacked.StackedLayers()

        model.add(tdl.dense.LinearLayer(
            units=functools.reduce(operator.mul, init_shape, 1)))
        model.add(tf_layers.Reshape(init_shape))

        for i in range(len(units)):
            model.add(tf_layers.LeakyReLU())
            model.add(tf_layers.Conv2DTranspose(
                units[i], kernels[i], strides=strides[i],
                padding='same', use_bias=False))
        return model

    @tdl.core.SubmodelInit
    def discriminator(self, units, kernels, strides, dropout=None):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        model = tdl.stacked.StackedLayers()

        for i in range(len(units)):
            model.add(tf_layers.Conv2D(
                units[i], kernels[i], strides=strides[i],
                padding='same'))
            model.add(tf_layers.LeakyReLU())
            if dropout is not None:
                model.add(tf_layers.Dropout(dropout))
        model.add(tf_layers.Flatten())
        model.add(tf_layers.Dense(1))
        return model

    def generator_trainer(self, batch_size):
        noise = tf.random.normal([batch_size, self.embedding_size])
        xsim = self.generator(noise)
        pred = self.discriminator(xsim)
        loss = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(pred), pred)
        step = tf.train.AdamOptimizer().minimize(loss)
        return tdl.core.SimpleNamespace(
            loss=loss, xsim=xsim, pred=pred, step=step
        )

    def discriminator_trainer(self, batch_size, xreal=None, input_shape=None):
        if xreal is None:
            xreal = tf.keras.Input(shape=input_shape)
        noise = tf.random.normal([batch_size, self.embedding_size])
        xsim = self.generator(noise)

        pred_real = self.discriminator(xreal)
        pred_sim = self.discriminator(xsim)
        pred = tf.concat([pred_real, pred_sim], axis=0)
        loss_real = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy()(
            tf.ones_like(pred_sim), pred_sim)
        loss = loss_real + loss_sim

        step = tf.train.AdamOptimizer().minimize(loss)
        return tdl.core.SimpleNamespace(
            loss=loss, xreal=xreal, xsim=xsim,
            output=pred, step=step
        )

    def __init__(self, embedding_size, name=None, **kargs):
        self.embedding_size = embedding_size
        super(DCGAN, self).__init__(name=None, **kargs)
