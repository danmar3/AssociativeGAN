import tensorflow
import tensorflow.keras.layers as tf_layers
import twodlearn as tdl
import operator
import functools


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
    def discriminator(self, units, kernels, strides, dropout):
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        model = tdl.stacked.StackedLayers()

        for i in range(len(units)):
            model.add(tf_layers.Conv2D(
                units[i], kernels[i], strides=strides[i],
                padding='same'))
            model.add(tf_layers.LeakyReLU())
            model.add(tf_layers.Dropout(0.3))
        model.add(tf_layers.Flatten())
        model.add(tf_layers.Dense(1))
        return model

    def __init__(self, embedding_size, name=None, **kargs):
        self.embedding_size = embedding_size
        super(DCGAN, self).__init__(name=None, **kargs)
