import twodlearn as tdl
import tensorflow as tf
import numpy as np


class NonLocal(tdl.core.Layer):
    '''
    Non local module as described in:
        https://arxiv.org/pdf/1805.08318.pdf
    '''
    channel_reduction = tdl.core.InputArgument.optional(
        'channel_reduction', doc="scale used to reduce the channel",
        default=8)

    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=False)

    @tdl.core.Submodel
    def fx(self, _):
        tdl.core.assert_initialized(self, 'fx', ['input_shape', 'use_bias'])
        input_units = self.input_shape.as_list()[-1]
        return tdl.convnet.Conv1x1Proj(
            units=input_units//self.channel_reduction,
            use_bias=self.use_bias)

    @tdl.core.Submodel
    def gx(self, _):
        tdl.core.assert_initialized(self, 'gx', ['input_shape', 'use_bias'])
        input_units = self.input_shape.as_list()[-1]
        return tdl.convnet.Conv1x1Proj(
            units=input_units//self.channel_reduction,
            use_bias=self.use_bias)

    @tdl.core.Submodel
    def hx(self, _):
        tdl.core.assert_initialized(self, 'hx', ['input_shape', 'use_bias'])
        input_units = self.input_shape.as_list()[-1]
        return tdl.convnet.Conv1x1Proj(
            units=input_units//self.channel_reduction,
            use_bias=self.use_bias)

    @tdl.core.Submodel
    def vx(self, _):
        tdl.core.assert_initialized(self, 'vx', ['input_shape', 'use_bias'])
        input_units = self.input_shape.as_list()[-1]
        return tdl.convnet.Conv1x1Proj(
            units=input_units,
            use_bias=self.use_bias)

    def call(self, inputs):
        fx = self.fx(inputs)
        gx = self.gx(inputs)
        hx = self.hx(inputs)

        f_shape = self.fx.compute_output_shape(inputs.shape).as_list()

        matrix_shape = [-1, f_shape[1]*f_shape[2], f_shape[-1]]
        fx = tf.reshape(fx, shape=matrix_shape)
        gx = tf.reshape(gx, shape=matrix_shape)
        hx = tf.reshape(hx, shape=matrix_shape)

        map = tf.nn.softmax(tf.matmul(fx, gx, transpose_b=True))
        output = self.vx(tf.matmul(map, hx))

        return tf.reshape(output, shape=[-1] + inputs.shape.as_list()[1:])


class NonLocalBlock(NonLocal):
    '''
    Non local model with a residual connection
    '''
    @tdl.core.SubmodelInit
    def attention_weigth(self, value=0.0, trainable=True):
        if value is None:
            value = 0.0
        if isinstance(value, int):
            value = float(int)
        if isinstance(value, float) and trainable:
            value = tf.Variable(value, dtype=tf.float32)
        elif isinstance(value, np.ndarray) and trainable:
            value = tf.Variable(value, dtype=tf.float32)
        return value

    def call(self, inputs):
        fx = self.fx(inputs)
        gx = self.gx(inputs)
        hx = self.hx(inputs)

        f_shape = self.fx.compute_output_shape(inputs.shape).as_list()

        matrix_shape = [-1, f_shape[1]*f_shape[2], f_shape[-1]]
        fx = tf.reshape(fx, shape=matrix_shape)
        gx = tf.reshape(gx, shape=matrix_shape)
        hx = tf.reshape(hx, shape=matrix_shape)

        map = tf.nn.softmax(tf.matmul(fx, gx, transpose_b=True))
        output = self.vx(tf.matmul(map, hx))

        residual = tf.reshape(output, shape=[-1] + inputs.shape.as_list()[1:])

        return self.attention_weigth * residual + inputs
