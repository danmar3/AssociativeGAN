import typing
from types import SimpleNamespace

import twodlearn as tdl
import tensorflow as tf
import tensorflow.keras.layers as tf_layers
from twodlearn.resnet import convnet as tdl_convnet

from . import equalized
from . import res_conv2d
from .base import VectorNormalizer, MinibatchStddev
from .losses import DLogistic, DLogisticSimpleGP, NegLogProb, EmbeddingLoss
from .encoder import Estimator, CallWrapper
from .gmm import GMM
from .pyramid import ImagePyramid

BATCHNORM_TYPES = {
    'batchnorm': tf_layers.BatchNormalization,
    'pixelwise': VectorNormalizer
}


class NoiseLayer(tdl.core.Layer):
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    @tdl.core.ParameterInit(lazzy=True)
    def kernel(self, initializer=None, trainable=True, **kargs):
        tdl.core.assert_initialized(self, 'kernel', ['input_shape'])
        if initializer is None:
            initializer = tf.keras.initializers.zeros()
        return self.add_weight(
            name='kernel',
            initializer=initializer,
            shape=[self.input_shape[-1].value],
            trainable=trainable,
            **kargs)

    def call(self, inputs):
        assert len(inputs.shape) == 4  # NHWC
        inputs = tf.convert_to_tensor(inputs)
        noise = tf.random_normal(
            [tf.shape(inputs)[0], inputs.shape[1], inputs.shape[2], 1],
            dtype=inputs.dtype)
        return inputs + noise * tf.reshape(tf.cast(self.kernel, inputs.dtype),
                                           [1, 1, 1, -1])


class Discriminator(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of units in each hidden block of the network')

    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.',
        default='pixelwise')
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)
    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=True)

    LAYER_TYPE = {'plain': res_conv2d.ResStages,
                  'bottleneck': res_conv2d.ResBottleneck}

    @tdl.core.SimpleParameter
    def layer_lib(self, _):
        tdl.core.assert_initialized(self, 'layer_lib', ['equalized'])
        if self.equalized:
            return equalized.get_layer_lib('equalized')
        else:
            return equalized.get_layer_lib('keras')

    def hidden_shapes(self):
        tdl.core.assert_initialized(
            self, 'hidden_shapes', ['input_shape', 'projections', 'hidden'])
        pyramid_shape = self.input_shape
        input_shapes = pyramid_shape[::-1]
        output_list = list()
        output_shape = self.input_layer.compute_output_shape(input_shapes[0])

        output_list.append(output_shape)
        for x_shape, projection, layer in\
                zip(input_shapes, self.projections, self.hidden[:len(self.units)]):
            proj_shape = projection.compute_output_shape(x_shape)
            extended = proj_shape[:-1].concatenate(
                proj_shape.as_list()[-1] + output_shape.as_list()[-1])
            output_shape = layer.compute_output_shape(extended)
            output_list.append(output_shape)
        for layer in self.hidden[len(self.units):]:
            output_shape = layer.compute_output_shape(output_shape)
            output_list.append(output_shape)
        return output_list

    @tdl.core.SubmodelInit(lazzy=True)
    def projections(self, activation=None):
        tdl.core.assert_initialized(
            self, 'projections',
            ['units', 'hidden', 'input_shape', 'use_bias', 'layer_lib'])
        projections = list()
        for units in self.units:
            projections.append(
                self.layer_lib.Conv1x1Proj(
                    units=units,
                    activation=activation,
                    use_bias=self.use_bias
                ))
        return projections

    @tdl.core.SubmodelInit(lazzy=True)
    def input_layer(self, units=None, kernel_size=3):
        tdl.core.assert_initialized(
            self, 'input_layer',
            ['layer_lib', 'units', 'use_bias'])
        if units is None:
            units = self.units[0]
        return self.layer_lib.Conv2D(
            filters=units,
            kernel_size=[kernel_size, kernel_size],
            padding='same',
            use_bias=self.use_bias)

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, stages=3, batchnorm=None, dropout=None,
               kernel_size=3, pooling=2,
               leaky_rate=0.2,
               layer_type="bottleneck"):
        tdl.core.assert_initialized(
            self, 'hidden', ['units', 'use_bias', 'equalized'])
        n_layers = len(self.units)
        # kernels arg
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernels = [kernel_size for _ in range(n_layers)]
        assert all(isinstance(ki, int) for ki in kernels)
        # pooling arg
        assert (isinstance(pooling, (int, list, tuple)) or (pooling is None))
        if isinstance(pooling, (int, dict)):
            pooling = [pooling for _ in range(n_layers)]
        assert all(isinstance(pi, (int, dict)) or (pi is None)
                   for pi in pooling)

        layers = list()
        for i in range(len(self.units)):
            layers.append(self.LAYER_TYPE[layer_type](
                stages=stages,
                units=self.units[i],
                kernel_size=kernels[i],
                batchnorm=batchnorm,
                leaky_rate=leaky_rate,
                pooling=pooling[i],
                dropout=dropout,
                use_bias=self.use_bias,
                equalized=self.equalized,
                layers={'resize_method': 'avg_pool'}
                ))
        layers.append(MinibatchStddev())
        layers.append(self.LAYER_TYPE[layer_type](
            stages=stages,
            units=self.units[-1],
            kernel_size=3,
            batchnorm=batchnorm,
            leaky_rate=leaky_rate,
            pooling=None,
            dropout=dropout,
            use_bias=self.use_bias,
            equalized=self.equalized,
            layers={'resize_method': 'avg_pool'}
            ))
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def flatten(self, method='global_maxpool'):
        layers = tdl.stacked.StackedLayers()
        if method == 'flatten':
            layers.add(tf_layers.Flatten())
        elif method == 'global_maxpool':
            layers.add(tf_layers.GlobalMaxPooling2D())
        elif method == 'global_avgpool':
            layers.add(tf_layers.GlobalAveragePooling2D())
        else:
            raise ValueError(f'flatten option {method} not recognized')
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def dense(self, **kargs):
        return tdl.stacked.StackedLayers(layers=[
            self.layer_lib.Dense(units=1)
            ])

    def call(self, inputs, output='logits'):
        '''the call expects a full pyramid.'''
        tdl.core.assert_initialized(self, 'call', ['projections'])
        depth = len(inputs)
        inputs = inputs[::-1]
        assert depth == len(self.projections),\
            'pyramid size does not match the number of projections'
        out = self.input_layer(inputs[0])
        # hidden
        hidden_h = [out]
        for x_i, projection, layer in\
                zip(inputs, self.projections, self.hidden[:len(self.units)]):
            proj = projection(x_i)
            extended = tf.concat([proj, out], axis=-1)
            out = layer(extended)
            hidden_h.append(out)
        # finalize
        for layer in self.hidden[len(self.units):]:
            out = layer(out)
            hidden_h.append(out)
        # flatten
        if output == 'hidden':
            return hidden_h
        if self.flatten:
            out = self.flatten(out)
        if output == 'flatten':
            return out
        elif output == 'logits':
            return self.dense(out)
        elif output == 'prob':
            return tf.nn.sigmoid(self.dense(out))
        else:
            raise ValueError(f'output {output} not a valid option.')


class NewAxis(tdl.core.Layer):
    @tdl.core.InputArgument
    def axis(self, value):
        if value is None:
            raise ValueError('axis must be specified')
        assert isinstance(value, (list, tuple))
        assert all(ai in (..., tf.newaxis) for ai in value)
        return value

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(self, 'compute_output_shape', ['axis'])
        output_shape = list()
        for ai in self.axis:
            if ai is tf.newaxis:
                output_shape.append(1)
            elif ai is ...:
                output_shape = output_shape + input_shape.as_list()
            else:
                raise ValueError('')
        return tf.TensorShape(output_shape)

    def call(self, inputs):
        return inputs[self.axis]


class TransposeLayer(tdl.core.Layer):
    @tdl.core.InputArgument
    def axis(self, value):
        if value is None:
            raise ValueError('axes must be specified')
        assert isinstance(value, (list, tuple))
        assert all(isinstance(ai, int) for ai in value)
        return value

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(self, 'compute_output_shape', ['axis'])
        return tf.TensorShape(input_shape[ai] for ai in self.axis)

    def call(self, inputs):
        return tf.transpose(inputs, self.axis)


class Generator(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of units in each hidden block of the network')

    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.',
        default='pixelwise')
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)
    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=True)
    output_activation = tdl.core.InputArgument.optional(
        'output_activation', doc="activation function at the output.",
        default=None)
    output_channels = tdl.core.InputArgument.optional(
        'output_channels', doc="number of output channels",
        default=3)

    LAYER_TYPE = {'plain': res_conv2d.ResStages,
                  'bottleneck': res_conv2d.ResBottleneck}

    @tdl.core.SimpleParameter
    def layer_lib(self, _):
        tdl.core.assert_initialized(self, 'layer_lib', ['equalized'])
        if self.equalized:
            return equalized.get_layer_lib('equalized')
        else:
            return equalized.get_layer_lib('keras')

    @tdl.core.InputArgument
    def embedding_size(self, value: typing.Union[int, None]):
        if value is None:
            tdl.core.assert_initialized(
                self, 'embedding_size', ['input_shape'])
            value = self.input_shape.as_list()[-1]
        return value

    def hidden_shapes(self):
        tdl.core.assert_initialized(
            self, 'hidden_shapes', ['embedding_size', 'input_layer', 'hidden'])
        _input_shape = tf.TensorShape([None, self.embedding_size])
        hidden_shapes = list()
        for layer in [self.input_layer] + self.hidden.layers:
            _input_shape = layer.compute_output_shape(_input_shape)
            hidden_shapes.append(_input_shape)
        return hidden_shapes

    def output_shape(self):
        return self.projections[-1].compute_output_shape(
            self.hidden_shapes()[-1])

    def pyramid_shapes(self):
        tdl.core.assert_initialized(
            self, 'pyramid_shapes', ['projections'])
        shapes = list()
        for proj, hidden_shape in zip(
                self.projections, self.hidden_shapes()[1:]):
            shapes.append(proj.compute_output_shape(hidden_shape))
        return shapes

    @tdl.core.Submodel
    def projections(self, value):
        if value is not None:
            return value
        tdl.core.assert_initialized(
            self, 'projections',
            ['output_activation', 'output_channels', 'use_bias', 'hidden',
             'layer_lib'])
        projections = [
            self.layer_lib.Conv1x1Proj(
                units=self.output_channels,
                activation=(self.output_activation() if self.output_activation
                            else None),
                use_bias=self.use_bias)
            for i in range(len(self.hidden.layers))]
        return projections

    @tdl.core.SubmodelInit(lazzy=True)
    def input_layer(self, target_shape, batchnorm=None, kernel_size=3):
        tdl.core.assert_initialized(
            self, 'input_layer', ['layer_lib', 'use_bias', 'batchnorm'])
        # batchnorm
        if self.batchnorm is not None:
            batchnorm = self.batchnorm
        if isinstance(batchnorm, str):
            batchnorm = BATCHNORM_TYPES[batchnorm]
        # project
        layers = tdl.stacked.StackedLayers()
        layers.add(NewAxis(axis=[..., tf.newaxis, tf.newaxis]))
        layers.add(TransposeLayer(axis=[0, 3, 2, 1]))

        target_shape = tf.TensorShape(target_shape).as_list()
        layers.add(self.layer_lib.Conv2DTranspose(
            filters=target_shape[-1],
            kernel_size=[target_shape[0], target_shape[1]]
        ))
        if batchnorm:
            layers.add(batchnorm())
        # conv
        # layers.add(tf_layers.LeakyReLU(self.leaky_rate))
        layers.add(self.layer_lib.Conv2D(
            filters=target_shape[-1],
            kernel_size=kernel_size,
            padding='same',
            use_bias=self.use_bias
        ))
        # layers.add(tf_layers.LeakyReLU(self.leaky_rate))
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, stages=3, batchnorm=None, dropout=None,
               kernel_size=3, upsample=2,
               leaky_rate=0.2, add_noise=True,
               layer_type="bottleneck"):
        tdl.core.assert_initialized(
            self, 'hidden', ['units', 'use_bias', 'equalized'])
        n_layers = len(self.units)
        # kernels arg
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernels = [kernel_size for _ in range(n_layers)]
        assert all(isinstance(ki, int) for ki in kernels)
        # upsample arg
        assert (isinstance(upsample, (int, dict, list, tuple)) or (upsample is None))
        if isinstance(upsample, (int, dict)):
            upsample = [upsample for _ in range(n_layers)]
        assert all(isinstance(pi, (int, dict)) or (pi is None)
                   for pi in upsample)
        # batchnorm
        if self.batchnorm is not None:
            batchnorm = self.batchnorm

        layers = tdl.stacked.StackedLayers()
        for i in range(len(self.units)):
            stage = tdl.stacked.StackedLayers()
            if add_noise:
                stage.add(NoiseLayer())
            stage.add(self.LAYER_TYPE[layer_type](
                stages=stages,
                units=self.units[i],
                kernel_size=kernels[i],
                batchnorm=batchnorm,
                leaky_rate=leaky_rate,
                upsample=(upsample[i] if isinstance(upsample[i], dict) else
                          {'size': upsample[i]}),
                dropout=dropout,
                use_bias=self.use_bias,
                equalized=self.equalized
                ))
            layers.add(stage)
        return layers

    def call(self, inputs, output='pyramid'):
        '''the call expects a full pyramid.'''
        out = inputs
        if self.input_layer:
            out = self.input_layer(out)
        hidden = self.hidden(out, output="hidden")
        out = hidden[-1]

        if output == 'hidden':
            return hidden
        elif output == 'image':
            return self.projections[-1](out)
        elif output == 'pyramid':
            return [pi(hi) for hi, pi in zip(hidden, self.projections)]

    def pyramid(self, inputs):
        return self(inputs, output='pyramid')


class NormalResNet(res_conv2d.ResNet):
    @tdl.core.Submodel
    def normal(self, _):
        tdl.core.assert_initialized(self, 'normal', ['dense'])
        return tdl.bayesnet.NormalModel(
            loc=lambda x: x,
            batch_shape=self.dense.layers[-1].units)

    def call(self, inputs, output='normal'):
        if output == 'normal':
            out = super(NormalResNet, self).call(inputs, output='dense')
            if self.normal:
                out = self.normal(out)
        else:
            out = super(NormalResNet, self).call(inputs, output=output)
        return out


@tdl.core.create_init_docstring
class BiGmmGan(tdl.core.TdlModel):
    batchnorm = tdl.core.InputArgument.optional(
        'batchnorm', doc='batch normalization to use.',
        default='pixelwise')
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)
    equalized = tdl.core.InputArgument.optional(
        'equalized', doc="use equalized version of conv layers",
        default=True)

    embedding_size = tdl.core.InputArgument.required(
        'embedding_size', doc='dimension of embedding space.')

    @tdl.core.InputArgument
    def discriminator_loss(self, value, **kargs):
        tdl.core.assert_initialized(
            self, 'discriminator_loss', ['discriminator'])
        if value is None:
            value = 'simplegp'
        if isinstance(value, str):
            losses = {'logistic': DLogistic, 'simplegp': DLogisticSimpleGP}
            value = losses[value](discriminator=self.discriminator, **kargs)
        return value

    @tdl.core.SubmodelInit
    def generator(self, init_shape, units, output_channels=3,
                  output_activation=None,
                  add_noise=False, **kargs):
        tdl.core.assert_initialized(
            self, 'generator',
            ['batchnorm', 'use_bias', 'equalized', 'embedding_size'])

        hidden = {'upsample': {'size': 2, 'method': 'nearest', 'layers': 1}}
        if 'hidden' in kargs:
            hidden = {**hidden, **kargs['hidden']}
        kargs['hidden'] = hidden
        return Generator(
            input_layer={'target_shape': init_shape},
            units=units,
            output_activation=output_activation,
            output_channels=output_channels,
            embedding_size=self.embedding_size,
            batchnorm=self.batchnorm,
            equalized=self.equalized,
            use_bias=self.use_bias,
            **kargs)

    @tdl.core.SubmodelInit
    def discriminator(self, units, **kargs):
        tdl.core.assert_initialized(
            self, 'generator',
            ['batchnorm', 'use_bias', 'equalized', 'embedding_size'])
        return Discriminator(
            units=units,
            batchnorm=self.batchnorm,
            equalized=self.equalized,
            use_bias=self.use_bias,
            **kargs)

    @tdl.core.SubmodelInit
    def embedding(self, n_components, init_loc=1e-5, init_scale=1.0,
                  min_scale_p=None, constrained_loc=False):
        '''Embedding model P(z)

        Args:
            n_components: number of clusters.
            init_loc: init mean.
            init_scale: init scale.
            min_scale_p: minimum scale (in percentage of maximum).
        '''
        tdl.core.assert_initialized(self, 'embedding', ['embedding_size'])
        model = GMM(
            n_dims=self.embedding_size,
            n_components=n_components,
            components={'init_loc': init_loc,
                        'init_scale': init_scale,
                        'constrained_loc': constrained_loc})
        return model

    @tdl.core.SubmodelInit
    def encoder(self, units, kernels=3, strides=None, pooling=2, stages=3,
                layer_type='plain'):
        tdl.core.assert_initialized(
            self, 'generator',
            ['batchnorm', 'use_bias', 'equalized', 'embedding_size'])
        return NormalResNet(
           input_layer={'units': 16},
           hidden={
               'units': units,
               'kernels': kernels,
               'strides': strides,
               'pooling': pooling,
               'stages': stages,
               'layer_type': layer_type,
               'pre_activation': True},
           flatten={'method': 'global_avgpool',
                    'batchnorm': 'batchnorm'},
           dense={'units': self.embedding_size},
           use_bias=self.use_bias,
           equalized=self.equalized,
           batchnorm=self.batchnorm)

    @tdl.core.SubmodelInit
    def pyramid(self, interpolation='nearest'):
        tdl.core.assert_initialized(self, 'pyramid', ['generator'])
        sizes = [si.as_list()[1:3]
                 for si in self.generator.hidden_shapes()[1:]]

        scales = [(sizes[idx][0]//sizes[idx-1][0],
                   sizes[idx][1]//sizes[idx-1][1])
                  for idx in range(1, len(sizes))]
        return ImagePyramid(scales=scales, interpolation=interpolation)

    def generator_trainer(self, batch_size, learning_rate, beta1=0.0):
        train_step = tf.Variable(0, dtype=tf.int32, name='train_step')
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate, beta1=beta1)

        xsim = self.generator(self.embedding(batch_size), output="pyramid")
        logits = self.discriminator(xsim)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(logits), logits)
        with tf.control_dependencies([train_step.assign_add(1)]):
            opt_step = optimizer.minimize(
                loss, var_list=tdl.core.get_trainable(self.generator))
        return SimpleNamespace(
            xsim=xsim[-1],
            loss=loss,
            train_step=train_step,
            optimizer=optimizer,
            step=opt_step,
            variables=(optimizer.variables() + [train_step]
                       + tdl.core.get_variables(self.generator))
        )

    def discriminator_trainer(self, xreal, batch_size, learning_rate, beta1=0.0):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['embedding', 'generator', 'discriminator', 'discriminator_loss',
             'pyramid'])
        train_step = tf.Variable(0, dtype=tf.int32, name='train_step')
        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate, beta1=beta1)

        xreal = [tf.stop_gradient(xi) for xi in self.pyramid(xreal)]
        xsim = self.generator(self.embedding(batch_size), output="pyramid")
        loss = self.discriminator_loss(xreal, xsim)

        with tf.control_dependencies([train_step.assign_add(1)]):
            opt_step = optimizer.minimize(
                loss, var_list=tdl.core.get_trainable(self.discriminator))
        return SimpleNamespace(
            xreal=xreal[-1],
            xsim=xsim[-1],
            loss=loss,
            train_step=train_step,
            optimizer=optimizer,
            step=opt_step,
            variables=(optimizer.variables() + [train_step]
                       + tdl.core.get_variables(self.discriminator))
        )

    def encoder_trainer(self, xreal, batch_size, optimizer=None):
        tdl.core.assert_initialized(
            self, 'encoder_trainer',
            ['generator', 'discriminator', 'embedding', 'encoder'])
        if optimizer is None:
            optimizer = dict()

        estimator = Estimator(loss=NegLogProb(), model=self.encoder)
        z_samp = self.embedding(batch_size)
        sim_pyramid = self.generator(z_samp, output="pyramid")
        x_sim = sim_pyramid[-1]

        optim = estimator.get_optimizer(x_sim, z_samp, **optimizer)
        return SimpleNamespace(
            estimator=estimator, optim=optim,
            variables=tdl.core.get_variables(self.encoder) +
            optim.optimizer.variables())

    def embedding_trainer(self, batch_size, xreal, optimizer=None,
                          embedding_kl=0.005):
        tdl.core.assert_initialized(
            self, 'embedding_trainer',
            ['generator', 'discriminator', 'pyramid', 'embedding', 'encoder'])

        if optimizer is None:
            optimizer = dict()

        z_sim = self.embedding(batch_size)
        pyramid_sim = self.generator(z_sim, output="pyramid")
        xsim = pyramid_sim[-1]

        zp_sim = self.encoder(xsim)
        zp_real = self.encoder(xreal)

        estimator = Estimator(
            loss=CallWrapper(
                model=EmbeddingLoss(
                    model=self.embedding,
                    reg_scale=embedding_kl,
                ),
                call_fn=lambda model, _, x: model(x[0], x[1])),
            model=CallWrapper(
                model=self.embedding,
                call_fn=lambda model, x: x)
            )
        optim = estimator.get_optimizer([zp_sim, zp_real], None, **optimizer)

        # return
        return SimpleNamespace(
            estimator=estimator, optim=optim,
            variables=(
                optim.optimizer.variables() +
                tdl.core.get_variables(self.embedding))
            )
