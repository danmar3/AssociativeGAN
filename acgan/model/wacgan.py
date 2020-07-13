import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow.keras.layers as tf_layers
import tensorflow_probability as tfp
from .msg_gan import (
    Conv1x1Proj, Conv2DLayer, AffineLayer, MinibatchStddev,
    USE_BIAS, LEAKY_RATE)
from .gmm_gan import GmmGan, GmmGeneratorTrainer
from .encoder import ResConv, Estimator, CallWrapper
from ..utils import replicate_to_list
from types import SimpleNamespace


class WacGanGeneratorTrainer(GmmGeneratorTrainer):
    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        var_list = list(set(tdl.core.get_trainable(self.generator)) |
                        set(tdl.core.get_trainable(self.model.embedding)))
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=var_list)
        return step


@tdl.core.create_init_docstring
class WacGan(GmmGan):
    GeneratorTrainer = WacGanGeneratorTrainer

    @tdl.core.SubmodelInit
    def discriminator(self, units, kernels, strides, dropout=None,
                      padding='same'):
        n_layers = len(units)
        kernels = replicate_to_list(kernels, n_layers)
        strides = replicate_to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = self.DiscriminatorBaseModel()
        for i in range(len(units)):
            model.add(self.DiscriminatorHidden(
                units=units[i], kernels=kernels[i], strides=strides[i],
                padding=padding[i], dropout=dropout))
        model.add(self.DiscriminatorOutput())
        return model


class WacGanV2(WacGan):
    EncoderModel = ResConv
    @tdl.core.SubmodelInit
    def encoder(self, units, kernels, strides, dropout=None, padding='same',
                pooling=None, flatten='global_maxpool', res_layers=2):
        '''CNN for recovering the encodding of a given image'''
        return self.EncoderModel(
            embedding_size=self.embedding_size,
            layers={'units': units, 'kernels': kernels, 'strides': strides,
                    'dropout': dropout, 'padding': padding, 'pooling': pooling,
                    'flatten': flatten, 'res_layers': res_layers})


class NegLogProb(tdl.core.Layer):
    def call(self, labels, predicted):
        return tf.reduce_mean(
            tdl.core.array.reduce_sum_rightmost(
                -predicted.log_prob(labels)))


def scale_loss(comp, ref):
    loss_tr = ref.scale.solve(comp.scale.to_dense())
    # frobenius
    loss_tr = 0.5*tf.reduce_sum(tf.square(loss_tr), axis=[-2, -1])
    loss_det = (ref.scale.log_abs_determinant()
                - comp.scale.log_abs_determinant())
    return loss_tr + loss_det


class EmbeddingLoss(tdl.core.Layer):
    model = tdl.core.Submodel.required('model', doc='model.')
    reg_scale = tdl.core.InputArgument.required(
        'reg_scale', doc='scale regularization loss.')
    linear_disc = tdl.core.Submodel.required(
        'linear_disc', doc='linear discriminant to improve sparsity.')

    @tdl.core.Submodel
    def sparsity(self, value):
        'weight for sparcity loss. Ussually 1/0 for enable/disable it'
        if value is None:
            value = tf.compat.v1.placeholder(tf.float32, shape=())
        return value

    def _loss_linear_disc(self, loc_trainable=True, scale_trainable=True):
        samp = tf.stack([
            comp.sample() for comp in self.model.get_components(
                loc_trainable=loc_trainable,
                scale_trainable=scale_trainable)],
                        axis=0)
        labels = tf.one_hot(list(range(self.model.n_components)),
                            depth=self.model.n_components)
        loss = tf.keras.losses.categorical_crossentropy(
            y_true=labels,
            y_pred=self.linear_disc(samp),
            from_logits=True
        )
        return tf.reduce_mean(loss)

    def call(self, zsim, zreal):
        z_sample = tf.concat(
                [tf.stop_gradient(zsim.sample()),
                 tf.stop_gradient(zreal.sample())],
                axis=0)
        loss = -self.model.log_prob(z_sample)

        if self.reg_scale is not None:
            cat = self.model.dist.cat
            n_comp = self.model.n_components
            cat_ref = tfp.distributions.Categorical(logits=n_comp*[1.0])
            cat_loss = tfp.distributions.kl_divergence(cat, cat_ref)
            loss = tf.reduce_mean(loss) + self.reg_scale*cat_loss
            # gaussian means
            sparse_loss = self._loss_linear_disc(
                loc_trainable=True, scale_trainable=False)
            loss = (loss + self.sparsity * self.reg_scale * tf.reduce_mean(sparse_loss))

            ref_scale = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros([self.model.n_dims]),
                scale_diag=tf.constant(
                    value=self.model.get_max_scale(),
                    shape=[self.model.n_dims],
                    dtype=tf.float32))
            comp_loss = [scale_loss(comp, ref_scale)
                         for comp in self.model.dist.components]
            loss = loss + self.reg_scale * tf.add_n(comp_loss)
        return loss


class Conv2DBlock(tdl.core.Layer):
    use_bias = tdl.core.InputArgument.optional(
        'use_bias', doc='use bias', default=True)

    leaky_rate = tdl.core.InputArgument.optional(
        'leaky_rate', doc='leaky rate', default=0.2)

    @tdl.core.SubmodelInit(lazzy=True)
    def conv(self, filters=None, kernels=3, padding='same', BatchNorm=None):
        tdl.core.assert_initialized(self, 'conv', ['use_bias', 'leaky_rate'])
        if filters is None:
            filters = [None, None]
        kernels = replicate_to_list(kernels, len(filters))
        layers = tdl.stacked.StackedLayers()
        for idx in range(len(filters)):
            kargs = {'filters': filters[idx]} if filters[idx] else {}
            layers.add(
                Conv2DLayer(
                    kernel_size=kernels[idx], strides=[1, 1],
                    padding=padding,
                    use_bias=self.use_bias,
                    **kargs))
            if BatchNorm:
                layers.add(BatchNorm())
            layers.add(
                tf_layers.LeakyReLU(self.leaky_rate))
        return layers

    @tdl.core.SubmodelInit(lazzy=True)
    def pooling(self, size=None, method="average"):
        if size is not None:
            if method == "average":
                return tf.keras.layers.AveragePooling2D(pool_size=size)
            else:
                ValueError(f'pooling method {method} not recognized.')
        else:
            return None

    @tdl.core.SubmodelInit(lazzy=True)
    def dropout(self, rate=None):
        if rate is not None:
            return tf_layers.Dropout(rate)
        else:
            return None

    def compute_output_shape(self, input_shape):
        tdl.core.assert_initialized(
            self, 'compute_output_shape', ['conv', 'pooling', 'dropout'])
        output_shape = self.conv.compute_output_shape(input_shape)
        if self.pooling:
            output_shape = self.pooling.compute_output_shape(output_shape)
        if self.dropout:
            output_shape = self.dropout.compute_output_shape(output_shape)
        return output_shape

    def call(self, inputs, **kargs):
        value = self.conv(inputs)
        if self.pooling:
            value = self.pooling(value)
        if self.dropout:
            value = self.dropout(value)
        return value


class Discriminator(tdl.core.Layer):
    units = tdl.core.InputArgument.required(
        'units', doc='number of units in each hidden block of the network')

    @tdl.core.LazzyProperty
    def hidden_shapes(self):
        tdl.core.assert_initialized(
            self, 'hidden_shapes', ['input_shape', 'projections'])
        pyramid_shape = self.input_shape
        input_shapes = pyramid_shape[::-1]
        output_list = list()
        proj_shape = self.projections[0].compute_output_shape(input_shapes[0])
        output_shape = self.hidden[0].compute_output_shape(proj_shape)
        output_list.append(output_shape)
        for x_shape, projection, layer in\
                zip(input_shapes[1:], self.projections[1:], self.hidden[1:]):
            proj_shape = projection.compute_output_shape(x_shape)
            extended = proj_shape[:-1].concatenate(
                proj_shape[-1].value + output_shape[-1].value)
            output_shape = layer.compute_output_shape(extended)
            output_list.append(output_shape)
        return output_list

    @tdl.core.SubmodelInit(lazzy=True)
    def projections(self, init_units=None, activation=None):
        tdl.core.assert_initialized(
            self, 'projections', ['units', 'hidden', 'input_shape'])
        projections = list()
        if init_units is None:
            init_units = self.units[0]//2
        units_list = [init_units] + [ui for ui in self.units]
        for units in units_list:
            projections.append(
                Conv1x1Proj(
                    units=units,
                    activation=activation,
                    use_bias=USE_BIAS['discriminator']
                ))
        return projections

    @tdl.core.SubmodelInit(lazzy=True)
    def hidden(self, kernels, strides, dropout=None, padding='same'):
        tdl.core.assert_initialized(self, 'hidden', ['units'])
        n_layers = len(self.units)
        kernels = replicate_to_list(kernels, n_layers)
        strides = replicate_to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)

        model = list()
        for i in range(len(self.units)):
            model.append(Conv2DBlock(
                conv={'filters': [self.units[i], self.units[i]],
                      'kernels': kernels[i],
                      'padding': padding[i]},
                pooling={'size': strides[i]},
                dropout={'rate': dropout},
                use_bias=USE_BIAS['discriminator'],
                leaky_rate=LEAKY_RATE))
        model.append(MinibatchStddev())
        model.append(Conv2DBlock(
            conv={'kernels': [3, 4], 'padding': 'same'},
            use_bias=USE_BIAS['discriminator'],
            leaky_rate=LEAKY_RATE))
        return model

    @tdl.core.SubmodelInit(lazzy=True)
    def dense(self, **kargs):
        return tdl.stacked.StackedLayers(layers=[
            tf_layers.Flatten(),
            AffineLayer(units=1)])

    def call(self, inputs, output='logits'):
        '''the call expects a full pyramid.'''
        tdl.core.assert_initialized(self, 'call', ['projections'])
        depth = len(inputs)
        inputs = inputs[::-1]
        assert depth == len(self.projections),\
            'pyramid size does not match the number of projections'
        proj = self.projections[0](inputs[0])
        out = self.hidden[0](proj)
        for x_i, projection, layer in\
                zip(inputs[1:], self.projections[1:], self.hidden[1:]):
            proj = projection(x_i)
            extended = tf.concat([proj, out], axis=-1)
            out = layer(extended)
        if output == 'hidden':
            return out
        elif output == 'logits':
            return self.dense(out)
        elif output == 'prob':
            return tf.nn.sigmoid(self.dense(out))
        else:
            raise ValueError(f'output {output} not a valid option.')


class WacGanV3(WacGanV2):
    @tdl.core.SubmodelInit
    def discriminator(self, units, kernels, strides, dropout=None,
                      padding='same'):
        return Discriminator(
            units=units,
            hidden={'kernels': kernels, 'strides': strides,
                    'dropout': dropout, 'padding': padding})

    def encoder_trainer(self, batch_size, xreal, optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'encoder_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid',
             'embedding', 'encoder'])
        if optimizer is None:
            optimizer = dict()

        estimator = Estimator(loss=NegLogProb(), model=self.encoder)
        z_samp = self.embedding(batch_size)
        sim_pyramid = self.generator.pyramid(z_samp)
        x_sim = sim_pyramid[-1]

        optim = estimator.get_optimizer(x_sim, z_samp, **optimizer)
        return SimpleNamespace(
            estimator=estimator, optim=optim,
            variables=tdl.core.get_variables(self.encoder) +
            optim.optimizer.variables())

    def embedding_trainer(self, batch_size, xreal, optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'embedding_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid',
             'embedding', 'encoder', 'linear_disc'])

        if optimizer is None:
            optimizer = dict()

        z_sim = self.embedding(batch_size)
        pyramid_sim = self.generator.pyramid(z_sim)
        xsim = pyramid_sim[-1]

        zp_sim = self.encoder(xsim)
        zp_real = self.encoder(xreal)

        estimator = Estimator(
            loss=CallWrapper(
                model=EmbeddingLoss(
                    model=self.embedding,
                    reg_scale=kwargs['loss']['embedding_kl'],
                    linear_disc=self.linear_disc
                ),
                call_fn=lambda model, _, x: model(x[0], x[1])),
            model=CallWrapper(
                model=self.embedding,
                call_fn=lambda model, x: x)
            )
        optim = estimator.get_optimizer([zp_sim, zp_real], None, **optimizer)
        # ----- linear disc -----
        loc_trainable = True
        scale_trainable = True
        samp = tf.stack([
            comp.sample() for comp in self.embedding.get_components(
                loc_trainable=loc_trainable,
                scale_trainable=scale_trainable)],
                        axis=0)
        labels = tf.one_hot(list(range(self.embedding.n_components)),
                            depth=self.embedding.n_components)
        linear_est = Estimator(
            model=self.linear_disc,
            loss={'value': 'crossentropy', 'from_logits': True},
            trainable_variables={
                'get_trainable':
                lambda: tdl.core.get_trainable(self.linear_disc)}
        )
        linear_optim = linear_est.get_optimizer(samp, labels, **optimizer)
        # return
        return SimpleNamespace(
            gmm={'estimator': estimator, 'optim': optim},
            linear={'estimator': linear_est, 'optim': linear_optim},
            variables=optim.optimizer.variables() +
            tdl.core.get_variables(self.embedding) +
            linear_optim.optimizer.variables() +
            tdl.core.get_variables(self.linear_disc)
            )


WacGanDev = WacGanV3
