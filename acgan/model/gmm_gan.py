import tensorflow as tf
import twodlearn as tdl
import twodlearn.bayesnet
import tensorflow.keras.layers as tf_layers
import tensorflow_probability as tfp
from .gmm import GMM
from .msg_gan import (MSG_GAN, MSG_DiscriminatorTrainer, MSG_GeneratorTrainer,
                      AffineLayer, Conv2DLayer, LEAKY_RATE)
from .base import BaseTrainer


class GmmDiscriminatorTrainer(MSG_DiscriminatorTrainer):
    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['batch_size'])
        sample = self.model.embedding(self.batch_size)
        return sample

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


class GmmEncoderTrainer(BaseTrainer):
    @tdl.core.SubmodelInit
    def optimizer(self, learning_rate, beta1=0.0):
        return {
            'encoder': tf.train.AdamOptimizer(learning_rate, beta1=beta1),
            'embedding': tf.train.AdamOptimizer(learning_rate, beta1=beta1)
            }

    @tdl.core.InputArgument
    def xreal(self, value):
        '''real images obtained from the dataset'''
        tdl.core.assert_initialized(self, 'xreal', ['train_step'])
        # noise_rate = self.model.noise_rate(self.train_step)
        # if noise_rate is not None:
        #     value = _add_noise(value, noise_rate)
        return value

    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['batch_size'])
        sample = self.model.embedding(self.batch_size)
        return sample

    @tdl.core.Submodel
    def sim_pyramid(self, _):
        tdl.core.assert_initialized(self, 'sim_pyramid', ['embedding'])
        return self.generator.pyramid(self.embedding)

    @tdl.core.Submodel
    def xsim(self, _):
        tdl.core.assert_initialized(self, 'xsim', ['sim_pyramid'])
        return self.sim_pyramid[-1]

    @tdl.core.Submodel
    def encoded(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['xsim', 'xreal'])
        return {'xsim': self.model.encoder(self.xsim),
                'xreal': self.model.encoder(self.xreal)}

    def _loss_encoder(self, z, zpred):
        '''loss of the encoder network p(z|x).'''
        return tf.reduce_mean(
            tdl.core.array.reduce_sum_rightmost(
                -zpred.log_prob(z)))

    def _loss_embedding(self, zsim, zreal, embedding_kl=None,
                        use_zsim=True):
        '''loss of the marginal embedding p(z).'''
        # negative likelihood
        if use_zsim:
            z_sample = tf.concat([zsim.sample(), zreal.sample()], axis=0)
        else:
            z_sample = zreal.sample()
        loss = -self.model.embedding.log_prob(z_sample)
        # kl loss
        if embedding_kl is not None:
            embedding = self.model.embedding
            cat = embedding.dist.cat
            n_comp = embedding.n_components
            cat_ref = tfp.distributions.Categorical(logits=n_comp*[1.0])
            kl_loss = tfp.distributions.kl_divergence(cat, cat_ref)
            loss = tf.reduce_mean(loss) + embedding_kl*kl_loss
            # Gaussian means
            normal_ref = tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros([embedding.n_dims]),
                    scale_diag=tf.ones([embedding.n_dims]))
            loc_batch = tf.stack([comp.loc for comp in embedding.components],
                                 axis=0)
            mean_loss = -normal_ref.log_prob(loc_batch)
            loss = loss + embedding_kl * tf.reduce_mean(mean_loss)
        # return
        return loss

    @tdl.core.SubmodelInit
    def loss(self, embedding_kl=None, use_zsim=True):
        tdl.core.assert_initialized(self, 'loss', ['embedding', 'encoded'])
        return {'encoder':
                self._loss_encoder(self.embedding, self.encoded['xsim']),
                'embedding':
                self._loss_embedding(
                    zsim=self.encoded['xsim'],
                    zreal=self.encoded['xreal'],
                    embedding_kl=embedding_kl,
                    use_zsim=use_zsim)}

    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = {
                'encoder':
                self.optimizer['encoder'].minimize(
                    self.loss['encoder'],
                    var_list=tdl.core.get_trainable(self.model.encoder)),
                'embedding':
                self.optimizer['embedding'].minimize(
                    self.loss['embedding'],
                    var_list=tdl.core.get_trainable(self.model.embedding)),
                }
        return step

    @property
    def variables(self):
        tdl.core.assert_initialized(
            self, 'variables', ['optimizer', 'train_step'])
        return (self.optimizer['encoder'].variables() +
                [self.train_step] +
                tdl.core.get_variables(self.model.encoder) +
                self.optimizer['embedding'].variables() +
                tdl.core.get_variables(self.model.embedding))


class GmmGeneratorTrainer(MSG_GeneratorTrainer):
    @tdl.core.Submodel
    def embedding(self, _):
        tdl.core.assert_initialized(self, 'embedding', ['batch_size'])
        sample = self.model.embedding.sample(self.batch_size)
        return sample

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

    @tdl.core.OutputValue
    def step(self, _):
        tdl.core.assert_initialized(
            self, 'step', ['loss', 'optimizer', 'train_step'])
        var_list = list(set(tdl.core.get_trainable(self.generator)))
        with tf.control_dependencies([self.train_step.assign_add(1)]):
            step = self.optimizer.minimize(
                self.loss, var_list=var_list)
        return step


@tdl.core.create_init_docstring
class GmmGan(MSG_GAN):
    '''GAN with a GMM embedding.'''
    EmbeddingModel = GMM
    EncoderModel = tdl.stacked.StackedModel
    EncoderTrainer = GmmEncoderTrainer
    GeneratorTrainer = GmmGeneratorTrainer

    DiscriminatorTrainer = GmmDiscriminatorTrainer

    @tdl.core.SubmodelInit
    def embedding(self, n_components, init_loc=1e-5, init_scale=1.0,
                  min_scale_p=None):
        '''Embedding model P(z)

        Args:
            n_components: number of clusters.
            init_loc: init mean.
            init_scale: init scale.
            min_scale_p: minimum scale (in percentage of maximum).
        '''
        tdl.core.assert_initialized(self, 'embedding', ['embedding_size'])
        model = self.EmbeddingModel(
            n_dims=self.embedding_size,
            n_components=n_components,
            components={'init_loc': init_loc,
                        'init_scale': init_scale})
        return model

    @tdl.core.SubmodelInit
    def encoder(self, units, kernels, strides, dropout=None, padding='same',
                pooling=None):
        '''CNN for recovering the encodding of a given image'''
        n_layers = len(units)
        kernels = self._to_list(kernels, n_layers)
        strides = self._to_list(strides, n_layers)
        padding = (padding if isinstance(padding, (list, tuple))
                   else [padding]*n_layers)
        pooling = self._to_list(pooling, n_layers)

        model = self.EncoderModel()
        for i in range(len(units)):
            # residual
            residual = tdl.stacked.StackedLayers()
            residual.add(Conv2DLayer(
                filters=units[i], strides=strides[i],
                kernel_size=kernels[i], padding=padding[i]))
            residual.add(tf_layers.LeakyReLU(LEAKY_RATE))
            if dropout is not None:
                residual.add(tf_layers.Dropout(rate=dropout))
            # resnet
            model.add(tdl.resnet.ResConv2D(residual=residual))
            if i < len(pooling) and pooling[i] is not None:
                model.add(tf_layers.MaxPooling2D(pool_size=pooling[i]))

        model.add(tf_layers.Flatten())
        model.add(AffineLayer(units=self.embedding_size))
        model.add(tdl.bayesnet.NormalModel(
            loc=lambda x: x,
            batch_shape=self.embedding_size
        ))
        return model

    def discriminator_trainer(self, batch_size, xreal, input_shape=None,
                              optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid',
             'embedding', 'encoder'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.DiscriminatorTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer=optimizer,
            **kwargs)

    def generator_trainer(self, batch_size, optimizer=None, **kwargs):
        tdl.core.assert_initialized(
            self, 'generator_trainer',
            ['generator', 'discriminator', 'pyramid', 'embedding', 'encoder'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.GeneratorTrainer(
            model=self, batch_size=2*batch_size,
            optimizer=optimizer,
            **kwargs)

    def encoder_trainer(self, batch_size, xreal, optimizer=None,
                        **kwargs):
        tdl.core.assert_initialized(
            self, 'discriminator_trainer',
            ['generator', 'discriminator', 'noise_rate', 'pyramid',
             'embedding', 'encoder'])
        if optimizer is None:
            optimizer = {'learning_rate': 0.0002, 'beta1': 0.0}
        return self.EncoderTrainer(
            model=self, batch_size=batch_size, xreal=xreal,
            optimizer=optimizer,
            **kwargs)
