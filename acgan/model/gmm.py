import twodlearn as tdl
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


@tdl.core.create_init_docstring
class GMM(tdl.core.layers.Layer):
    n_dims = tdl.core.InputArgument.required(
        'n_dims', doc='dimensions of the GMM model')
    n_components = tdl.core.InputArgument.required(
         'n_components', doc='number of mixture components')

    def get_max_scale(self):
        tdl.core.assert_initialized(
            self, 'components', ['n_components', 'n_dims'])
        return np.power(1.0/self.n_components, 1.0/np.prod(self.n_dims))

    @tdl.core.SubmodelInit(lazzy=True)
    def components(self,
                   init_loc=1e-5, init_scale=1.0,
                   trainable=True, tolerance=1e-5,
                   min_scale_p=None,
                   constrained_loc=False):
        tdl.core.assert_initialized(
            self, 'components', ['n_components', 'n_dims'])
        max_scale = self.get_max_scale()
        if min_scale_p:
            min_scale = tolerance + min_scale_p*max_scale
        else:
            min_scale = tolerance

        def batch_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))[
                ..., tf.newaxis]
        if constrained_loc:
            loc = tf.Variable(
                  tf.truncated_normal(
                      shape=[self.n_components, self.n_dims],
                      mean=init_loc),
                  constraint=lambda weights: tf.where(
                      tf.broadcast_to(batch_norm(weights) > 1.0,
                                      [self.n_components, self.n_dims]),
                      x=weights/batch_norm(weights),
                      y=weights
                  ),
                  trainable=trainable)
        else:
            loc = tf.Variable(
                  tf.truncated_normal(
                      shape=[self.n_components, self.n_dims],
                      mean=init_loc),
                  trainable=trainable)
        components = tdl.core.SimpleNamespace(
            loc=loc,
            scale=tdl.constrained.ConstrainedVariable(
                0.9*max_scale*tf.ones([self.n_components, self.n_dims]),
                min=min_scale,
                max=max_scale))
        return components

    @tdl.core.SubmodelInit(lazzy=True)
    def logits(self, trainable=True, tolerance=1e-5):
        tdl.core.assert_initialized(self, 'logits', ['n_components'])
        return tf.Variable(tf.zeros(self.n_components),
                           trainable=trainable)

    def get_components(self, loc_trainable=True, scale_trainable=True):
        def get_trainable(x, trainable):
            x = tf.convert_to_tensor(x)
            return x if trainable else tf.stop_gradient(x)
        return [tfp.distributions.MultivariateNormalDiag(
                    loc=get_trainable(
                        self.components.loc, loc_trainable)[idx, ...],
                    scale_diag=get_trainable(
                        self.components.scale, scale_trainable)[idx, ...])
                for idx in range(self.n_components)]

    @tdl.core.SubmodelInit
    def dist(self, mix=None):
        tdl.core.assert_initialized(self, 'dist', ['components', 'logits'])
        bimix_gauss = tfp.distributions.Mixture(
            cat=tfp.distributions.Categorical(logits=self.logits),
            components=[
                tfp.distributions.MultivariateNormalDiag(
                    loc=tf.convert_to_tensor(
                        self.components.loc)[idx, ...],
                    scale_diag=tf.convert_to_tensor(
                        self.components.scale)[idx, ...])
                for idx in range(self.n_components)])
        return bimix_gauss

    @tdl.core.SubmodelInit(lazzy=True)
    def init_op(self):
        tdl.core.assert_initialized(self, 'dist', ['components', 'logits'])
        # update variables as building has not finished
        self._update_variables()
        return tf.group([var.initializer for var in
                         tdl.core.get_trainable(self)])

    @tdl.core.layers.build_wrapper
    def sample(self, sample_shape):
        return self.dist.sample(sample_shape)

    @tdl.core.layers.build_wrapper
    def log_prob(self, inputs):
        return self.dist.log_prob(inputs)

    def call(self, inputs=None):
        return self.sample(inputs)
