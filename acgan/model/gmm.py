import twodlearn as tdl
import tensorflow as tf
import tensorflow_probability as tfp


@tdl.core.create_init_docstring
class GMM(tdl.core.layers.Layer):
    n_dims = tdl.core.InputArgument.required(
        'n_dims', doc='dimensions of the GMM model')
    n_components = tdl.core.InputArgument.required(
         'n_components', doc='number of mixture components')

    @tdl.core.SubmodelInit(lazzy=True)
    def components(self,
                   init_loc=1e-5, init_scale=1.0,
                   trainable=True, tolerance=1e-5):
        tdl.core.assert_initialized(
            self, 'components', ['n_components', 'n_dims'])
        components = [
            tdl.core.SimpleNamespace(
              loc=tf.Variable(
                  tf.truncated_normal(shape=[self.n_dims], mean=init_loc),
                  trainable=trainable),
              scale=tdl.constrained.PositiveVariable(
                  init_scale*tf.ones([self.n_dims]),
                  tolerance=tolerance,
                  trainable=trainable))
            for k in range(self.n_components)]
        return components

    @tdl.core.SubmodelInit(lazzy=True)
    def logits(self, trainable=True, tolerance=1e-5):
        tdl.core.assert_initialized(self, 'logits', ['n_components'])
        return tf.Variable(tf.zeros(self.n_components),
                           trainable=trainable)

    @tdl.core.SubmodelInit
    def dist(self, mix=None):
        tdl.core.assert_initialized(self, 'dist', ['components', 'logits'])
        bimix_gauss = tfp.distributions.Mixture(
            cat=tfp.distributions.Categorical(logits=self.logits),
            components=[
                tfp.distributions.MultivariateNormalDiag(
                    loc=comp.loc, scale_diag=tf.convert_to_tensor(comp.scale))
                for comp in self.components])
        return bimix_gauss

    @tdl.core.layers.build_wrapper
    def sample(self, sample_shape):
        return self.dist.sample(sample_shape)

    @tdl.core.layers.build_wrapper
    def log_prob(self, inputs):
        return self.dist.log_prob(inputs)

    def call(self, inputs=None):
        return self.sample(inputs)
