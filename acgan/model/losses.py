import twodlearn as tdl
import tensorflow as tf
import tensorflow_probability as tfp
from twodlearn.core import nest


class DLogisticSimpleGP(tdl.core.Layer):
    """ Simplified gradient penalty

    Presented in https://arxiv.org/pdf/1801.04406.pdf
    Based from https://github.com/NVlabs/stylegan/blob/master/training/loss.py
    """
    r1_gamma = tdl.core.InputArgument.optional(
        'r1_gamma', doc='loss weight', default=10.0)
    r2_gamma = tdl.core.InputArgument.optional(
        'r2_gamma', doc='loss weight', default=10.0)

    discriminator = tdl.core.Submodel.required(
        'discriminator', doc='discriminator model')

    def _to_list(self, value):
        if isinstance(value, list):
            return value
        if isinstance(value, tf.Tensor):
            return [value]
        if nest.is_nested(value):
            return nest.flatten(value)
        raise ValueError(f'{value} not recognized as a valid option.')

    def call(self, x_real, x_sim):
        scores_real = self.discriminator(x_real)
        scores_sim = self.discriminator(x_sim)

        loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                        tf.ones_like(scores_real), scores_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                    tf.zeros_like(scores_sim), scores_sim)
        loss = (loss_real + loss_sim)/2.0

        x_real = self._to_list(x_real)
        x_sim = self._to_list(x_sim)
        if self.r1_gamma != 0.0:
            with tf.name_scope('R1Penalty'):
                real_loss = tf.reduce_sum(scores_real)
                real_grads = tf.gradients(real_loss, x_real)
                r1_penalty = tf.add_n([
                    tf.reduce_sum(tf.square(grads_i), axis=[1, 2, 3])
                    for grads_i in real_grads
                ])/len(real_grads)
            loss += tf.reduce_mean(r1_penalty) * (self.r1_gamma * 0.5)

        if self.r2_gamma != 0.0:
            with tf.name_scope('R2Penalty'):
                fake_loss = tf.reduce_sum(scores_sim)
                fake_grads = tf.gradients(fake_loss, x_sim)
                r2_penalty = tf.add_n(
                    [tf.reduce_sum(tf.square(grads_i), axis=[1, 2, 3])
                     for grads_i in fake_grads])/len(x_sim)
            loss += tf.reduce_mean(r2_penalty) * (self.r2_gamma * 0.5)
        assert loss.shape == tf.TensorShape(())
        return loss


class DLogistic():
    discriminator = tdl.core.Submodel.required(
        'discriminator', doc='discriminator model')

    def call(self, x_real, x_sim):
        pred_real = self.discriminator(x_real)
        pred_sim = self.discriminator(x_sim)
        loss_real = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(pred_real), pred_real)
        loss_sim = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.zeros_like(pred_sim), pred_sim)
        loss = (loss_real + loss_sim)/2.0
        return loss


class NegLogProb(tdl.core.Layer):
    def call(self, labels, predicted):
        return tf.reduce_mean(-predicted.log_prob(labels))


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

            # gaussian scale
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
