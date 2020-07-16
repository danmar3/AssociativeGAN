import twodlearn as tdl
import tensorflow as tf
from twodlearn.core import nest


class LogisticSimpleGP(tdl.core.Layer):
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


class Logistic():
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
