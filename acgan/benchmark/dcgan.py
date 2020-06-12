import os
import tensorflow_gan as tfgan
import acgan
import tensorflow as tf
from tensorflow_gan.examples.cifar import networks

PARAMS = {
    'embedding_size': 64,
    'batch_size': 32,
    'generator_lr': 0.0002,
    'discriminator_lr': 0.0002,
    'master': '',
    'train_log_dir': '/tmp/tfgan_logdir/cifar',
    'ps_replicas': 0,
    'task': 0
}


def _get_session(session=None):
    session = (session if session is not None
               else tf.compat.v1.get_default_session()
               if tf.compat.v1.get_default_session() is not None
               else tf.InteractiveSession())
    return session


class DCGAN(object):
    @staticmethod
    def load_generator(checkpoint=None, params=PARAMS):
        num_images_generated = 32
        generator_inputs = tf.random.normal(
            [num_images_generated, params['embedding_size']])
        generator_fn = networks.generator
        # In order for variables to load, use the same variable scope as in the
        # train job.
        with tf.compat.v1.variable_scope('Generator'):
            x_sim = generator_fn(generator_inputs, is_training=False)

        session = _get_session()
        session.run(tf.variables_initializer(tf.global_variables()))
        # saver = tf.compat.v1.train.Saver(gan_model.generator_variables)
        saver = tf.compat.v1.train.Saver(tf.global_variables())
        saver.restore(session, checkpoint)
        return x_sim

    def __init__(self, images, params=PARAMS):
        self.params = params

        if not tf.io.gfile.exists(self.params['train_log_dir']):
            tf.io.gfile.makedirs(self.params['train_log_dir'])

        generator_inputs = tf.random.normal(
            [self.params['batch_size'], self.params['embedding_size']])
        self.model = tfgan.gan_model(
            networks.generator,
            networks.discriminator,
            real_data=images,
            generator_inputs=generator_inputs)

    def _get_train_ops(self):
        tfgan.eval.add_gan_model_image_summaries(self.model)

        # Get the GANLoss tuple. Use the selected GAN loss functions.
        with tf.compat.v1.name_scope('loss'):
            gan_loss = tfgan.gan_loss(
                self.model, gradient_penalty_weight=1.0, add_summaries=True)

        # Get the GANTrain ops using the custom optimizers and optional
        # discriminator weight clipping.
        with tf.compat.v1.name_scope('train'):
            gen_opt = tf.compat.v1.train.AdamOptimizer(
                self.params['generator_lr'], 0.5)
            dis_opt = tf.compat.v1.train.AdamOptimizer(
                self.params['discriminator_lr'], 0.5)

            train_ops = tfgan.gan_train_ops(
                self.model,
                gan_loss,
                generator_optimizer=gen_opt,
                discriminator_optimizer=dis_opt,
                summarize_gradients=True)
        return train_ops

    def run_training(self, max_number_of_steps=1000000):
        if not hasattr(self, '_train_ops'):
            self._train_ops = self._get_train_ops()
        status_message = tf.strings.join(
            ['Starting train step: ',
             tf.as_string(tf.compat.v1.train.get_or_create_global_step())],
            name='status_message')

        tfgan.gan_train(
            self._train_ops,
            hooks=([
                tf.estimator.StopAtStepHook(num_steps=max_number_of_steps),
                tf.estimator.LoggingTensorHook(
                    [status_message], every_n_iter=10)
            ]),
            logdir=self.params['train_log_dir'],
            master=self.params['master'],
            is_chief=self.params['task'] == 0)


def load_model(checkpoint):
    pass
