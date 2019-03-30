import acgan
import unittest
import tensorflow as tf
import twodlearn as tdl


class DcganTest(unittest.TestCase):
    def test_disc_loss(self):
        tf.random.set_random_seed(0)
        BATCH_SIZE = 128
        dataset = acgan.data.load_celeb_a(BATCH_SIZE)
        model = acgan.model.DCGAN(
            embedding_size=256,
            generator={'init_shape': (4, 4, 1024),
                       'units': [516, 256, 128, 3],
                       'kernels': 5,
                       'strides': 2,
                       'padding': ['same', 'same', 'same', 'same']},
            discriminator={'units': [128, 256, 512, 1024],
                           'kernels': 5,
                           'strides': 2,
                           'dropout': None})
        # model.noise_rate.init(rate=0.001)
        generator_shape = model.generator.compute_output_shape(
            input_shape=[None, 100])
        assert (generator_shape.as_list()
                == tf.TensorShape([None, 64, 64, 3]).as_list()),\
            'generator shape does not match the expected shape'
        # losses
        iter = dataset.make_one_shot_iterator()
        xreal = iter.get_next()

        gen = model.generator_trainer(BATCH_SIZE, learning_rate=0.0002)
        dis = model.discriminator_trainer(BATCH_SIZE, xreal=xreal,
                                          learning_rate=0.0002)
        with tf.Session().as_default():
            tdl.core.variables_initializer(gen.variables).run()
            tdl.core.variables_initializer(dis.variables).run()
            self.assertAlmostEqual(dis.loss.eval(), 0.693, places=2)
            assert len(tdl.core.get_variables(model.generator)) == 17
            assert len(tdl.core.get_variables(model.discriminator)) == 26
            assert (set(tdl.core.get_variables(model.generator)) &
                    set(tdl.core.get_variables(model.discriminator))) == set()


if __name__ == "__main__":
    unittest.main()
