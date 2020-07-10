import acgan
import unittest
import tensorflow as tf
import acgan.model.base


class DcganTest(unittest.TestCase):
    def test_minibatchstddev(self):
        with tf.Session().as_default():
            layer = acgan.model.base.MinibatchStddev()
            inputs = tf.random.normal([64, 28, 28, 3])
            output = layer(inputs)
            assert output.shape.as_list() == [64, 28, 28, 4]
            assert (tf.reduce_sum(tf.cast(tf.equal(
                output, output[0, 0, 0, 3]), tf.int32)).eval() ==
                    64*28*28)
            assert (output.shape.as_list() ==
                    layer.compute_output_shape(inputs.shape).as_list())

    def test_conv2d(self):
        layer = acgan.model.msg_gan.Conv2DLayer(
            kernel_size=[3, 3], use_bias=False, padding='same')
        inputs = tf.random.normal([64, 28, 28, 3])
        output = layer(inputs)
        assert layer.bias is None
        assert output.shape.as_list() == [64, 28, 28, 3]


if __name__ == "__main__":
    unittest.main()
