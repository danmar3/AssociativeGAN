import acgan
import unittest
import numpy as np
import tensorflow as tf


class CheckTF(unittest.TestCase):
    def test_check_gpu(self):
        r = [[1.0, 0.0],    [0.0, 1.0]]
        x, y = np.meshgrid(list(range(400)), list(range(400)))
        coords = np.stack([x, y], -1).reshape((400, 400, 2, 1))
        coords = tf.convert_to_tensor(coords, dtype=tf.float32)

        r1 = tf.constant(r)

        newCoords = tf.matmul(r1, coords)

        sess = tf.Session()
        ret = sess.run(newCoords, feed_dict={r1: r})
        ret.sum() == 63840000.0


if __name__ == "__main__":
    unittest.main()
