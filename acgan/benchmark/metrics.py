import tqdm
import inspect
import functools
import numpy as np
import tensorflow as tf
from tensorflow_gan.python.eval import (
    classifier_metrics, classifier_score_from_logits)
from tensorflow_gan.python.eval.inception_metrics import (
    run_inception, classifier_fn_from_tfhub,
    INCEPTION_TFHUB, INCEPTION_OUTPUT)

from ..main import eager_function

inception_score = functools.partial(
    classifier_metrics.classifier_score,
    classifier_fn=classifier_fn_from_tfhub(
        INCEPTION_TFHUB, INCEPTION_OUTPUT, True))


def _get_session(session=None):
    session = (session if session is not None
               else tf.compat.v1.get_default_session()
               if tf.compat.v1.get_default_session() is not None
               else tf.InteractiveSession())
    return session


def tensor_to_generator(x_tensor):
    ''' convert a tf.Tensor to a generator.'''
    while True:
        yield x_tensor.eval()


class InceptionScore(object):
    def __init__(self):
        self.logits_h = tf.compat.v1.placeholder(tf.float32)
        self.score_h = classifier_score_from_logits(self.logits_h)

    def _run_classifier_score_from_logits(self, logits):
        return self.score_h.eval(feed_dict={self.logits_h: logits})

    @eager_function
    def _eval_inception(self, inputs):
        return run_inception(inputs)['logits']

    def run(self, imgs, max_eval=50000, splits=10):
        """ Evaluate inception score over a set of images
        Args:
            imgs: a generator of images in the format of batched numpy arrays.
            max_eval: number of images to evaluate.
            splits: number of splits used to evaluate standard deviation of the
              metric.
        """
        if isinstance(imgs, tf.Tensor):
            imgs = tensor_to_generator(imgs)
        assert inspect.isgenerator(imgs), "imgs should be a generator."
        logits = list()
        with tqdm.tqdm(total=max_eval) as pbar:
            for next_batch in imgs:
                logits.append(self._eval_inception(inputs=next_batch))
                pbar.update(logits[-1].shape[0])
                if sum((batch.shape[0] for batch in logits)) >= max_eval:
                    break

        logits = np.concatenate(logits, 0)
        split_size = logits.shape[0] // splits
        scores = [
            self._run_classifier_score_from_logits(
                logits[i*split_size:(i+1)*split_size])
            for i in range(splits)]
        return np.mean(scores), np.std(scores)
