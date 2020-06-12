import tqdm
import numpy as np
import tensorflow as tf


def _as_numpy(tensor):
    if tf.executing_eagerly():
        return tensor.numpy()
    else:
        return tensor.eval()


def _get_session(session=None):
    session = (session if session is not None
               else tf.compat.v1.get_default_session()
               if tf.compat.v1.get_default_session() is not None
               else tf.InteractiveSession())
    return session


def _compute_split_scores(split):
    kl = split * (np.log(split) - np.log(np.expand_dims(np.mean(split, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    return np.exp(kl)


def _preprocess(imgs, scale=False):
    imgs = tf.image.resize(imgs, (299, 299))
    if scale:
        imgs = tf.keras.applications.inception_v3.preprocess_input(imgs)
    return imgs


def _get_inception_prob(imgs, scale):
    inception = tf.keras.applications.InceptionV3()
    imgs = _preprocess(imgs, scale=scale)
    return inception(imgs)


def inception_score(imgs, scale=False, max_eval=50000, splits=10):
    prob_h = _get_inception_prob(imgs, scale=scale)
    probs = [prob_h.eval()]
    with tqdm.tqdm(total=max_eval) as pbar:
        while sum((batch.shape[0] for batch in probs)) < max_eval:
            probs.append(prob_h.eval())
            pbar.update(probs[-1].shape[0])

    probs = np.concatenate(probs, 0)
    split_size = probs.shape[0] // splits
    scores = [_compute_split_scores(probs[i*split_size:(i+1)*split_size])
              for i in range(splits)]
    return np.mean(scores), np.std(scores)
