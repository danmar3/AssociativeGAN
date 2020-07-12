import twodlearn as tdl
from twodlearn.core import nest
import tensorflow as tf
import functools


def eager_function(func):
    @functools.wraps(func)
    def wrapper(self, **kwargs):
        if not hasattr(self, '__tdl__'):
            setattr(self, '__tdl__', tdl.core.common.__TDL__(self))
        if not hasattr(self.__tdl__, func.__name__):
            tf_kwargs = {key: tf.compat.v1.placeholder(
                shape=[None] + [i for i in value.shape[1:]],
                dtype=value.dtype)
                         for key, value in kwargs.items()}
            out = func(self, **tf_kwargs)
            setattr(self.__tdl__, func.__name__, {
                'out': out, 'kwargs': tf_kwargs})
        tf_nodes = getattr(self.__tdl__, func.__name__)
        session = tf.compat.v1.get_default_session()
        feed_dict = {tf_nodes['kwargs'][key]: value
                     for key, value in kwargs.items()}
        out_ = session.run(tf_nodes['out'], feed_dict=feed_dict)
        return nest.pack_sequence_as(
            out_, [out_i for out_i in nest.flatten(out_)
                   if out_i is not None])
    return wrapper


def replicate_to_list(value, n_elements):
    '''check if value is a list. If not, return a list with n_elements.

    If value is an integer, each element is a 2-dim tuple (value, value).
    If value is an iterable, each element of the new list is a tuple
    with duplicated elements (value[i], value[i]) '''
    if isinstance(value, int):
        value = [value]*n_elements
    if isinstance(value[0], int) or (value[0] is None):
        value = [[vi, vi] for vi in value]
    assert len(value) == n_elements, \
        'list does not have the expected number of elements'
    assert all([len(vi) == 2 for vi in value]), \
        'list does not have the expected number of elements'
    return value
