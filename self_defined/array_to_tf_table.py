import tensorflow as tf
import numpy as np


class ArrayToTableTFFn:

    def __init__(self, writer, header, scope, title, names, precision=None):

        self.writer = writer
        self.header = header
        self.names = names
        self.precision = precision
        self.title = title
        self.scope = scope

    def write(self, tf_array, tf_step):
        # tf_array is a tensor or np.ndarray
        # tf_step is an int tensor or python int
        assert isinstance(tf_array, (tf.Tensor, np.ndarray, list, tuple))
        if not isinstance(tf_array, tf.Tensor):
            tf_array = tf.convert_to_tensor(tf_array)
        assert isinstance(tf_step, (tf.Tensor, int))

        header = self.header
        names = self.names
        precision = self.precision
        title = self.title
        scope = self.scope

        assert tf_array.dtype in (tf.int32, tf.int64, tf.float32, tf.float64)
        assert tf_array.ndim == 2
        num_examples, num_fields = tf_array.shape
        assert num_examples is not None
        assert num_fields is not None
        assert isinstance(header, (list, tuple))
        if isinstance(header, tuple):
            header = list(header)
        assert len(header) == num_fields
        header = ['id', 'name'] + header
        header = tf.constant(header)
        assert isinstance(names, list)
        assert len(names) == num_examples
        names = tf.constant(names)[:, None]
        assert names.dtype == tf.string
        ids = [str(i) for i in range(1, num_examples + 1)]
        ids = tf.constant(ids)[:, None]
        if precision is None:
            if tf_array.dtype in (tf.float32, tf.float64):
                precision = 4
            else:
                precision = -1
        tf_array = tf.as_string(tf_array, precision=precision)
        tf_array = tf.concat([ids, names, tf_array], axis=1)
        tf_array.set_shape([num_examples, num_fields + 2])
        tf_array = tf.strings.reduce_join(tf_array, axis=1, separator=' | ')
        tf_array = tf.strings.reduce_join(tf_array, separator='\n')
        header = tf.strings.reduce_join(header, separator=' | ')
        sep = tf.constant(['---'])
        sep = tf.tile(sep, [num_fields + 2])
        sep = tf.strings.reduce_join(sep, separator=' | ')
        tf_array = tf.strings.join([header, sep, tf_array], separator='\n')
        assert isinstance(title, str)
        tf_array = tf.strings.join([tf.constant(title), tf_array], separator='\n\n')
        assert isinstance(scope, str)
        # assert tf_step.dtype == tf.int64
        with self.writer.as_default():
            tf.summary.text(scope, tf_array, step=tf_step)
