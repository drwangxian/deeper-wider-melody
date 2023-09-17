import tensorflow as tf
from self_defined.get_name_scope import get_name_scope


def bn_relu_drop_fn(inputs, drop_rate):
    outputs = inputs
    name = get_name_scope()
    outputs = tf.keras.layers.BatchNormalization(
        name=name + 'bn',
        scale=False
    )(outputs)
    outputs = tf.keras.layers.ReLU(
        name=name + 'relu'
    )(outputs)
    if drop_rate > 0:
        outputs = tf.keras.layers.Dropout(
            name=name + 'dropout',
            rate=drop_rate
        )(outputs)

    return outputs