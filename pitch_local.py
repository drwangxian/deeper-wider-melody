# pitch local context

import tensorflow as tf
from self_defined import get_name_scope
from lstm_sub import VocalPitchLSTM


def bn_relu_fn(inputs, dropping_prob=None, training=None):

    assert dropping_prob is not None
    assert dropping_prob >= 0
    assert dropping_prob < 1

    assert training is None or training is False

    assert get_name_scope() != ''

    outputs = inputs

    if training is False:
        outputs = tf.keras.layers.BatchNormalization(
            name=get_name_scope() + 'bn',
            scale=False
        )(outputs, training=False)
    else:
        outputs = tf.keras.layers.BatchNormalization(
            name=get_name_scope() + 'bn',
            scale=False
        )(outputs)

    outputs = tf.keras.layers.ReLU(
        name=get_name_scope() + 'relu'
    )(outputs)

    if dropping_prob > 0:
        outputs = tf.keras.layers.Dropout(
            name=get_name_scope() + 'drop',
            rate=dropping_prob
        )(outputs)

    return outputs


def create_acoustic_model_fn():

    dropping_prob = .2
    inputs = tf.keras.Input([None, 500], batch_size=1, name='vqt')
    outputs = inputs
    outputs = outputs[..., None]

    with tf.name_scope('local'):
        for layer_idx in range(6):
            with tf.name_scope('layer_{}'.format(layer_idx)):
                outputs = tf.keras.layers.Conv2D(
                    name=get_name_scope() + 'conv',
                    kernel_size=[3, 3],
                    dilation_rate=[1, 1] if layer_idx < 2 else [2 ** (layer_idx - 1), 2 ** (layer_idx - 1)],
                    padding='SAME',
                    use_bias=False,
                    filters=32
                )(outputs)
                outputs.set_shape([None, None, 500, None])
                outputs = bn_relu_fn(
                    outputs,
                    dropping_prob=0 if layer_idx == 0 else dropping_prob,
                    training=False
                )

    with tf.name_scope('global'):
        with tf.name_scope('conv'):
            outputs = tf.pad(outputs, [[0, 0], [0, 0], [240, 60], [0, 0]])
            outputs = tf.keras.layers.Conv2D(
                name=get_name_scope() + 'conv',
                kernel_size=[1, 33],
                dilation_rate=[1, 15],
                padding='VALID',
                use_bias=False,
                filters=128
            )(outputs)
            outputs.set_shape([None, None, 320, None])
            outputs = bn_relu_fn(outputs, dropping_prob=dropping_prob, training=False)

        with tf.name_scope('fusion'):
            outputs = tf.keras.layers.Dense(
                name=get_name_scope() + 'dense',
                units=64,
                use_bias=False
            )(outputs)
            outputs = bn_relu_fn(outputs, dropping_prob=dropping_prob, training=False)

    with tf.name_scope('pitch_local'):
        for layer_idx in range(6):
            with tf.name_scope('layer_{}'.format(layer_idx)):
                outputs = tf.keras.layers.Conv2D(
                    name=get_name_scope() + 'conv',
                    kernel_size=[3, 3],
                    dilation_rate=[1, 1] if layer_idx < 2 else [2 ** (layer_idx - 1), 2 ** (layer_idx - 1)],
                    padding='SAME',
                    use_bias=False,
                    filters=64
                )(outputs)
                outputs.set_shape([None, None, 320, None])
                outputs = bn_relu_fn(outputs, dropping_prob, training=False)

    with tf.name_scope('temporal'):
        outputs = VocalPitchLSTM(
            name=get_name_scope() + 'lstm',
            ss_context=6,
            rnn_units=64,
            num_f_bins=320,
        )(outputs)
        outputs.set_shape([None, None, 320, None])
        outputs = tf.keras.layers.Dropout(name=get_name_scope() + 'drop', rate=dropping_prob)(outputs)

    with tf.name_scope('melody'):
        outputs = tf.keras.layers.Dense(
            name=get_name_scope() + 'dense',
            use_bias=True,
            units=1
        )(outputs)
        outputs = tf.squeeze(outputs, axis=[0, -1])
        outputs.set_shape([None, 320])

    model = tf.keras.Model(inputs, outputs)

    return model


if __name__ == '__main__':

    model = create_acoustic_model_fn()
    model.summary(line_length=150)
    for idx, w in enumerate(model.trainable_variables):
        print(idx, w.name, w.shape)
    inputs = tf.random.uniform([1, 1200, 500])
    outputs = model(inputs, training=False)
    print(outputs.shape)


