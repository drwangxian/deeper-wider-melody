import tensorflow as tf
import numpy as np
import time


class VocalPitchLSTM(tf.keras.layers.Layer):

    def __init__(self, ss_context, rnn_units, num_f_bins, **kwargs):

        super(VocalPitchLSTM, self).__init__(**kwargs)

        self.ss_context = ss_context
        self.local_window_size = 2 * ss_context + 1
        self.f_bins = num_f_bins
        self.rnn_units = rnn_units

        self.edge_modifications = tf.convert_to_tensor(self.edge_modification_fn())
        self.range_over_fs = tf.range(self.f_bins, dtype=tf.int32) - ss_context
        self.zeros_initial = tf.zeros([self.f_bins, self.rnn_units])

        self.lstm_cell = tf.keras.layers.LSTMCell(self.rnn_units)

    def call(self, inputs):

        shape = tf.shape(inputs)
        b = shape[0]
        T = shape[1]
        tf.debugging.assert_equal(b, 1)
        inputs = tf.squeeze(inputs, axis=0)

        lstm_cell = self.lstm_cell
        fs = self.f_bins
        lstm_units = self.rnn_units

        inputs.set_shape([None, fs, None])

        hidden_state_seq = tf.TensorArray(tf.float32, size=T, dynamic_size=False, clear_after_read=True, element_shape=[fs, lstm_units])
        zeros_initial = self.zeros_initial

        previous_pitch_bins = self.peak_bins_fn(inputs)
        h = zeros_initial
        c = zeros_initial
        for t in tf.range(T, dtype=tf.int32):
            bins = previous_pitch_bins[t]
            h = tf.gather(h, indices=bins, axis=0)
            h.set_shape([fs, lstm_units])
            c = tf.gather(c, indices=bins, axis=0)
            c.set_shape([fs, lstm_units])
            x = inputs[t]
            _, (h, c) = lstm_cell(x, states=[h, c])
            hidden_state_seq = hidden_state_seq.write(t, h)

        hidden_state_seq = hidden_state_seq.stack()
        hidden_state_seq = hidden_state_seq[None, ...]

        return hidden_state_seq

    def edge_modification_fn(self):

        ss_context = self.ss_context
        local_window_size = self.local_window_size
        fs = self.f_bins
        modifications = np.zeros([fs, local_window_size], dtype=np.float32)
        for f in np.arange(ss_context):
            n = ss_context - f
            modifications[f, :n] -= 1
            modifications[fs - 1 - f, -n:] -= 1

        return modifications

    def peak_bins_fn(self, inputs):

        fs = self.f_bins
        cs = None
        ss_context = self.ss_context
        local_window_size = self.local_window_size
        edge_modifications = self.edge_modifications
        edge_modifications.set_shape([fs, local_window_size])

        inputs.set_shape([None, fs, None])
        current = inputs
        previous = current[:-1]
        previous = tf.pad(previous, [[1, 0], [0, 0], [0, 0]])
        previous = tf.pad(previous, [[0, 0], [ss_context, ss_context], [0, 0]], mode='reflect')
        previous = tf.signal.frame(previous, frame_length=local_window_size, frame_step=1, pad_end=False, axis=1)
        previous.set_shape([None, fs, local_window_size, cs])

        correlation = tf.einsum('...wc,...c->...w', previous, current)
        correlation.set_shape([None, fs, local_window_size])
        correlation = correlation + edge_modifications
        previous_pitch_bins = tf.argmax(correlation, axis=-1, output_type=tf.int32)
        previous_pitch_bins.set_shape([None, fs])
        previous_pitch_bins = previous_pitch_bins + self.range_over_fs

        return previous_pitch_bins


class ShaunLSTM(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super(ShaunLSTM, self).__init__()

        self.lstm_cell = tf.keras.layers.LSTMCell(64)

    def call(self, inputs):

        inputs.set_shape([1, None, 320, None])

        outputs = inputs

        outputs = tf.squeeze(outputs, axis=0)
        T = tf.shape(outputs)[0]
        h = tf.zeros([320, 64])
        c = h
        seq = tf.TensorArray(tf.float32, size=T, dynamic_size=False, clear_after_read=True, element_shape=[320, 64])
        for t in tf.range(T):
            x = outputs[t]
            _, (h, c) = self.lstm_cell(x, states=[h, c])
            seq = seq.write(t, h)

        seq = seq.stack()
        seq.set_shape([None, 320, 64])
        seq = seq[None, ...]

        return seq


if __name__ == '__main__':

    inputs = tf.random.normal([1, 1200, 320, 64])

    lstm_layer = VocalPitchLSTM(
        ss_context=6,
        rnn_units=64,
        num_f_bins=320
    )

    n = 10 ** 4
    for i in range(n):
        print(i)
        sth = lstm_layer(inputs, training=True)





































