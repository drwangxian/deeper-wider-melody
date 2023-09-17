"""
12-Jan-2022, no USE_DOUBLE
"""


import tensorflow as tf
import numpy as np
import librosa
import os
import soundfile


USE_DOUBLE = False


def rc01_fn(N):

    assert isinstance(N, (int, np.int32, np.int64))
    assert N >= 2

    rc = .5 - .5 * np.cos(np.pi * np.arange(N) / float(N))
    assert rc[0] == 0. and rc[-1] < 1.

    return rc


def load_samples_from_wav_file_fn(wav_file):

    samples, _sr = soundfile.read(wav_file, dtype='int16', samplerate=None)
    assert _sr == 44100
    assert samples.dtype == np.int16
    assert samples.ndim in (1, 2)
    samples = samples.astype(np.float64)
    if samples.ndim == 2:
        assert samples.shape[1] == 2
        samples = np.mean(samples, axis=1)
    samples = samples / float(32768)
    samples = samples.astype(np.float32)

    return samples


class NSGT:

    def __init__(self, Ls):

        assert isinstance(Ls, int)
        self.use_double = USE_DOUBLE

        self.Ls = Ls
        self.hLs = self.Ls // 2

        if self.use_double:
            np_float = np.float64
            tf_float = tf.float64
            np_complex = np.complex128
            tf_complex = tf.complex128
        else:
            np_float = np.float32
            tf_float = tf.float32
            np_complex = np.complex64
            tf_complex = tf.complex64
        self.np_float = np_float
        self.np_complex = np_complex
        self.tf_float = tf_float
        self.tf_complex = tf_complex

        self.B = 60
        self.factor = 2. ** (1. / self.B)
        self.sr = 44100
        self.nf = self.sr // 2
        self.gamma = 14
        self.fmin = librosa.midi_to_hz(24) / self.factor ** 2
        self.hop_size = 64
        self.num_frames_per_Ls = self.Ls // self.hop_size
        self.uni_side_cyc_frames = int(2.88 / self.gamma * self.sr / self.hop_size)

        self._gen_gs_gds_ranges_fn()

    def _gen_gs_gds_ranges_fn(self):

        Ls = self.Ls
        hLs = self.hLs
        hop_size = self.hop_size
        factor = self.factor
        fmin = self.fmin
        sr = self.sr
        nf = self.nf
        gamma = self.gamma

        fbas = []
        f = fmin
        while f < nf:
            fbas.append(f)
            f = f * factor

        fbas = np.asarray(fbas, dtype=np.float64)
        Lfbas = len(fbas)
        self.Lfbas = Lfbas
        nyq_pos = Lfbas + 1
        self.nyq_pos = nyq_pos

        fft_res = float(sr) / float(Ls)
        posit = fbas / fft_res
        posit = np.round(posit)
        posit = posit.astype(np.int32)
        posit = np.concatenate([[0], posit, [hLs]])
        posit = posit.astype(np.int32)
        posit = np.pad(posit, [[0, Lfbas]], mode='reflect')
        posit[nyq_pos + 1:] = Ls - posit[nyq_pos + 1:]
        assert posit[0] == 0 and posit[nyq_pos] == hLs
        assert posit[1] + posit[-1] == Ls
        assert posit[nyq_pos - 1] + posit[nyq_pos + 1] == Ls

        bw = np.empty(Lfbas + 2, dtype=np.int32)
        min_bw = int(gamma / 2. / fft_res)
        min_bw = 2 * min_bw + 1

        ranges = []
        for idx in range(Lfbas + 2):

            if idx == 0:
                bw[idx] = 2 * posit[1] + 1
                _range = np.arange(-posit[1], posit[1] + 1, dtype=np.int32)
                ranges.append(_range)

            elif idx == 1:
                bw[idx] = min_bw
                t = min_bw // 2
                _range = np.arange(-t, t + 1, dtype=np.int32)
                ranges.append(_range)

            else:
                _bw = posit[idx + 1] - posit[idx - 1] + 1
                if _bw <= min_bw:
                    bw[idx] = bw[1]
                    _range = ranges[1]
                    ranges.append(_range)
                else:
                    bw[idx] = _bw
                    llen = posit[idx - 1] - posit[idx]
                    rlen = posit[idx + 1] - posit[idx]
                    _range = np.arange(llen, rlen + 1, dtype=np.int32)
                    assert len(_range) == bw[idx]
                    ranges.append(_range)

        for idx in (0, 1, nyq_pos):
            assert bw[idx] % 2 == 1

        assert bw[0] > bw[1]
        assert bw[nyq_pos - 2] > bw[nyq_pos - 1] > bw[nyq_pos]

        assert len(bw) == Lfbas + 2
        bw = np.pad(bw, [[0, Lfbas]], mode='reflect')
        assert bw[1] == bw[-1]
        assert bw[nyq_pos - 1] == bw[nyq_pos + 1]

        gs = []

        idx = 1
        _range = ranges[1]
        llen = -_range[0]
        left = rc01_fn(llen)
        rlen = _range[-1]
        assert rlen == llen
        right = left[::-1]
        g1 = np.concatenate([left, [1], right])
        assert len(g1) == len(_range) and np.array_equal(g1, g1[::-1])

        idx = 0
        g0 = np.ones(bw[0])
        g0[:llen] = left
        g0[-llen:] = right
        assert np.array_equal(g0, g0[::-1])
        assert len(g0) == len(ranges[0])

        gs.extend([g0, g1])

        for idx in range(2, nyq_pos + 1):
            if bw[idx] == bw[1]:
                _g = gs[1]
                gs.append(_g)
                continue
            _range = ranges[idx]
            llen = -_range[0]
            rlen = _range[-1]
            left = rc01_fn(llen)
            right = rc01_fn(rlen)[::-1]
            _g = np.concatenate([left, [1], right])
            assert len(_g) == len(_range)
            gs.append(_g)

        assert len(gs) == len(ranges) == Lfbas + 2

        _gs = []
        _ranges = []
        for g, _range in zip(gs[-2:-len(gs):-1], ranges[-2:-len(ranges):-1]):
            g = g[::-1]
            _gs.append(g)

            _range = -_range[::-1]
            _ranges.append(_range)
        gs.extend(_gs)
        ranges.extend(_ranges)
        assert len(gs) == len(ranges) == 2 * Lfbas + 2

        win_range_list = []
        for ii in range(2 * Lfbas + 2):
            _range = ranges[ii]
            win_range = (posit[ii] + _range) % Ls
            assert win_range.dtype == np.int32
            win_range_list.append(win_range)

        assert Ls / float(hop_size) > bw.max()
        max_bw = bw.max()
        max_bw = np.log2(max_bw)
        max_bw = np.ceil(max_bw)
        max_bw = 2 ** int(max_bw)
        assert Ls // max_bw == hop_size

        norm_factor = 2. * max_bw / Ls
        for ii, g in enumerate(gs):
            gs[ii] = g * norm_factor

        diagonal = np.zeros(Ls)
        assert len(win_range_list) == 2 * Lfbas + 2

        for ii in range(2 * Lfbas + 2):
            g = gs[ii]
            win_range = win_range_list[ii]
            diagonal[win_range] += g ** 2

        assert np.all(diagonal > 0.)
        print('diagonal - minimum - {}'.format(diagonal.min()))
        # the theoretical minimum is : 2 / hop_size ** 2
        print('theoretical minimum - {}'.format(2. / hop_size ** 2))

        def _symmetry_chk_fn(diagonal):

            t_pos = diagonal[1:hLs]
            t_neg = diagonal[hLs + 1:][::-1]
            t = t_pos - t_neg
            t = np.sum(np.abs(t))
            print('diagonal difference -', t)

        _symmetry_chk_fn(diagonal)

        diagonal = np.pad(diagonal[:hLs + 1], [[0, hLs - 1]], mode='reflect')
        assert len(diagonal) == Ls
        _symmetry_chk_fn(diagonal)

        gds = []
        for ii in range(2 * Lfbas + 2):
            gd = gs[ii]
            win_range = win_range_list[ii]
            gd = gd / diagonal[win_range]
            gds.append(gd)

        self.gs = gs
        self.gds = gds
        self.posit = posit
        self.bw = bw
        self.max_bw = max_bw
        self.ranges = ranges
        self.win_range_list = win_range_list

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float64 if USE_DOUBLE else tf.float32)], autograph=False)
    def forward_tf_fn(self, samples):

        Ls = self.Ls
        Lfbas = self.Lfbas
        gs = self.gs
        win_range_list = self.win_range_list
        ranges = self.ranges
        max_bw = self.max_bw
        posit = self.posit
        hLs = self.hLs

        samples = tf.convert_to_tensor(samples)
        samples.set_shape([Ls])
        assert samples.dtype == self.tf_float

        samples_fft = tf.signal.rfft(samples)
        assert samples_fft.dtype == self.tf_complex
        t = hLs + 1
        samples_fft.set_shape([t])
        positive_fft = samples_fft[-2:-t:-1]
        positive_fft = tf.math.conj(positive_fft)
        samples_fft = tf.concat([samples_fft, positive_fft], axis=0)
        samples_fft.set_shape([Ls])

        f_list = []
        for ii in range(Lfbas + 2):
            g = gs[ii]
            lg = len(g)
            g = tf.convert_to_tensor(g, dtype=self.tf_float)

            win_range = win_range_list[ii]
            f = tf.gather(samples_fft, win_range, axis=0)
            f_real = tf.math.real(f)
            f_imag = tf.math.imag(f)
            f_real = f_real * g
            f_imag = f_imag * g
            f = tf.complex(real=f_real, imag=f_imag)

            t = max_bw - lg
            assert t >= 0
            if t:
                f = tf.pad(f, [[0, t]])

            llen = -ranges[ii][0]
            assert llen > 0
            displace = posit[ii] % max_bw - llen
            if displace:
                f = tf.roll(f, displace, axis=0)
            f_list.append(f)

        f_list = tf.stack(f_list, axis=0)
        f_list.set_shape([Lfbas + 2, max_bw])
        f_list = tf.signal.ifft(f_list)
        assert f_list.dtype == self.tf_complex
        f_list.set_shape([Lfbas + 2, max_bw])

        return f_list

    @tf.function(input_signature=[
        tf.TensorSpec([566 + 2, None], dtype=tf.complex128 if USE_DOUBLE else tf.complex64)],
        autograph=False
    )
    def inverse_tf_fn(self, nsgt_coeffs):

        Lfbas = self.Lfbas
        max_bw = self.max_bw
        posit = self.posit
        gds = self.gds
        ranges = self.ranges
        win_range_list = self.win_range_list
        hLs = self.hLs
        Ls = self.Ls

        nsgt_coeffs = tf.convert_to_tensor(nsgt_coeffs)
        nsgt_coeffs.set_shape([Lfbas + 2, max_bw])
        t = tf.shape(nsgt_coeffs)
        tf.debugging.assert_equal(t, [Lfbas + 2, max_bw])
        assert nsgt_coeffs.dtype == self.tf_complex
        nsgt_coeffs = tf.signal.fft(nsgt_coeffs)
        nsgt_coeffs = tf.unstack(nsgt_coeffs, axis=0)
        assert len(nsgt_coeffs) == Lfbas + 2
        samples_fft = tf.zeros([Ls], dtype=self.tf_complex)
        for ii, fft in enumerate(nsgt_coeffs):
            displace = posit[ii] % max_bw
            assert displace >= 0
            if displace:
                fft = tf.roll(fft, -displace, axis=0)
            # pos, zeros, neg

            gd = gds[ii]
            lg = len(gd)
            gd = tf.convert_to_tensor(gd, dtype=self.tf_float)
            _range = ranges[ii]
            true_range = _range % max_bw
            fft = tf.gather(fft, true_range, axis=0)
            fft.set_shape([lg])
            # neg, pos

            fft_real, fft_imag = tf.math.real(fft), tf.math.imag(fft)
            fft_real = fft_real * gd
            fft_imag = fft_imag * gd
            fft = tf.complex(real=fft_real, imag=fft_imag)
            assert fft.dtype == self.tf_complex

            win_range = win_range_list[ii]
            win_range = tf.convert_to_tensor(win_range[:, None])
            samples_fft = tf.tensor_scatter_nd_add(
                samples_fft, win_range, fft
            )

        t = hLs - 1
        spec_0, spec_pos, spec_nyq, _ = tf.split(samples_fft, [1, t, 1, t], axis=0)

        spec_0 = tf.math.real(spec_0)
        spec_0 = tf.complex(real=spec_0, imag=tf.zeros_like(spec_0))

        spec_nyq = tf.math.real(spec_nyq)
        spec_nyq = tf.complex(real=spec_nyq, imag=tf.zeros_like(spec_nyq))

        spec = tf.concat([spec_0, spec_pos, spec_nyq], axis=0)
        spec.set_shape([hLs + 1])

        spec = tf.signal.irfft(spec)
        assert spec.dtype == self.tf_float
        spec.set_shape([Ls])

        return spec

    def validty_chk_fn(self):

        Ls = self.Ls
        sr = self.sr
        t = (Ls + sr - 1) // sr
        wav_file = os.environ['wav_file_short']
        samples, _sr = librosa.load(wav_file, sr=None, mono=True, duration=t, dtype=self.np_float)
        assert _sr == sr
        samples = samples[:Ls]
        _samples = self.inverse_tf_fn(self.forward_tf_fn(samples))
        _samples = _samples.numpy()
        assert _samples.dtype == samples.dtype

        if samples.dtype == np.float32:
            samples = samples.astype(np.float64)
            _samples = _samples.astype(np.float64)

        t = np.sum((samples - _samples) ** 2)
        t1 = np.sum(samples ** 2)

        snr = 10. * (np.log10(t1) - np.log10(t))

        print('snr - {}'.format(snr))

    def nsgt_of_wav_file_fn(self, wav_file_or_samples):

        hop_size = self.hop_size
        num_frames_per_Ls = self.num_frames_per_Ls
        cyc_frames = self.uni_side_cyc_frames
        num_payload_frames_per_Ls = num_frames_per_Ls - 2 * cyc_frames
        Ls = self.Ls
        Lfbas = self.Lfbas

        if not isinstance(wav_file_or_samples, np.ndarray):
            samples = load_samples_from_wav_file_fn(wav_file_or_samples)
        else:
            samples = wav_file_or_samples
        num_samples = len(samples)

        r = num_samples % hop_size
        if r > 0:
            assert r < hop_size
            t = hop_size - r
            samples = np.pad(samples, [[0, t]])
            num_samples = num_samples + t

        num_samples_before_padding = num_samples
        num_frames_before_padding = num_samples_before_padding // hop_size
        num_Ls_snippets = (num_frames_before_padding + num_payload_frames_per_Ls - 1) // num_payload_frames_per_Ls
        assert num_Ls_snippets >= 2
        r = (num_frames_before_padding - num_payload_frames_per_Ls) % (num_Ls_snippets - 1)
        if r > 0:
            assert r < num_Ls_snippets - 1
            paddings = (num_Ls_snippets - 1 - r) * hop_size
            samples = np.pad(samples, [[0, paddings]])
        num_samples_after_padding = len(samples)
        num_frames_after_padding = num_samples_after_padding // hop_size
        r = (num_frames_after_padding - num_payload_frames_per_Ls) % (num_Ls_snippets - 1)
        assert r == 0
        hop_frames = (num_frames_after_padding - num_payload_frames_per_Ls) // (num_Ls_snippets - 1)
        assert hop_frames <= num_payload_frames_per_Ls

        assert (num_Ls_snippets - 1) * hop_frames + num_payload_frames_per_Ls == num_frames_after_padding

        snippet_nsgt_list = []

        for Ls_idx in range(num_Ls_snippets):

            start_frame = Ls_idx * hop_frames
            end_frame = start_frame + num_payload_frames_per_Ls
            assert end_frame <= num_frames_after_padding

            start_frame = start_frame - cyc_frames
            end_frame = end_frame + cyc_frames

            if start_frame < 0:
                pre_paddings = -start_frame * hop_size
                start_frame = 0
            else:
                pre_paddings = 0

            if end_frame > num_frames_after_padding:
                post_padding = (end_frame - num_frames_after_padding) * hop_size
                end_frame = num_frames_after_padding
            else:
                post_padding = 0

            Ls_samples = samples[start_frame * hop_size:end_frame * hop_size]
            Ls_samples = np.pad(Ls_samples, [[pre_paddings, post_padding]])
            assert len(Ls_samples) == Ls
            snippet_nsgt = self.forward_tf_fn(Ls_samples)
            tf.ensure_shape(snippet_nsgt, [Lfbas + 2, num_frames_per_Ls])
            snippet_nsgt = tf.abs(snippet_nsgt)
            snippet_nsgt = snippet_nsgt.numpy()
            if Ls_idx < num_Ls_snippets - 1:
                snippet_nsgt = snippet_nsgt[:, cyc_frames:cyc_frames + hop_frames]
            else:
                snippet_nsgt = snippet_nsgt[:, cyc_frames:cyc_frames + num_payload_frames_per_Ls]
            snippet_nsgt_list.append(snippet_nsgt)
        nsgt = np.concatenate(snippet_nsgt_list, axis=1)
        _num_frames = nsgt.shape[1]
        assert _num_frames == (num_Ls_snippets - 1) * hop_frames + num_payload_frames_per_Ls
        assert _num_frames == num_frames_after_padding
        nsgt = nsgt[:, :num_frames_before_padding]
        nsgt = nsgt.T
        nsgt = np.require(nsgt, np.float32, requirements=['O', 'C'])
        nsgt.flags['WRITEABLE'] = False
        assert nsgt.shape == (num_frames_before_padding, Lfbas + 2)

        return nsgt


if __name__ == '__main__':

    nsgt_ins = NSGT(2 ** 18)
    nsgt_ins.validty_chk_fn()





























