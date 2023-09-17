import logging
logging.basicConfig(
    level=logging.WARNING
)
import medleydb as mdb
import numpy as np
import os


def assert_close_fn(t0, t1):

    t0 = np.round(t0, 6)
    t1 = np.round(t1, 6)

    assert t0 == t1


def is_vocals_m2m3_fn(track_name):

    track = mdb.MultiTrack(track_name)

    if not track.has_melody:
        return None

    m2_times_freqs = track.melody2_annotation
    num_frames = len(m2_times_freqs)
    t_last = (num_frames - 1) * 256. / 44100
    assert m2_times_freqs[0][0] == 0
    assert_close_fn(t_last, m2_times_freqs[-1][0])
    num_melody_insts = len(track.melody_rankings.keys())
    assert all(len(ll) == 2 for ll in m2_times_freqs)
    m3_times_freqs = track.melody3_annotation
    assert all(len(ll) == num_melody_insts + 1 for ll in m3_times_freqs)
    assert m3_times_freqs[0][0] == 0
    assert_close_fn(t_last, m3_times_freqs[-1][0])
    assert len(m2_times_freqs) == len(m3_times_freqs)

    vocal_indices_in_m3_melody = np.zeros([num_melody_insts], np.bool_)
    melody_rankings = track.melody_rankings
    for stem_idx in melody_rankings.keys():
        inst = track.stems[stem_idx].instrument
        assert len(inst) == 1
        inst = inst[0]
        if 'singer' in inst or 'vocalists' in inst:
            melody_rank = melody_rankings[stem_idx]
            assert melody_rank >= 1
            melody_rank = melody_rank - 1
            assert not vocal_indices_in_m3_melody[melody_rank]
            vocal_indices_in_m3_melody[melody_rank] = True

    is_vocals = np.zeros([num_frames], np.bool_)

    if track.is_instrumental:
        any_true = np.any(vocal_indices_in_m3_melody)
        assert not any_true

        return is_vocals

    n_exceptions = 0

    for idx, (m2_f, m3_fs) in enumerate(zip(m2_times_freqs, m3_times_freqs)):

        m2_f = m2_f[1]
        if m2_f == 0:
            continue

        m3_fs = m3_fs[1:]

        is_closes = [np.array_equal(m2_f, f) for f in m3_fs]
        is_closes = np.asarray(is_closes, np.int32)
        any_close = np.sum(is_closes, dtype=np.int32)
        assert any_close >= 1

        if any_close == 1:
            which_close = np.argmax(is_closes)
            t = vocal_indices_in_m3_melody[which_close]
            if t:
                is_vocals[idx] = True
            continue

        for t_idx, t_vocal in enumerate(vocal_indices_in_m3_melody):
            if t_vocal:
                if is_closes[t_idx]:
                    is_vocals[idx] = True
                    n_exceptions = n_exceptions + 1
                    break

    any_true = np.any(is_vocals)
    if track.is_instrumental:
        assert not any_true
    else:
        assert any_true

    if n_exceptions > 0:
        logging.warning(f'n_exceptions / n_frames --> {n_exceptions} / {num_frames}')

    return is_vocals


def is_vocals_singer_fn(track_id):

    melody2_dir = os.environ['melody2_dir']
    melody2_suffix = '_MELODY2.csv'
    sr = 44100

    annot_path = os.path.join(melody2_dir, track_id + melody2_suffix)
    times_labels = np.genfromtxt(annot_path, delimiter=',')
    assert np.all(np.logical_not(np.isnan(times_labels)))
    assert times_labels.ndim == 2 and times_labels.shape[1] == 2
    num_frames = len(times_labels)
    tmp = np.arange(num_frames) * (256. / 44100.)
    assert np.all(tmp == times_labels[:, 0])

    is_vocals = np.zeros([num_frames], dtype=np.bool_)
    section_file = os.environ['section_dir']
    section_file = os.path.join(section_file, track_id + '_SOURCEID.lab')
    h = 256
    hh = h // 2

    with open(section_file, 'r') as fh:
        lines = fh.readlines()

        for line in lines:
            if 'start_time' in line:
                continue

            parts = line.split(',')
            instrument = parts[-1]
            if 'singer' not in instrument:
                continue
            st = float(parts[0])
            et = float(parts[1])

            ss = np.int(np.ceil(st * sr))
            es = np.int(np.floor(et * sr))

            sf = (ss + hh) // h
            ef = (es + hh) // h
            is_vocals[sf:ef + 1] = True

    is_vocals = np.logical_and(is_vocals, times_labels[:, 1] > 0.)

    return is_vocals


if __name__ == '__main__':

    track_names = mdb.TRACK_LIST_V1
    for idx, track_name in enumerate(track_names):
        print(idx)
        is_vocals = is_vocals_m2m3_fn(track_name)

