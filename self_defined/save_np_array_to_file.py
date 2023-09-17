import numpy as np
import os

def save_np_array_to_file_fn(file_name, output, rec_name):

    assert isinstance(rec_name, str)
    assert len(rec_name)
    assert ' ' not in rec_name
    assert isinstance(output, np.ndarray)
    assert output.ndim >= 1

    with open(file_name, 'wb') as fh:
        fh.write(rec_name.encode('utf-8'))
        fh.write(b' ')

        c_flag = output.flags['C_CONTIGUOUS']
        f_flag = output.flags['F_CONTIGUOUS']
        if output.ndim == 1:
            contiguous = 'C'
        else:
            assert  c_flag or f_flag
            # assert not(c_flag and f_flag)
            contiguous = 'C' if c_flag else 'F'
        fh.write(contiguous.encode('utf-8'))
        fh.write(b' ')

        dtype = '{}'.format(output.dtype).encode('utf-8')
        fh.write(dtype)

        if output.ndim > 1 and f_flag:
            output = np.require(output, requirements=['C'])

        for dim_size in output.shape:
            fh.write(b' ')
            fh.write('{:d}'.format(dim_size).encode('utf-8'))
        fh.write(b'\n')
        fh.write(output.data)
        fh.flush()
        os.fsync(fh.fileno())
