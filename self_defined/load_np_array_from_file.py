import numpy as np

def load_np_array_from_file_fn(file_name):

    with open(file_name, 'rb') as fh:
        first_line = fh.readline().decode('utf-8')
        first_line = first_line.split()
        rec_name = first_line[0]

        if first_line[1] in ('C', 'F'):
            contiguous = first_line[1]
            assert contiguous in ('C', 'F')
            c_flag = contiguous == 'C'
            dtype = first_line[2]
            dim = first_line[3:]
            dim = [int(_item) for _item in dim]
            output = np.frombuffer(fh.read(), dtype=dtype).reshape(*dim)

            if len(dim) > 1 and not c_flag:
                output = np.require(output, requirements=['F'])
        else:
            dtype = first_line[1]
            dim = first_line[2:]
            dim = [int(_item) for _item in dim]
            output = np.frombuffer(fh.read(), dtype=dtype).reshape(*dim)

        return rec_name, output
