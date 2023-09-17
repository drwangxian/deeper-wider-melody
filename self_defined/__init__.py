import logging
from self_defined.get_name_scope import get_name_scope
from self_defined.load_np_array_from_file import load_np_array_from_file_fn
from self_defined.bn_relu_drop import bn_relu_drop_fn
from self_defined.save_np_array_to_file import save_np_array_to_file_fn
from array_to_tf_table import ArrayToTableTFFn
from is_vocals import is_vocals_m2m3_fn
from is_vocals import is_vocals_singer_fn
try:
    from torch_bce_tf_style import torch_bce_tf_style_fn
    from torch_bce_tf_style import pytorch_set_shape_fn
except ModuleNotFoundError:
    logging.info('pytorch not installed, so torch_bce_shaun_fn is not imported')
import constants
