import tensorflow as tf
from tensorflow.python.eager import context


def get_name_scope():

    ctx = context.context()
    if ctx.executing_eagerly():
        return ctx.scope_name
    else:
        scope_name = tf.compat.v1.get_default_graph().get_name_scope()

        if len(scope_name) > 0 and not scope_name.endswith('/'):
            scope_name = scope_name + '/'

        return scope_name
