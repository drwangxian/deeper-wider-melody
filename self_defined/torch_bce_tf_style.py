"""
tf style bce loss for torch implemented by shaun
"""

import torch


def torch_bce_tf_style_fn(*, logits, labels):

    zeros = torch.zeros_like(logits)
    cond = logits >= zeros
    max_term = torch.where(cond, logits, zeros)
    xz_term = logits * labels
    minus_abs_x = torch.where(cond, -logits, logits)
    log_term = torch.log1p(torch.exp(minus_abs_x))
    loss = max_term - xz_term + log_term
    loss = loss.cpu().numpy()

    return loss


def pytorch_set_shape_fn(tensor, shape):

    ndim = len(shape)
    assert tensor.ndim == ndim

    for s1, s2 in zip(tensor.shape, shape):
        if s2 is None:
            continue
        assert s1 == s2