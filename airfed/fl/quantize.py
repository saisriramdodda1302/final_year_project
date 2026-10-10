"""Gradient-sensitive adaptive quantization (paper Eqs. 82-84).

Parameters whose gradients are large matter more, so they keep more bits.
Ranks map linearly onto bit widths between b_min and b_max.
"""

import torch

ID_BITS = 4      # a bit width between 4 and 16 fits in a 4-bit identifier
FLOAT_BITS = 32  # min and max of each tensor travel as float32


def bit_widths(sensitivity, b_min, b_max):
    """Eq. 82. Rank 1 (largest |gradient|) gets b_max, the last rank gets b_min."""
    n = sensitivity.numel()
    order = torch.argsort(sensitivity, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(n)                                  # 0-based, so this is r_i - 1
    # integer maths avoids float rounding: floor((1 - rank/(n-1)) * (b_max - b_min))
    return b_min + ((n - 1 - rank) * (b_max - b_min)) // max(n - 1, 1)


def quantize(theta, bits):
    """Eq. 83: map each value onto 2^b - 1 levels between the tensor's min and max."""
    lo, hi = theta.min(), theta.max()
    levels = (2.0 ** bits.float()) - 1.0
    q = torch.round((theta - lo) / (hi - lo).clamp_min(1e-12) * levels)
    return q, lo, hi


def dequantize(q, bits, lo, hi):
    """Eq. 84: what the receiver reconstructs."""
    levels = (2.0 ** bits.float()) - 1.0
    return lo + q / levels * (hi - lo)


def compress(tensors, grads, b_min, b_max):
    """Quantize a list of tensors as one message.

    Returns the tensors as the receiver would see them and the message size in bits
    (values + per-value bit-width ids + each tensor's min/max).
    """
    flat_grad = torch.cat([g.flatten() for g in grads])
    bits = bit_widths(flat_grad.abs(), b_min, b_max)
    out, start, size = [], 0, 0
    for t in tensors:
        n = t.numel()
        b = bits[start:start + n].view_as(t)
        q, lo, hi = quantize(t, b)
        out.append(dequantize(q, b, lo, hi))
        size += int(b.sum()) + ID_BITS * n + 2 * FLOAT_BITS
        start += n
    return out, size


def full_precision(tensors):
    """No quantization: every value travels as float32."""
    return [t.clone() for t in tensors], FLOAT_BITS * sum(t.numel() for t in tensors)
