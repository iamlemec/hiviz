# tensor utilities

import torch

def quantize(tensor, scale=None):
    if scale is not None:
        lo, hi = scale
        tensor = (tensor - lo) / (hi - lo)
    return (tensor * 255).round().clamp(0, 255).to(torch.uint8)
