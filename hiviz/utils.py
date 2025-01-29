# tensor utilities

import torch

def quantize(tensor, scale=None):
    if scale is not None:
        lo, hi = scale
        tensor = (tensor - lo) / (hi - lo)
    return (tensor * 255).round().clamp(0, 255).to(torch.uint8)

def meshgrid(size, xlim=(0, 1), ylim=(0, 1), device='cuda'):
    width, height = size
    xlo, xhi = xlim
    ylo, yhi = ylim
    x = torch.linspace(xlo, xhi, width, device=device)
    y = torch.linspace(ylo, yhi, height, device=device)
    return torch.meshgrid(x, y, indexing='ij')
