# testing functions

import torch

from .utils import quantize
from .hiviz import HiViz

# generate white noise
def generate_white_noise(size=(512, 512)):
    while True:
        yield torch.randint(0, 256, size, device='cuda', dtype=torch.uint8)

# generate wave fronts
def generate_wave_fronts(size=(512, 512)):
    width, height = size
    x, y = meshgrid(size, device='cuda')
    t, delta = 0, 0.01
    while True:
        v = 1 + torch.sin((x * y + t) * 2 * torch.pi)
        yield quantize(v, scale=(0, 2))
        t = (t + delta) % 1

def minimize(f, x0, step=0.01):
    x = x0.to('cuda', copy=True)
    x.requires_grad = True
    opt = torch.optim.SGD([x], lr=step)
    while True:
        loss = f(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        yield quantize(x.detach(), scale=(-2, 2))

# run test function
def run_test(gen, size=(512, 512)):
    viz = HiViz(size)
    viz.animate(gen(size))
    viz.cleanup()
