### 2019 Sep 10
### Zihang
### ATGroup, NUS

# Rodieck(1965) https://www.sciencedirect.com/science/article/pii/0042698965900337?via%3Dihub
# Difference-of-Gaussians model of ganglion cell receptive fields

import numpy as np
from . import parameters as param
from scipy import signal

def rf(raw_data):
    image = raw_data.reshape(param.pixel_x, param.pixel_y)
    
    # receptive field kernel
    k = [[  -0.5, -0.125,  0.125, -0.125,   -0.5],
         [-0.125,  0.125,  0.625,  0.125, -0.125],
         [ 0.125,  0.625,      1,  0.625,  0.125],
         [-0.125,  0.125,  0.625,  0.125, -0.125],
         [  -0.5, -0.125,  0.125, -0.125,   -0.5]]

    # convolve
    potential = signal.convolve2d(image, k, mode='same', boundary='fill', fillvalue=0)
    
    return potential.flatten()
