### Zihang
### ATGroup, NUS

# parameters taken and calculated from
# Rullen(2001) https://www.mitpressjournals.org/doi/abs/10.1162/08997660152002852

# Michaelis-Menten function
# I(C)=t_ref+1/(contrast_gain*c)
t_ref = 1 # msec
contrast_gain = 5

import numpy as np
import math
from . import parameters as param

# apply rate coding defined as Michaelis-Menten function
def encode(pot):
    train = np.zeros((pot.size, param.total_ticks), dtype=bool)

    for p in range(pot.size):
        #calculating firing rate proportional to the membrane potential
        #contrast = np.interp(pot[p], [-1.069,2.781], [0, 1])
        contrast = max(0.001, pot[p])

        interval = (t_ref+1/(contrast*contrast_gain))/param.dt

        #generating spikes according to the firing rate
        k = np.random.randint(0, interval)
                
        while k<param.total_ticks:
            train[p][int(k)] = True
            k = k + interval

    return train
