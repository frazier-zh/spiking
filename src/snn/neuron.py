### Zihang
### ATGroup, NUS

import numpy as np

adaptive_neuron_default_init_params = {
    'resting_potential': 0,
    'resting_threshold': 0.2,
    'sensitivity': 0.5,
    'time_constant': 200, # /time_step=0.1
    'firing_potential': 2,
    'refractory_period': 20, # /time_step=0.1
    'threshold_time_constant': 1000, # /time_step=0.1
    'threshold_increment': 0.02
}

def adaptive_neuron_init(size, init_params=None):
    """Initialize adaptive neuron states.
    """
    if init_params is None:
        init_params = adaptive_neuron_default_init_params

    init_vector = np.array([
        init_params['resting_potential'],
        init_params['resting_threshold'],
        init_params['sensitivity'],
        init_params['resting_potential'],
        init_params['time_constant'],
        init_params['resting_threshold'],
        init_params['threshold_time_constant'],
        init_params['threshold_increment'],
        0, # refractory countdown
        init_params['refractory_period'],
    ])
    return np.tile(init_vector, (size, 1)).transpose()

def adaptiva_neuron_dynamics(n, i):
    """Adaptive neuron dynamics.
    n: numpy array of neuron states
        (10, size)
        [0] membrane potential
        [1] adaptive threshold

        [2] nueron sensitivity
        [3] neuron resting potential
        [4] neuron time constant
        [5] neuron resting threshold
        [6] neuron threshold time constant
        [7] neuron threshold increment

        [8] refractory countdown
        [9] refractory period
    i: numpy list of input spikes
        (size, )
    """
    active_mask = n[8] <= 0 # get active neuron mask
    n[0] = n[0] + i * active_mask * n[2] + (n[3] - n[0]) / n[4] # update membrane potential

    fired = n[0] >= n[1] # check if neuron fires
    n[0] = n[3] * fired + n[0] * ~fired # reset membrane potential if neuron fires
    n[8] = n[9] * fired + (n[8] - 1) * ~fired # update refractory countdown
    
    n[1] = n[1] + fired * n[7] + (n[5] - n[1]) / n[6] # update adaptive threshold

    return fired # return spike

def adaptiva_neuron_reset(n):
    """Reset adaptive neuron states.
    """
    n[8] = 0
