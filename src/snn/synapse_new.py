### Zihang
### ATGroup, NUS

import numpy as np

synapse_default_init_params = {
    'min_weight': 0,
    'max_weight': 1,
    'causal_amplitude': 1e-2,
    'causal_time_constant': 10, # /time_step=0.1
    'causal_time_range': 20, # /time_step=0.1
    'acausal_amplitude': -1e-2,
    'acausal_time_constant': 10, # /time_step=0.1
    'acausal_time_range': 20, # /time_step=0.1
    'linear_decay_rate': 1e-4,
}

def synapse_init(shape, init_params=None):
    """Initialize synapse states.
    shape: (output_size, input_size)
    """
    if init_params is None:
        init_params = synapse_default_init_params

    init_vector = np.array([
        0, # initial weight
        np.inf,
        np.inf,
        init_params['min_weight'],
        init_params['max_weight'],
        init_params['causal_amplitude'],
        init_params['causal_time_constant'],
        init_params['causal_time_range'],
        init_params['acausal_amplitude'],
        init_params['acausal_time_constant'],
        init_params['acausal_time_range'],
        init_params['linear_decay_rate'],
    ])
    return np.tile(init_vector, (*shape, 1)).transpose()

def synapse_dynamic(s, pre, post):
    """Synapse dynamics.
    s: numpy array of synapse states
        (12, input_size, output_size)
        [0] synapse weight
        [1] last pre-spike time
        [2] last post-spike time
        [3] min weight
        [4] max weight
        [5] causal amplitude
        [6] causal time constant
        [7] causal time range
        [8] acausal amplitude
        [9] acausal time constant
        [10] acausal time range
        [11] linear decay rate
    pre: numpy list of pre-synaptic spikes
        (input_size, )
    post: numpy list of post-synaptic spikes
        (output_size, )
    """
    output = np.dot(s[0], pre) # multiply

    for i in range(post.shape[0]):
        if post[i]:
            # process post-synaptic spikes, causal
            causal_update = s[5,i] * np.exp(-s[1,i]/s[6,i])
            s[0,i] = s[0,i] + causal_update * (s[1,i] <= s[7,i])
            s[0,i] = s[0,i] - s[11,i] * (s[1,i] > s[7,i]) # linear decay
            s[2,i] = 0 # update last post-spike time
        else:
            s[2,i] = s[2,i] + 1 # update last post-spike time

    for i in range(pre.shape[0]):
        if pre[i]:
            # process pre-synaptic spikes, acasual
            acausal_update = s[8,:,i] * np.exp(-s[2,:,i]/s[9,:,i])
            s[0,:,i] = s[0,:,i] + acausal_update * (s[2,:,i] <= s[10,:,i])
            s[1,:,i] = 0 # update last pre-spike time
        else:
            s[1,:,i] = s[1,:,i] + 1 # update last pre-spike time

    if np.any(pre) or np.any(post):
        s[0] = np.clip(s[0], s[3], s[4])

    return output # return MAC result

def synapse_reset(s):
    """Reset synapse states.
    """
    s[1] = np.inf
    s[2] = np.inf
