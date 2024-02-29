### Zihang
### ATGroup, NUS

import numpy as np

synapse_default_init_params = {
}

def synapse_init(shape, init_params=None):
    """Initialize synapse states.
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
    return

def synapse_dynamic(s, pre, post):
    """Synapse dynamics.
    s: numpy array of synapse states
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
        (size, 1)
    post: numpy list of post-synaptic spikes
        (size, 1)
    """
    output = np.dot(s, pre) # multiply

    if np.any(pre):
        # process pre-synaptic spikes, acasual
        acausal_update = s[8] * np.exp(-s[2]/s[9])
        s[0] = s[0] + pre * acausal_update * (s[2] <= s[10])
        
    if np.any(post):
        # process post-synaptic spikes, causal
        causal_update = s[5] * np.exp(-s[1]/s[6])
        s[0] = s[0] + post * causal_update * (s[1] <= s[7])
        s[0] = s[0] - post * s[11] * (s[1] > s[7]) # linear decay

    if np.any(pre) or np.any(post):
        s[0] = np.clip(s[0], s[3], s[4])

    # update last spike time
    s[1] = (s[1] + 1) * (1 - pre)
    s[2] = (s[2] + 1) * (1 - post)

    return output # return MAC result

def synapse_reset(s):
    """Reset synapse states.
    """
    s[1] = np.inf
    s[2] = np.inf

# pre-generated lookup table
# STDP weight update rule, negative dtick suggests long-term potentiation
# soft boundary applied
_f_causal = [+param.A_causal*np.exp(-x/param.tau_causal_tick) for x in range(1, param.t_causal_tick+1)]
_f_causal.reverse()
_f_acause = [-param.A_acause*np.exp(-x/param.tau_acause_tick) for x in range(1, param.t_acause_tick+1)]
f_causal = np.concatenate((_f_causal, [0]))
f_acause = np.concatenate(([0], _f_acause))
f_stdp = np.concatenate((_f_causal, [0], _f_acause))
f_astdp = np.flip(f_stdp)

from numba import njit, prange, extending
@extending.overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    def np_clip_impl(a, a_min, a_max, out=None):
        if a_min is None and a_max is None:
            raise ValueError("array_clip: must set either max or min")
        if out is None:
            out = np.empty_like(a)
        for i in range(len(a)):
            if a_min is not None and a[i] < a_min:
                out[i] = a_min
            elif a_max is not None and a[i] > a_max:
                out[i] = a_max
            else:
                out[i] = a[i]
        return out
    return np_clip_impl

t_acause_tick = param.t_acause_tick
t_causal_tick = param.t_causal_tick
total_ticks = param.total_ticks
a_linear_decay = param.eta_lin
w_min = param.w_min
w_max = param.w_max
@njit(parallel=True)
def update(synapses, spike_train, tick, decay=True):
    tick_lo = max(0, tick-t_causal_tick)
    t_diff = tick-tick_lo
    tick_hi = min(total_ticks, tick+t_acause_tick+1)

    for i in prange(spike_train.shape[0]):
        fired = np.where(spike_train[i][tick_lo:tick_hi])[0]
        if fired.size:
            t_fired = fired[np.argmin(np.abs(fired-t_diff))]
            synapses[i] += f_stdp[t_causal_tick+t_fired-t_diff]
        elif decay:
            synapses[i] -= a_linear_decay

    np.clip(synapses, out=synapses, a_min=w_min, a_max=w_max)

@njit(parallel=True)
def update_lateral(synapses_to, synapses_from, spike_train, tick, decay=True):
    tick_lo = max(0, tick-t_causal_tick)
    t_diff = tick-tick_lo
    
    for i in prange(spike_train.shape[0]):
        fired = np.where(spike_train[i][tick_lo:tick])[0]
        if fired.size:
            dtick = t_diff-fired[-1]
            synapses_from[i] += f_acause[dtick]
            synapses_to[i] += f_causal[-dtick]
        elif decay:
            synapses_from[i] += a_linear_decay
            synapses_to[i] += a_linear_decay

    np.clip(synapses_to, out=synapses_to, a_min=w_min, a_max=w_max)
    np.clip(synapses_from, out=synapses_from, a_min=w_min, a_max=w_max)

def init(shape):
    return np.random.normal(param.w_init_mean, param.w_init_dev, shape)

def init_lateral(size):
    new_synapses = np.ones((size, size))-np.eye(size)
    new_synapses *= 0.2
    return new_synapses