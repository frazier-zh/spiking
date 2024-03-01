### Zihang
### ATGroup, NUS

import numpy as np
from .parameters import synapse_param as param

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
            synapses[i] += (f_stdp[t_causal_tick+t_fired-t_diff]) #*(1+(2*np.random.rand()-1)*0.1)
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
            synapses_from[i] += f_acause[dtick] #*(1+(2*np.random.rand()-1)*0.1)
            synapses_to[i] += f_causal[-dtick] #*(1+(2*np.random.rand()-1)*0.1)
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