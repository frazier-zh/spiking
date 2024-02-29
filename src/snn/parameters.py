### 2019 Sep 04
### Zihang
### ATGroup, NUS

dt = 0.1 # msec
total_time = 20 # msec try:5
total_ticks = int(total_time/dt)

pixel_x = 5
pixel_y = 5
pixels = pixel_x*pixel_y

n_classes = 25
n_layers = 2
n_layer_neurons = [pixels, n_classes]
#n_layer_neurons = [pixels, 50, n_classes]

class neuron_param:
    # refratory period
    t_ref = 2 # msec
    amp = 0.5

    # LIF model parameter
    tau = 20 # msec

    # threshold and resting potential
    v_rest = 0
    v_th = 0.2
    v_spike = 2

    # adaptive threshold parameter
    tau_ath = 100 # msec
    v_ath = 0.02

    # parameters in tick unit
    t_ref_tick = int(t_ref/dt)
    tau_tick = int(tau/dt)
    tau_ath_tick = int(tau_ath/dt)


class synapse_param:
    # STDP parameters
    t_causal = 2 # msec
    tau_causal = 1 # msec
    A_causal = 1e-2

    t_acause = 2
    tau_acause = 1
    A_acause = A_causal

    # maximum and minimum synapses
    w_max = 1
    w_min = 0.001
    
    w_init_mean = 0.5
    w_init_dev = 0.1

    # synapse efficacy decay
    w_th = 0.1
    eta_exp = 7.5e-4 # decay rate
    eta_lin = 1e-3

    # parameters in tick unit
    t_causal_tick = int(t_causal/dt)
    tau_causal_tick = int(tau_causal/dt)
    t_acause_tick = int(t_acause/dt)
    tau_acause_tick = int(tau_acause/dt)

    total_ticks = total_ticks
