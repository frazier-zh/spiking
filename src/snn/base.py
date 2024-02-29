### 2019 Sep 10
### Zihang
### ATGroup, NUS

import numpy as np
from .neuron import adaptiva_neuron_dynamics, adaptive_neuron_init, adaptiva_neuron_reset
from . import synapse

class preceptron_layer():
    def __init__(self, input_shape, output_shape,
                 homeostasis_spike_count=4):
        """
        input_shape: int
        output_shape: int
        homeostasis_spike_count: int, 0 for disable
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.synapse_update_enabled = True
        self.homeostasis_spike_count = homeostasis_spike_count

        self.init()

    def init(self):
        self.neurons = adaptive_neuron_init(self.output_shape)
        self.synapses = synapse.init((self.output_shape, self.input_shape))

    def disable_synapse_update(self):
        self.synapse_update_enabled = False

    def enable_synapse_update(self):
        self.synapse_update_enabled = True
        
    def forward(self, input):
        """
        input: (spike_train_length, input_shape)
        """
        spike_train_length = input.shape[0]
        output = np.zeros((self.output_shape, spike_train_length), dtype=bool)

        adaptiva_neuron_reset(self.neurons)

        homeostasis_spike_counter = 0
        for t in range(spike_train_length):
            # neuron dynamics
            synapse_output = np.dot(self.synapses, input[:, t]) / self.input_shape
            neuron_output = adaptiva_neuron_dynamics(self.neurons, synapse_output)
            output[:, t] = neuron_output

            # Winner-take-all for every layer
            fired_location = np.where(neuron_output)[0]

            for loc in fired_location:
                if self.synapse_update_enabled:
                    synapse.update(self.synapses[loc], input, t)

            # for homeostasis
            if self.homeostasis_spike_count:
                homeostasis_spike_counter += len(fired_location)
                if homeostasis_spike_counter>=self.homeostasis_spike_count:
                    break

        return output

class preceptron_lateral_layer():
    def __init__(self, input_shape, output_shape,
                 homeostasis_spike_count=4):
        """
        input_shape: int
        output_shape: int
        homeostasis_spike_count: int, 0 for disable
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.synapse_update_enabled = True
        self.homeostasis_spike_count = homeostasis_spike_count

        self.init()

    def init(self):
        self.neurons = adaptive_neuron_init(self.output_shape)
        self.synapses = synapse.init((self.output_shape, self.input_shape))
        self.lateral_synapses = synapse.init_lateral(self.output_shape)

    def disable_synapse_update(self):
        self.synapse_update_enabled = False

    def enable_synapse_update(self):
        self.synapse_update_enabled = True
        
    def forward(self, input):
        """
        input: (spike_train_length, input_shape)
        """
        spike_train_length = input.shape[0]
        output = np.zeros((self.output_shape, spike_train_length), dtype=bool)

        adaptiva_neuron_reset(self.neurons)

        homeostasis_spike_counter = 0
        for t in range(spike_train_length):
            # neuron dynamics
            synapse_output = np.dot(self.synapses, input[:, t]) / self.input_shape
            lateral_synapse_output = np.dot(self.lateral_synapses, output[:, t-1])
            neuron_output = adaptiva_neuron_dynamics(self.neurons, synapse_output - lateral_synapse_output)
            output[:, t] = neuron_output

            # Winner-take-all for every layer
            fired_location = np.where(neuron_output)[0]

            for loc in fired_location:
                if self.synapse_update_enabled:
                    synapse.update(self.synapses[loc], input, t)
                    synapse.update_lateral(self.lateral_synapses[loc,:], self.lateral_synapses[:,loc], output, t)

            # for homeostasis
            if self.homeostasis_spike_count:
                homeostasis_spike_counter += len(fired_location)
                if homeostasis_spike_counter>=self.homeostasis_spike_count:
                    break

        return output
