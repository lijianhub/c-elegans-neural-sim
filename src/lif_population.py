"""
Many Leaky Integrate-and-Fire neurons at once.

This is exactly the model in lif_neuron.py - same equation, same constants,
same refractory rule - but every neuron's state lives in a numpy array so 302
of them can be advanced together in one step. It exists because looping over
LIFNeuron objects in Python is far too slow once every neuron in the worm is
simulated.

Running `python src/lif_population.py` checks that a population of one
neuron produces identical spikes to LIFNeuron.
"""

import numpy as np


class LIFPopulation:
    def __init__(
        self,
        n,
        tau_m=10.0,
        v_rest=-65.0,
        v_reset=-70.0,
        v_threshold=-50.0,
        r_m=10.0,
        refractory_period=2.0,
        dt=0.1,
    ):
        self.n = n
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.r_m = r_m
        self.refractory_period = refractory_period
        self.dt = dt

        self.v = np.full(n, v_rest)
        self.refractory_timer = np.zeros(n)

    def step(self, i_input):
        """
        Advance every neuron by one `dt`. `i_input` is the input current for
        each neuron (an array of length n). Returns a boolean array: True where
        that neuron fired on this step.
        """
        refractory = self.refractory_timer > 0
        self.refractory_timer[refractory] -= self.dt

        # Neurons in their refractory period ignore input and sit at v_reset;
        # everyone else integrates one Euler step of the membrane equation.
        dv = (-(self.v - self.v_rest) + self.r_m * i_input) / self.tau_m
        self.v = np.where(refractory, self.v_reset, self.v + dv * self.dt)

        fired = ~refractory & (self.v >= self.v_threshold)
        self.v[fired] = self.v_reset
        self.refractory_timer[fired] = self.refractory_period
        return fired


if __name__ == "__main__":
    from lif_neuron import LIFNeuron

    current = np.concatenate([np.zeros(200), np.full(3000, 2.5), np.zeros(500)])
    single = LIFNeuron()
    population = LIFPopulation(1)
    same = True
    for i_in in current:
        a = single.step(i_in)
        b = population.step(np.array([i_in]))[0]
        same &= (a == b) and abs(single.v - population.v[0]) < 1e-9
    print(f"LIFNeuron and LIFPopulation(1) agree on every step: {bool(same)}")
    print(f"spikes: {len(single.spike_times)}")
