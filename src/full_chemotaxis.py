"""
The full version of the chemotaxis simulation: every one of the worm's 302
neurons, wired exactly as in the measured connectome.

chemotaxis_circuit.py is the simple version - two neurons wired by hand, to
show the idea. This file keeps the same environment, the same worm and the
same closed loop, and swaps only the circuit:

    simple : rise -> AIY,  fall -> AIZ,  AIZ out-votes AIY -> turn
    full   : odor change -> AWA / AWC sensory neurons
             -> activity spreads through all 302 LIF neurons along the real
                synapses and gap junctions (connectome.py, lif_population.py)
             -> the backward vs forward command interneurons decide the action

What is real and what is chosen by hand
---------------------------------------
Real: which neurons exist and how many synapses connect each pair.
Assumed: (1) GABA neurons inhibit and all others excite (see connectome.py);
(2) the odor is fed only to AWA (when it gets stronger) and AWC (when it gets
weaker); (3) "turn" means the backward command neurons (AVA, AVD, AVE) fire
more than the forward ones (AVB, PVC); (4) one global number,
SYNAPSE_SCALE, sets how strongly a synapse count turns into current.

SYNAPSE_SCALE is a genuine free parameter. The real connectome gives counts,
not strengths, and the network is very sensitive to it: too weak and the
signal never leaves the sensory neurons, too strong and every neuron fires no
matter what the input is (like a seizure). It was set just below that
runaway point, where the network still tells the two sensory inputs apart.

What this version does and does not show
----------------------------------------
Measured over 20 random starts (1500 steps each), the worm reaches the food:

    simple two-neuron circuit                      16 of 20
    full connectome (SYNAPSE_SCALE = 0.115)         4 of 20
    control: turns at random, ignores the odor      0 to 5 of 20

So the raw connectome does respond to the odor - a falling signal through AWC
reaches the backward command neurons while a rising one through AWA does not -
but only weakly better than chance. The reason is visible in the activity:
a stretch of "getting weaker" recruits more and more of the network, and
that activity keeps going for 25+ steps after the odor turns "getting
stronger", so the worm keeps turning when it should not. With only GABA
neurons inhibitory there is too little braking. A global inhibition term did
not fix it (the network just went silent or stayed saturated).

The wiring diagram alone is not enough: a working model also needs to know
which synapses excite and which inhibit (many glutamate synapses inhibit in
the real worm) and how strong each one is. That is the gap between a
connectome and a working circuit, and it is why the simple version, where
the signs and strengths are set by hand, climbs the gradient and this one
mostly does not.
"""

import numpy as np

from chemotaxis_circuit import ChemotaxisCircuit
from connectome import load_connectome
from lif_population import LIFPopulation

# Sensory neurons that receive the odor signal.
RISE_NEURONS = ("AWAL", "AWAR")   # AWA: fires when the odor gets stronger
FALL_NEURONS = ("AWCL", "AWCR")   # AWC: fires when the odor is removed / gets weaker
# Command interneurons that decide the action.
BACKWARD_NEURONS = ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")
FORWARD_NEURONS = ("AVBL", "AVBR", "PVCL", "PVCR")


class FullConnectomeCircuit:
    GAIN = ChemotaxisCircuit.GAIN  # same odor -> current conversion as the simple version
    SYNAPSE_SCALE = 0.115  # current added to a receiver per synapse, per spike (see docstring)
    GAP_SCALE = 0.002      # current per gap junction per mV of voltage difference
    TAU_SYN = 5.0          # ms; how long a synaptic current lingers after a spike

    def __init__(self, dt=0.1, sub_steps=50):
        self.dt = dt
        self.sub_steps = sub_steps  # LIF steps run per simulation tick
        self.connectome = load_connectome()
        c = self.connectome
        self.population = LIFPopulation(len(c.names), dt=dt)
        self.synaptic_current = np.zeros(len(c.names))

        self._to_receiver = c.chemical.T * self.SYNAPSE_SCALE  # [receiver, sender]
        self._gap = c.gap * self.GAP_SCALE
        self._gap_total = self._gap.sum(axis=1)
        self._syn_decay = np.exp(-dt / self.TAU_SYN)

        self._rise = c.ids(*RISE_NEURONS)
        self._fall = c.ids(*FALL_NEURONS)
        self._backward = c.ids(*BACKWARD_NEURONS)
        self._forward = c.ids(*FORWARD_NEURONS)

        self.prev_concentration = None
        # Spikes each neuron fired during the most recent `decide` call, kept
        # only so a visualisation can replay it. Nothing here reads it.
        self.last_activity = {"spikes": np.zeros(len(c.names))}

    def decide(self, concentration_now):
        """Return True if this step's net vote is "turn"."""
        if self.prev_concentration is None:
            self.prev_concentration = concentration_now
            return False  # no history yet - first step, just go straight

        frac_change = (concentration_now - self.prev_concentration) / max(
            self.prev_concentration, 1e-3
        )
        self.prev_concentration = concentration_now

        external = np.zeros(len(self.connectome.names))
        external[self._rise] = max(frac_change, 0.0) * self.GAIN
        external[self._fall] = max(-frac_change, 0.0) * self.GAIN

        pop = self.population
        spikes = np.zeros(len(external))
        for _ in range(self.sub_steps):
            gap_current = self._gap @ pop.v - self._gap_total * pop.v
            fired = pop.step(self.synaptic_current + gap_current + external)
            self.synaptic_current = self.synaptic_current * self._syn_decay + self._to_receiver @ fired
            spikes += fired
        self.last_activity = {"spikes": spikes}

        return spikes[self._backward].sum() > spikes[self._forward].sum()


if __name__ == "__main__":
    import math

    from chemotaxis_circuit import EAT_RADIUS, _ascii_trajectory, run_simulation

    worm, env, turn_count, start_distance, end_distance = run_simulation(
        circuit=FullConnectomeCircuit()
    )
    print("Legend: S = start, E = end, F = food source, . = path")
    print(_ascii_trajectory(worm, env))
    print()
    print(f"Turns triggered:     {turn_count}")
    print(f"Distance to food, start -> end: {start_distance:.1f} -> {end_distance:.1f}")
    print(f"Steps taken:         {len(worm.path) - 1}")
    print("Reached the food" if end_distance <= EAT_RADIUS else "Did not reach the food this run")
