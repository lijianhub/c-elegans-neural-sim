"""
The measured C. elegans connectome, loaded as two 302 x 302 matrices.

This is the "Layer 2" of the README made concrete: instead of wiring two
neurons by hand (as chemotaxis_circuit.py does), we read every synapse that
has been counted in the real worm from data/connectome_edges.csv.

    chemical[i, j]  synapses from neuron i (sender) to neuron j (receiver).
                    The sign says what the sender does to the receiver:
                    positive = excites, negative = inhibits.
    gap[i, j]       gap junctions between i and j (symmetric): a direct
                    electrical link that pulls the two voltages together.

The edge file only records *that* a synapse exists and how many were counted,
not whether it excites or inhibits. We use the one well-established rule from
the literature: the GABA-releasing neurons are inhibitory and every other
neuron is excitatory. It is a simplification (some chemical synapses in the
worm act differently), and it is the main assumption the full simulation
rests on.
"""

import csv
import os
import re

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# GABAergic neurons (McIntire et al., 1993, Nature 364:337): the D-type motor
# neurons, the four RME head neurons, AVL and DVB.
_GABA = re.compile(r"^(DD\d+|VD\d+|RME[LRDV]|AVL|DVB)$")


def is_inhibitory(name):
    return bool(_GABA.match(name))


class Connectome:
    def __init__(self, names, chemical, gap):
        self.names = names
        self.index = {n: i for i, n in enumerate(names)}
        self.chemical = chemical
        self.gap = gap

    def ids(self, *names):
        """Row/column indices for the given neuron names."""
        return np.array([self.index[n] for n in names])


def load_connectome():
    with open(os.path.join(DATA_DIR, "neuron_positions.csv"), newline="") as f:
        names = [r["name"] for r in csv.DictReader(f)]
    index = {n: i for i, n in enumerate(names)}

    chemical = np.zeros((len(names), len(names)))
    gap = np.zeros((len(names), len(names)))
    with open(os.path.join(DATA_DIR, "connectome_edges.csv"), newline="") as f:
        for r in csv.DictReader(f):
            i, j, count = index[r["source"]], index[r["target"]], float(r["synapses"])
            if i == j:
                continue  # a neuron synapsing onto itself: ignored here
            if r["type"] == "chemical":
                chemical[i, j] += -count if is_inhibitory(r["source"]) else count
            else:
                # The file records nearly every gap junction once in each
                # direction with the same count, so take the larger of the two
                # rather than adding them, and make sure both directions exist.
                gap[i, j] = max(gap[i, j], count)
                gap[j, i] = max(gap[j, i], count)
    return Connectome(names, chemical, gap)


if __name__ == "__main__":
    c = load_connectome()
    n_chem = int((c.chemical != 0).sum())
    n_gap = int((np.triu(c.gap) > 0).sum())
    n_inh = int((c.chemical < 0).sum())
    print(f"{len(c.names)} neurons")
    print(f"{n_chem} chemical connections ({n_inh} inhibitory), {n_gap} gap junctions")
    for pre, post in (("AWCL", "AIYL"), ("AWAL", "AIZL"), ("AIYL", "AIZL")):
        print(f"{pre} -> {post}: {c.chemical[c.index[pre], c.index[post]]:.0f} synapses")
