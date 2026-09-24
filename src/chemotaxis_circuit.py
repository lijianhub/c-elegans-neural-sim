"""
A minimal simulation of the C. elegans chemotaxis circuit shown in the
README's diagram:

    Environment -> Sensory neurons -> Interneurons -> Motor neurons -> Muscles
         ^                                                                |
         '---------------------- body moves -----------------------------'

This implements all four layers from the README with the smallest circuit
that still shows the real mechanism:

    Layer 1 (single neuron)  -> LIFNeuron, imported from lif_neuron.py
    Layer 2 (connectome)     -> two interneurons, each wired to one signal:
                                   AIY gets excited when concentration RISES
                                   AIZ gets excited when concentration FALLS
    Layer 3 (time-stepping)  -> the main loop in `run_simulation`
    Layer 4 (behaviour)      -> whichever interneuron spikes more this step
                                 decides "keep straight" vs "turn"

The key, counter-intuitive point from the README: the circuit never looks at
the absolute concentration. It only ever asks "is it higher or lower than
one instant ago" (dC/dt) - that single bit is enough to climb a gradient
with no map and no memory of position.
"""

import math
import random

from lif_neuron import LIFNeuron


class Environment:
    """
    A single food source that creates a concentration gradient, modelled as
    exponential decay with distance from `food_pos`. This stands in for the
    "chemical gradient" box in the README's diagram - it doesn't simulate
    diffusion, it just gives a smooth, continuous field for the worm to
    climb.

    Exponential decay (rather than a Gaussian) is a deliberate choice: its
    *relative* rate of change with distance, -1/spread, is constant no
    matter how far from the food you are. That keeps the sensory signal
    usable across the whole arena instead of vanishing a few `spread`
    lengths out, the way a Gaussian's would.
    """

    def __init__(self, food_pos=(0.0, 0.0), peak=100.0, spread=15.0):
        self.food_pos = food_pos
        self.peak = peak
        self.spread = spread

    def concentration_at(self, position):
        dx = position[0] - self.food_pos[0]
        dy = position[1] - self.food_pos[1]
        dist = math.hypot(dx, dy)
        return self.peak * math.exp(-dist / self.spread)


class Worm:
    """
    Tracks the worm's position and heading, and exposes the one motor
    action this circuit needs: move forward, optionally turning first.
    """

    def __init__(self, position=(-40.0, 30.0), heading_rad=0.0, speed=1.0):
        self.position = position
        self.heading = heading_rad
        self.speed = speed
        self.path = [position]

    def turn(self, delta_rad):
        self.heading += delta_rad

    def advance(self, dt):
        x, y = self.position
        x += math.cos(self.heading) * self.speed * dt
        y += math.sin(self.heading) * self.speed * dt
        self.position = (x, y)
        self.path.append(self.position)


class ChemotaxisCircuit:
    """
    The two-interneuron decision circuit from the diagram: AIY (excited by a
    rising gradient) vs AIZ (excited by a falling gradient). Whichever one
    is actually spiking on a given step casts the "vote" for that step's
    motor decision - this is a direct stand-in for the real
    AWA/AWC -> AIY/AIZ -> motor neuron pathway.
    """

    # How strongly a given FRACTIONAL change in concentration
    # (d_concentration / concentration) drives current into the
    # interneurons. Using the fractional change rather than the raw delta
    # is what keeps the circuit equally sensitive near the food and far
    # from it - a real chemoreceptor's response is closer to logarithmic
    # than linear too (Weber-Fechner-style scaling is common in sensory
    # biology). GAIN itself is tuned only so a typical gradient step drives
    # the LIF neuron's steady-state potential past threshold - see
    # lif_neuron.py's demo for why an input current needs to exceed
    # (v_threshold - v_rest) / r_m = 1.5 to ever fire at all.
    GAIN = 80.0

    def __init__(self, dt=0.1, sub_steps=50):
        self.dt = dt
        self.sub_steps = sub_steps  # LIF steps run per simulation tick
        self.aiy = LIFNeuron(dt=dt)  # "keep going straight" neuron
        self.aiz = LIFNeuron(dt=dt)  # "turn" neuron
        self.prev_concentration = None
        # What each neuron did during the most recent `decide` call (membrane
        # potential and spike flag per LIF sub-step), kept only so a
        # visualisation can replay it. Nothing in the circuit reads it.
        self.last_activity = self._activity(
            [self.aiy.v] * sub_steps, [False] * sub_steps,
            [self.aiz.v] * sub_steps, [False] * sub_steps,
        )

    @staticmethod
    def _activity(aiy_v, aiy_fired, aiz_v, aiz_fired):
        return {
            "aiy_v": aiy_v, "aiy_fired": aiy_fired,
            "aiz_v": aiz_v, "aiz_fired": aiz_fired,
        }

    @staticmethod
    def _drive(neuron, current, sub_steps):
        """Run one neuron for `sub_steps` and return (potentials, spike flags)."""
        potentials, fired = [], []
        for _ in range(sub_steps):
            fired.append(neuron.step(current))
            potentials.append(neuron.v)
        return potentials, fired

    def decide(self, concentration_now):
        """
        Feed the change in sensed concentration into the two interneurons
        and return True if this step's net vote is "turn".
        """
        if self.prev_concentration is None:
            self.prev_concentration = concentration_now
            return False  # no history yet - first step, just go straight

        # Fractional change since last step, e.g. -0.03 means "3% weaker
        # than a moment ago". Floor the denominator so this stays well
        # behaved even when the worm is so far away concentration is
        # near zero (matching the real worm: no reliable signal that far
        # out, so no meaningful drive either way).
        frac_change = (concentration_now - self.prev_concentration) / max(
            self.prev_concentration, 1e-3
        )
        self.prev_concentration = concentration_now

        # This is the crux of the whole circuit: split the rate of change
        # into two non-negative drives, one per interneuron. Only one of
        # these can be nonzero at a time, because a change is either a
        # rise or a fall (or exactly zero, driving neither).
        rising_drive = max(frac_change, 0.0) * self.GAIN
        falling_drive = max(-frac_change, 0.0) * self.GAIN

        aiy_v, aiy_fired = self._drive(self.aiy, rising_drive, self.sub_steps)
        aiz_v, aiz_fired = self._drive(self.aiz, falling_drive, self.sub_steps)
        self.last_activity = self._activity(aiy_v, aiy_fired, aiz_v, aiz_fired)
        aiy_spikes = sum(aiy_fired)
        aiz_spikes = sum(aiz_fired)

        # Motor decision: AIZ (falling / "getting worse") has to actually
        # out-vote AIY to trigger a turn - a tie or silence means "stay
        # straight", matching the README's "rising -> suppress turning".
        return aiz_spikes > aiy_spikes


# The worm has "eaten" once it is this close to the food, and stops there.
# The circuit itself has no notion of arriving: it only ever compares "better
# or worse than a moment ago", so on its own the worm would keep wandering
# around the food source instead of stopping on it.
EAT_RADIUS = 2.0


def run_simulation(steps=1500, dt=0.5, seed=7, history=None, circuit=None):
    """
    Run the closed loop until the worm reaches the food (within EAT_RADIUS)
    or `steps` run out. If a list is passed as `history`, one dict per step
    is appended to it (concentration sensed, whether the worm turned, and the
    interneurons' activity) so the run can be replayed as an animation.
    `history[i]` describes the decision made at `worm.path[i]`.

    `circuit` is whatever decides "turn or not" from the sensed concentration:
    anything with `decide(concentration)` and `last_activity`. It defaults to
    the two-interneuron ChemotaxisCircuit; full_chemotaxis.py passes in a
    circuit built from the whole connectome instead.
    """
    random.seed(seed)

    env = Environment(food_pos=(0.0, 0.0))
    worm = Worm(position=(-40.0, 30.0), heading_rad=random.uniform(0, 2 * math.pi))
    circuit = circuit or ChemotaxisCircuit(dt=0.1)

    turn_count = 0
    start_distance = math.dist(worm.position, env.food_pos)

    for _ in range(steps):
        if math.dist(worm.position, env.food_pos) <= EAT_RADIUS:
            break  # reached the food

        c_now = env.concentration_at(worm.position)

        turned = circuit.decide(c_now)
        if turned:
            # A real pirouette reorients to a mostly-new random heading,
            # not a small correction - mimicking that "sharp turn" behaviour.
            worm.turn(random.uniform(-math.pi, math.pi))
            turn_count += 1

        if history is not None:
            history.append(
                {"concentration": c_now, "turned": turned, **circuit.last_activity}
            )

        worm.advance(dt)

    end_distance = math.dist(worm.position, env.food_pos)
    return worm, env, turn_count, start_distance, end_distance


def _ascii_trajectory(worm, env, width=60, height=24):
    """
    Render the worm's path and the food source on a text grid, so you can
    see the gradient-climbing behaviour without a plotting library.
    """
    xs = [p[0] for p in worm.path] + [env.food_pos[0]]
    ys = [p[1] for p in worm.path] + [env.food_pos[1]]
    x_min, x_max = min(xs) - 2, max(xs) + 2
    y_min, y_max = min(ys) - 2, max(ys) + 2

    grid = [[" "] * width for _ in range(height)]

    def to_cell(pos):
        x, y = pos
        col = int((x - x_min) / (x_max - x_min or 1) * (width - 1))
        row = int((y_max - y) / (y_max - y_min or 1) * (height - 1))
        return row, col

    # Draw the path first as dots, oldest to newest, so later work (start
    # marker, food marker) draws on top and is never overwritten.
    for pos in worm.path:
        row, col = to_cell(pos)
        grid[row][col] = "."

    r, c = to_cell(worm.path[0])
    grid[r][c] = "S"  # Start
    r, c = to_cell(worm.path[-1])
    grid[r][c] = "E"  # End
    r, c = to_cell(env.food_pos)
    grid[r][c] = "F"  # Food

    return "\n".join("".join(row) for row in grid)


if __name__ == "__main__":
    worm, env, turn_count, start_distance, end_distance = run_simulation()

    print("Legend: S = start, E = end, F = food source, . = path")
    print(_ascii_trajectory(worm, env))
    print()
    print(f"Turns triggered:     {turn_count}")
    print(f"Distance to food, start -> end: {start_distance:.1f} -> {end_distance:.1f}")
    print(f"Steps taken:         {len(worm.path) - 1}")
    if end_distance <= EAT_RADIUS:
        print("Reached the food")
    else:
        print(
            "Did not reach the food this run (try a different seed - it's a random walk "
            "biased by the gradient, not a guaranteed solver)"
        )
