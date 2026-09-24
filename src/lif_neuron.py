"""
Leaky Integrate-and-Fire (LIF) neuron.

This is the "Layer 1" model described in the README: a compromise between
- McCulloch-Pitts (a single weighted sum + threshold, no concept of time), and
- Hodgkin-Huxley (a full system of differential equations for ion channels).

Mental model — a leaky bucket of water:
    - Input current from other neurons pours water in (integrate).
    - The bucket has a hole in the bottom, so the water level drifts back
      down toward a resting level even with no input at all (leaky).
    - Fill it past the rim (threshold) and it "fires" (produces a spike) —
      then it's emptied back to a reset level and briefly can't fire again
      (refractory period), mimicking a real neuron's recovery time.

The governing equation (a simplified membrane-potential ODE):

    tau_m * dv/dt = -(v - v_rest) + r_m * i_input

Read it as two competing forces on the membrane potential `v`:
    -(v - v_rest)   -> pulls v back toward rest (the "leak")
    r_m * i_input   -> pushes v up when there's input current (the "integrate")
    tau_m           -> how fast this all happens (a bigger tau_m = slower,
                       more sluggish neuron)
"""


class LIFNeuron:
    def __init__(
        self,
        tau_m=10.0,          # membrane time constant (ms) - how fast v reacts
        v_rest=-65.0,        # resting potential (mV) - where v drifts to with no input
        v_reset=-70.0,       # potential right after a spike (mV)
        v_threshold=-50.0,   # fire when v crosses this (mV)
        r_m=10.0,            # membrane resistance - how much input current raises v
        refractory_period=2.0,  # ms the neuron is forced silent after firing
        dt=0.1,              # simulation time step (ms)
    ):
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.r_m = r_m
        self.refractory_period = refractory_period
        self.dt = dt

        # State that changes as the simulation runs.
        self.v = v_rest            # current membrane potential
        self.refractory_timer = 0.0  # ms remaining until it can fire again
        self.time = 0.0            # ms elapsed since this neuron was created
        self.spike_times = []      # every timestamp (ms) this neuron fired at

    def step(self, i_input):
        """
        Advance the neuron by one `dt`, given an input current `i_input`
        (the weighted sum of whatever upstream neurons/sensors are feeding
        it this instant).

        Returns True if the neuron fired a spike on this step, else False.
        """
        self.time += self.dt

        # While in the refractory period, the neuron ignores input entirely
        # and stays pinned at v_reset — this is what makes a real neuron's
        # firing rate saturate rather than fire infinitely fast under a huge
        # input current.
        if self.refractory_timer > 0:
            self.refractory_timer -= self.dt
            self.v = self.v_reset
            return False

        # Euler-integrate the membrane equation by one small time step:
        # dv = [ -(v - v_rest) + r_m * i_input ] / tau_m  * dt
        dv = (-(self.v - self.v_rest) + self.r_m * i_input) / self.tau_m
        self.v += dv * self.dt

        if self.v >= self.v_threshold:
            self.spike_times.append(self.time)
            self.v = self.v_reset
            self.refractory_timer = self.refractory_period
            return True

        return False


def _run_trace(neuron, i_input, duration_ms):
    """Step `neuron` for `duration_ms`, returning (times, voltages, spikes)."""
    times, voltages = [], []
    steps = int(duration_ms / neuron.dt)
    for _ in range(steps):
        neuron.step(i_input)
        times.append(neuron.time)
        voltages.append(neuron.v)
    return times, voltages


def _ascii_plot(times, voltages, v_rest, v_threshold, width=70, height=12):
    """
    Render a voltage trace as a text sparkline, so you can see the neuron
    charging up and firing without needing matplotlib installed.
    """
    v_min = min(voltages + [v_rest])
    v_max = max(voltages + [v_threshold]) + 1  # headroom so spikes don't clip
    v_span = v_max - v_min or 1.0

    # Downsample to `width` columns by taking the max voltage in each bucket
    # (max, not average, so brief spikes stay visible instead of being
    # smoothed away).
    bucket_size = max(1, len(voltages) // width)
    columns = [
        max(voltages[i : i + bucket_size])
        for i in range(0, len(voltages), bucket_size)
    ][:width]

    rows = []
    for row in range(height, 0, -1):
        threshold_v = v_min + v_span * row / height
        line = "".join("#" if c >= threshold_v else " " for c in columns)
        rows.append(f"{threshold_v:6.1f} |{line}")
    rows.append(" " * 7 + "-" * len(columns))
    return "\n".join(rows)


if __name__ == "__main__":
    v_rest, v_threshold = -65.0, -50.0

    print("=" * 72)
    print("Case 1: no input -> potential just sits at rest (nothing pushes it up)")
    print("=" * 72)
    neuron = LIFNeuron()
    times, voltages = _run_trace(neuron, i_input=0.0, duration_ms=50)
    print(_ascii_plot(times, voltages, v_rest, v_threshold))
    print(f"Spikes: {len(neuron.spike_times)}\n")

    print("=" * 72)
    print("Case 2: sub-threshold input -> potential rises, but never fires")
    print("=" * 72)
    neuron = LIFNeuron()
    times, voltages = _run_trace(neuron, i_input=1.0, duration_ms=50)
    print(_ascii_plot(times, voltages, v_rest, v_threshold))
    print(f"Spikes: {len(neuron.spike_times)}\n")

    print("=" * 72)
    print("Case 3: supra-threshold input -> fires repeatedly (see the sawtooth)")
    print("=" * 72)
    neuron = LIFNeuron()
    times, voltages = _run_trace(neuron, i_input=2.5, duration_ms=50)
    print(_ascii_plot(times, voltages, v_rest, v_threshold))
    print(f"Spikes: {len(neuron.spike_times)} at t={[round(t, 1) for t in neuron.spike_times]} ms\n")

    print("=" * 72)
    print("Case 4: stronger input -> higher firing rate (same duration, more spikes)")
    print("=" * 72)
    neuron = LIFNeuron()
    _run_trace(neuron, i_input=5.0, duration_ms=50)
    print(f"Spikes: {len(neuron.spike_times)} (compare to Case 3's count above)")
