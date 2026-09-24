"""
Render the chemotaxis simulation as an animated GIF: output_results/chemotaxis.gif

Left        : the arena. Background shading is the odor concentration, the line
              is the worm's path so far, orange dots are the moments it turned.
Right, top  : all 302 neurons of the worm, drawn at their real positions in the
              body (top strip = whole worm, bottom = zoom on the head, where
              most of them sit). Only the neurons this circuit is modelled on
              light up; the other 294 stay dark because they are not simulated.
              A lit neuron glows and fades out, like the neuron-by-neuron
              views used for whole-brain fly simulations.
Right, below: the membrane potentials of the two LIF interneurons (AIY, AIZ).

Neuron positions: data/neuron_positions.csv (see the README for the source).

This is the only part of the repo that needs third-party packages:

    pip install matplotlib pillow
    python src/make_gif.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # draw to memory, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba

from chemotaxis_circuit import EAT_RADIUS, ChemotaxisCircuit, run_simulation

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "..", "output_results", "chemotaxis.gif")
POSITIONS_PATH = os.path.join(HERE, "..", "data", "neuron_positions.csv")

GREEN, ORANGE, INK, GREY = "#1f7a4d", "#b5541c", "#1d2a24", "#8a918d"
LIT_GREEN, LIT_ORANGE, LIT_BLUE = "#5dffa8", "#ff9d4d", "#7fd4ff"  # colour when fully lit
PANEL, DIM = "#0e1512", "#3e5148"
V_THRESHOLD = -50.0  # LIFNeuron's default firing threshold (mV)
WINDOW_TICKS = 24    # how many simulation steps of membrane potential to show
TARGET_FRAMES = 170  # roughly how many frames the GIF gets: more = slower, smoother playback
FADE_TICKS = 8       # a lit neuron dims to ~37% after this many simulation steps
FPS = 12             # playback speed; lower is slower

# The real neurons that stand in for the simulated ones. The sensory pair is
# lit by the size of the sensed change; AIY / AIZ light up when the LIF neuron
# of the same name spikes (both the left and right neuron of each pair).
GROUPS = [
    ("sensory", ("AWAL", "AWAR", "AWCL", "AWCR"), LIT_BLUE),
    ("AIY", ("AIYL", "AIYR"), LIT_GREEN),
    ("AIZ", ("AIZL", "AIZR"), LIT_ORANGE),
]
HEAD_Y_RANGE = (-322.0, -198.0)  # micrometres along the body, for the head zoom


def _load_neurons():
    """Return (names, positions) where positions[:, 0] runs head -> tail and [:, 1] is dorsal-ventral."""
    with open(POSITIONS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    pos = np.array([[float(r["y_um"]), float(r["z_um"])] for r in rows])
    return names, pos


def _membrane_trace(history, i, key, sub_steps):
    """Membrane potential over the last WINDOW_TICKS steps, ending at step i."""
    first = max(0, i - WINDOW_TICKS + 1)
    v = np.concatenate([history[k][key + "_v"] for k in range(first, i + 1)])
    fired = np.concatenate([history[k][key + "_fired"] for k in range(first, i + 1)])
    t = first + np.arange(len(v)) / sub_steps  # x axis measured in steps
    return t, v, fired


def _activation(history):
    """
    How active each group of neurons was at each simulation step, 0 to 1.
    Returns an array of shape (steps, len(GROUPS)) in the order of GROUPS.

    sensory : how strong the sensed change in concentration was (either way)
    AIY/AIZ : 1 on any step where that LIF neuron spiked
    """
    act = np.zeros((len(history), len(GROUPS)))
    prev = None
    for k, h in enumerate(history):
        if prev is not None:
            frac = abs(h["concentration"] - prev) / max(prev, 1e-3)
            act[k, 0] = min(1.0, frac * ChemotaxisCircuit.GAIN / 3.0)
        prev = h["concentration"]
        act[k, 1] = float(any(h["aiy_fired"]))
        act[k, 2] = float(any(h["aiz_fired"]))
    return act


def _glow(act, i):
    """Brightness of each group at step i: each activation fades out over FADE_TICKS."""
    first = max(0, i - 4 * FADE_TICKS)
    age = i - np.arange(first, i + 1)
    return (act[first: i + 1] * np.exp(-age / FADE_TICKS)[:, None]).max(axis=0)


def _badge(history, i, every):
    """
    One-line plain-English description of what the worm is doing. Frames skip
    steps, so a turn counts if it happened at any step since the last frame.
    """
    if any(h["turned"] for h in history[max(0, i - every + 1): i + 1]):
        return "Smell getting weaker  ->  AIZ out-votes AIY  ->  TURN", ORANGE
    if i > 0 and history[i]["concentration"] > history[i - 1]["concentration"]:
        return "Smell getting stronger  ->  AIY fires  ->  keep going straight", GREEN
    return "Change too small to fire either neuron  ->  keep going straight", GREY


def _build_neuron_panel(fig, ax, names, pos):
    """
    Draw the dark panel with all 302 neurons. Returns (halos, cores): one
    scatter per GROUP, whose colours the animation updates every frame.
    """
    w = ax.get_position().width * fig.get_figwidth() * fig.dpi   # panel size in pixels,
    h = ax.get_position().height * fig.get_figheight() * fig.dpi  # so 1 unit = 1 pixel below
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # pixel coordinates, origin at the top-left
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    # Whole-worm strip (top) and head zoom (bottom) share one scale rule: keep
    # micrometres equal in both directions so the worm is not distorted.
    y_all = pos[:, 0]
    strip_scale = (w - 30) / (y_all.max() - y_all.min())
    strip_cy = 58
    z_mid = (pos[:, 1].max() + pos[:, 1].min()) / 2

    in_head = (y_all >= HEAD_Y_RANGE[0]) & (y_all <= HEAD_Y_RANGE[1])
    head_dz = np.ptp(pos[in_head, 1])
    zbox_top, zbox_bot = strip_cy + 32 + 16, h - 22  # the zoom box fills what is left below the strip
    zoom_cy = (zbox_top + zbox_bot) / 2
    zoom_scale = min((w - 30) / (HEAD_Y_RANGE[1] - HEAD_Y_RANGE[0]), (zbox_bot - zbox_top - 16) / head_dz)
    zoom_z_mid = (pos[in_head, 1].max() + pos[in_head, 1].min()) / 2
    zoom_x0 = (w - zoom_scale * (HEAD_Y_RANGE[1] - HEAD_Y_RANGE[0])) / 2

    def strip_xy(p):
        return 15 + (p[..., 0] - y_all.min()) * strip_scale, strip_cy - (p[..., 1] - z_mid) * strip_scale

    def zoom_xy(p):
        return zoom_x0 + (p[..., 0] - HEAD_Y_RANGE[0]) * zoom_scale, zoom_cy - (p[..., 1] - zoom_z_mid) * zoom_scale

    ax.text(10, 13, "All 302 neurons, at their real positions", color="white", fontsize=10, fontweight="bold", va="center")
    ax.text(w - 10, 13, "dark = not simulated", color="#6f8279", fontsize=8, ha="right", va="center")

    # Dim dots for every neuron, in both views.
    sx, sy = strip_xy(pos)
    ax.scatter(sx, sy, s=6, c=DIM, linewidths=0, zorder=2)
    zx, zy = zoom_xy(pos[in_head])
    ax.scatter(zx, zy, s=16, c=DIM, linewidths=0, zorder=2)

    # Bracket on the strip, box around the zoom, and lines joining them.
    bx0, bx1 = strip_xy(np.array([[HEAD_Y_RANGE[0], 0.0], [HEAD_Y_RANGE[1], 0.0]]))[0]
    zx0, zx1 = zoom_x0, zoom_x0 + zoom_scale * (HEAD_Y_RANGE[1] - HEAD_Y_RANGE[0])
    line = dict(color="#31443b", lw=1, zorder=1)
    ax.plot([bx0, bx1, bx1, bx0, bx0], [strip_cy - 32, strip_cy - 32, strip_cy + 32, strip_cy + 32, strip_cy - 32], **line)
    ax.plot([zx0, zx1, zx1, zx0, zx0], [zbox_top, zbox_top, zbox_bot, zbox_bot, zbox_top], **line)
    ax.plot([bx0, zx0], [strip_cy + 32, zbox_top], **line)
    ax.plot([bx1, zx1], [strip_cy + 32, zbox_top], **line)
    ax.text(w - 10, h - 10, "head, zoomed", color="#6f8279", fontsize=8, ha="right", va="center")

    # The neurons that light up: a halo (soft glow) and a core (the dot itself).
    idx = {n: k for k, n in enumerate(names)}
    halos, cores = [], []
    for label, members, colour in GROUPS:
        p = pos[[idx[m] for m in members]]
        gx, gy = zoom_xy(p)
        halos.append(ax.scatter(gx, gy, s=22 ** 2, c=[(*to_rgba(colour)[:3], 0.0)] * len(members), linewidths=0, zorder=3))
        cores.append(ax.scatter(gx, gy, s=7 ** 2, c=[to_rgba(DIM)] * len(members), linewidths=0, zorder=4))
        sxy = strip_xy(p)
        ax.scatter(sxy[0], sxy[1], s=14, c=colour, alpha=0.9, linewidths=0, zorder=3)
    labels = {  # label text, offset from the (mean) dot position in pixels
        "sensory": ("sensory\nAWA · AWC", (-46, -30)),
        "AIY": ("AIY", (40, 22)),
        "AIZ": ("AIZ", (42, -14)),
    }
    for (label, members, colour) in GROUPS:
        p = pos[[idx[m] for m in members]]
        gx, gy = zoom_xy(p)
        text, (dx, dy) = labels[label]
        ax.annotate(text, (gx.mean(), gy.mean()), xytext=(gx.mean() + dx, gy.mean() + dy), color=colour,
                    fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.8, shrinkA=2, shrinkB=6, alpha=0.8))
    return halos, cores


def main(max_steps=1300, seed=7):
    history = []
    worm, env, turns, start_d, end_d = run_simulation(steps=max_steps, seed=seed, history=history)
    steps = len(history)  # the run stops early once the worm reaches the food
    path = np.array(worm.path)  # steps + 1 points: the last one is where it ended up
    ate = end_d <= EAT_RADIUS
    sub_steps = len(history[0]["aiy_v"])
    act = _activation(history)
    names, neuron_pos = _load_neurons()

    # Pre-compute the concentration field once for the background.
    pad = 8
    x0, x1 = path[:, 0].min() - pad, path[:, 0].max() + pad
    y0, y1 = path[:, 1].min() - pad, path[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    field = env.peak * np.exp(-np.hypot(gx - env.food_pos[0], gy - env.food_pos[1]) / env.spread)

    fig = plt.figure(figsize=(11.5, 7.6), dpi=72, facecolor="white")
    grid = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.05], height_ratios=[6.2, 1, 1],
                            left=0.045, right=0.97, top=0.79, bottom=0.05, wspace=0.10, hspace=0.35)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_net = fig.add_subplot(grid[0, 1])
    ax_aiy = fig.add_subplot(grid[1, 1])
    ax_aiz = fig.add_subplot(grid[2, 1])

    fig.text(0.045, 0.95, "How a worm with 2 neurons finds food", fontsize=17,
             fontweight="bold", color=INK, va="center")
    badge = fig.text(0.045, 0.875, "", fontsize=12.5, fontweight="bold", va="center")

    # ---- Map -----------------------------------------------------------
    ax_map.imshow(field, extent=(x0, x1, y0, y1), origin="lower", cmap="Greens",
                  alpha=0.55, vmin=0, vmax=env.peak)
    ax_map.plot(*env.food_pos, marker="*", ms=20, color="#c9a227", mec=INK, zorder=5)
    ax_map.text(env.food_pos[0] + 2, env.food_pos[1] + 2, "food", fontsize=10, color=INK)
    ax_map.add_patch(plt.Circle(env.food_pos, EAT_RADIUS, fill=False, ec="#c9a227", lw=1.2, ls=":", zorder=5))
    ax_map.set_aspect("equal")
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y0, y1)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title("Odor concentration (darker = stronger)", fontsize=11, color=INK, loc="left")
    trail, = ax_map.plot([], [], color=INK, lw=1.6, zorder=3)
    turn_dots, = ax_map.plot([], [], "o", color=ORANGE, ms=7, mec="white", zorder=4)
    body, = ax_map.plot([], [], "o", color=INK, ms=9, mec="white", zorder=6)
    nose, = ax_map.plot([], [], color=INK, lw=3, solid_capstyle="round", zorder=6)
    distance = ax_map.text(0.02, 0.03, "", transform=ax_map.transAxes, fontsize=10.5,
                           color=INK, bbox=dict(fc="white", ec="none", alpha=0.8))

    # ---- All 302 neurons, on a dark panel ------------------------------
    halos, cores = _build_neuron_panel(fig, ax_net, names, neuron_pos)
    lit = [np.array(to_rgba(g[2])) for g in GROUPS]
    dim = np.array(to_rgba(DIM))

    # ---- Membrane potentials --------------------------------------------
    lines = {}
    for ax, key, label, color in ((ax_aiy, "aiy", "AIY  (fires when smell gets stronger)", GREEN),
                                  (ax_aiz, "aiz", "AIZ  (fires when smell gets weaker)", ORANGE)):
        ax.set_title(label, fontsize=10, color=color, loc="left", fontweight="bold", pad=3)
        ax.axhline(V_THRESHOLD, color=GREY, ls="--", lw=1)
        ax.text(0.995, 0.95, "threshold", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=GREY)
        ax.set_ylim(-73, -44)
        ax.set_yticks([])
        ax.set_ylabel("voltage", fontsize=9, color=GREY)
        ax.set_xticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lines[key] = (ax.plot([], [], color=color, lw=1.4)[0],
                      ax.plot([], [], color=color, lw=2.2)[0])  # trace, spike marks

    every = max(1, steps // TARGET_FRAMES)
    turn_steps = [k for k, h in enumerate(history) if h["turned"]]

    def draw(i):
        j = min(i, steps - 1)  # last step whose decision was recorded

        # Map: path so far, turn markers, current position and heading.
        trail.set_data(path[: i + 1, 0], path[: i + 1, 1])
        done_turns = [k for k in turn_steps if k <= i]
        turn_dots.set_data(path[done_turns, 0], path[done_turns, 1])
        x, y = path[i]
        body.set_data([x], [y])
        dx, dy = path[j + 1] - path[j]
        heading = np.arctan2(dy, dx)
        nose.set_data([x, x + 3.5 * np.cos(heading)], [y, y + 3.5 * np.sin(heading)])
        distance.set_text(f"distance to food: {np.hypot(x - env.food_pos[0], y - env.food_pos[1]):.0f}"
                          f"     turns: {len(done_turns)}")

        if i >= steps:
            text, color = ("Reached the food!  Stops and eats." if ate else "Out of time before reaching the food"), GREEN
        else:
            text, color = _badge(history, i, every)
        badge.set_text(text)
        badge.set_color(color)

        # Neurons: each group's brightness fades between dim and its lit colour.
        g = _glow(act, j)
        for k, (halo, core) in enumerate(zip(halos, cores)):
            n = len(core.get_offsets())
            core.set_facecolors([dim + (lit[k] - dim) * g[k]] * n)
            halo.set_facecolors([(*lit[k][:3], 0.4 * g[k])] * n)

        for key in ("aiy", "aiz"):
            t, v, fired = _membrane_trace(history, j, key, sub_steps)
            trace, spikes = lines[key]
            trace.set_data(t, v)
            spike_t = t[fired]
            # Each spike is drawn as a tall line so it is visible at this scale.
            spikes.set_data(np.repeat(spike_t, 3),
                            np.tile([V_THRESHOLD, -45, np.nan], len(spike_t)))
            (ax_aiy if key == "aiy" else ax_aiz).set_xlim(max(0, j - WINDOW_TICKS + 1), max(WINDOW_TICKS, j + 1))

    frames = list(range(0, steps, every)) + [steps] * (2 * FPS)  # hold on the final frame for 2 s
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)

    out = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=FPS))
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(frames)} frames)")
    print(f"Turns: {turns}   distance to food: {start_d:.1f} -> {end_d:.1f}")


if __name__ == "__main__":
    main()
