"""
Render the FULL connectome simulation as an animated GIF:
output_results/full_connectome.gif

Left        : the arena, as in the simple version's GIF (make_gif.py).
Right, top  : all 302 neurons at their real positions in the body (upper strip:
              the whole worm; lower box: zoom on the head). Every neuron lights
              up when it fires and fades out afterwards, coloured by type:
              blue = sensory, green = interneuron, orange = motor.
Right, below: how many neurons fired on each step (network activity), with a
              tick for every turn. Note how activity builds up and lingers.

This is the honest picture of what running the raw connectome does: the odor
does set off the network, but the activity reverberates instead of following
the odor - see full_chemotaxis.py for why the worm ends up navigating poorly.

    pip install matplotlib pillow
    python src/make_full_gif.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")  # draw to memory, no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba

from chemotaxis_circuit import EAT_RADIUS, run_simulation
from full_chemotaxis import FullConnectomeCircuit
from make_gif import DIM, GREEN, GREY, HEAD_Y_RANGE, INK, ORANGE, PANEL, POSITIONS_PATH

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output_results", "full_connectome.gif")
TARGET_FRAMES = 120
FADE_TICKS = 8
FPS = 12
TYPE_COLOUR = {"sensory": "#7fd4ff", "interneuron": "#5dffa8", "motor": "#ff9d4d", "unknown": "#e8e8e8"}
FOOTER = "Reaches the food in 4 of 20 runs (the two-neuron version: 16 of 20)"


def _load_neurons():
    with open(POSITIONS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    names = [r["name"] for r in rows]
    pos = np.array([[float(r["y_um"]), float(r["z_um"])] for r in rows])
    lit = np.array([to_rgba(TYPE_COLOUR[r["type"]]) for r in rows])
    return names, pos, lit


def _glow(act, i):
    """Brightness of every neuron at step i: each spike fades out over FADE_TICKS."""
    first = max(0, i - 4 * FADE_TICKS)
    age = i - np.arange(first, i + 1)
    return (act[first: i + 1] * np.exp(-age / FADE_TICKS)[:, None]).max(axis=0)


def _build_neuron_panel(fig, ax, pos):
    """Draw the dark panel. Returns (strip_dots, zoom_dots, zoom_halos, in_head)."""
    w = ax.get_position().width * fig.get_figwidth() * fig.dpi
    h = ax.get_position().height * fig.get_figheight() * fig.dpi
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # pixel coordinates, origin at the top-left
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    y_all = pos[:, 0]
    strip_scale = (w - 30) / (y_all.max() - y_all.min())
    strip_cy = 58
    z_mid = (pos[:, 1].max() + pos[:, 1].min()) / 2

    in_head = (y_all >= HEAD_Y_RANGE[0]) & (y_all <= HEAD_Y_RANGE[1])
    head_dz = np.ptp(pos[in_head, 1])
    zbox_top, zbox_bot = strip_cy + 32 + 16, h - 22
    zoom_cy = (zbox_top + zbox_bot) / 2
    zoom_scale = min((w - 30) / (HEAD_Y_RANGE[1] - HEAD_Y_RANGE[0]), (zbox_bot - zbox_top - 16) / head_dz)
    zoom_z_mid = (pos[in_head, 1].max() + pos[in_head, 1].min()) / 2
    zoom_x0 = (w - zoom_scale * (HEAD_Y_RANGE[1] - HEAD_Y_RANGE[0])) / 2

    def strip_xy(p):
        return 15 + (p[:, 0] - y_all.min()) * strip_scale, strip_cy - (p[:, 1] - z_mid) * strip_scale

    def zoom_xy(p):
        return zoom_x0 + (p[:, 0] - HEAD_Y_RANGE[0]) * zoom_scale, zoom_cy - (p[:, 1] - zoom_z_mid) * zoom_scale

    ax.text(10, 13, "All 302 neurons firing, at their real positions", color="white", fontsize=10,
            fontweight="bold", va="center")
    ax.text(w - 10, h - 10, "head, zoomed", color="#6f8279", fontsize=8, ha="right", va="center")

    bx0, bx1 = strip_xy(np.array([[HEAD_Y_RANGE[0], 0.0], [HEAD_Y_RANGE[1], 0.0]]))[0]
    zx0, zx1 = zoom_x0, zoom_x0 + zoom_scale * (HEAD_Y_RANGE[1] - HEAD_Y_RANGE[0])
    line = dict(color="#31443b", lw=1, zorder=1)
    ax.plot([bx0, bx1, bx1, bx0, bx0], [strip_cy - 32, strip_cy - 32, strip_cy + 32, strip_cy + 32, strip_cy - 32], **line)
    ax.plot([zx0, zx1, zx1, zx0, zx0], [zbox_top, zbox_top, zbox_bot, zbox_bot, zbox_top], **line)
    ax.plot([bx0, zx0], [strip_cy + 32, zbox_top], **line)
    ax.plot([bx1, zx1], [strip_cy + 32, zbox_top], **line)

    dim = to_rgba(DIM)
    sx, sy = strip_xy(pos)
    strip_dots = ax.scatter(sx, sy, s=9, c=[dim] * len(pos), linewidths=0, zorder=3)
    zx, zy = zoom_xy(pos[in_head])
    zoom_halos = ax.scatter(zx, zy, s=16 ** 2, c=[(*dim[:3], 0.0)] * in_head.sum(), linewidths=0, zorder=2)
    zoom_dots = ax.scatter(zx, zy, s=24, c=[dim] * in_head.sum(), linewidths=0, zorder=3)

    # Colour key.
    for k, (kind, label) in enumerate((("sensory", "sensory"), ("interneuron", "inter"), ("motor", "motor"))):
        x = 10 + k * 62
        ax.scatter([x], [h - 10], s=30, c=[TYPE_COLOUR[kind]], linewidths=0, zorder=3)
        ax.text(x + 7, h - 10, label, color="#8fa298", fontsize=8, va="center")
    return strip_dots, zoom_dots, zoom_halos, in_head


def main(max_steps=1500, seed=7):
    history = []
    worm, env, turns, start_d, end_d = run_simulation(
        steps=max_steps, seed=seed, history=history, circuit=FullConnectomeCircuit()
    )
    steps = len(history)
    path = np.array(worm.path)
    ate = end_d <= EAT_RADIUS
    names, neuron_pos, lit = _load_neurons()

    spikes = np.array([h["spikes"] for h in history])  # (steps, 302): spikes per neuron per step
    act = np.minimum(1.0, spikes / 2.0)                # 2 spikes in one step = fully lit
    firing_per_step = (spikes > 0).sum(axis=1)         # how many neurons fired on each step
    turn_steps = np.array([k for k, h in enumerate(history) if h["turned"]])

    pad = 8
    x0, x1 = path[:, 0].min() - pad, path[:, 0].max() + pad
    y0, y1 = path[:, 1].min() - pad, path[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    field = env.peak * np.exp(-np.hypot(gx - env.food_pos[0], gy - env.food_pos[1]) / env.spread)

    fig = plt.figure(figsize=(11.5, 7.6), dpi=72, facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[5.2, 1.3],
                            left=0.045, right=0.97, top=0.79, bottom=0.07, wspace=0.10, hspace=0.30)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_net = fig.add_subplot(grid[0, 1])
    ax_act = fig.add_subplot(grid[1, 1])

    fig.text(0.045, 0.95, "Full version: the whole worm brain, real wiring", fontsize=17,
             fontweight="bold", color=INK, va="center")
    badge = fig.text(0.045, 0.875, "", fontsize=12.5, fontweight="bold", va="center")
    fig.text(0.045, 0.018, FOOTER, fontsize=10, color=GREY, va="center")

    # ---- Map -----------------------------------------------------------
    ax_map.imshow(field, extent=(x0, x1, y0, y1), origin="lower", cmap="Greens", alpha=0.55, vmin=0, vmax=env.peak)
    ax_map.plot(*env.food_pos, marker="*", ms=20, color="#c9a227", mec=INK, zorder=5)
    ax_map.text(env.food_pos[0] + 2, env.food_pos[1] + 2, "food", fontsize=10, color=INK)
    ax_map.add_patch(plt.Circle(env.food_pos, EAT_RADIUS, fill=False, ec="#c9a227", lw=1.2, ls=":", zorder=5))
    ax_map.set_aspect("equal")
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y0, y1)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_title("Odor concentration (darker = stronger)", fontsize=11, color=INK, loc="left")
    trail, = ax_map.plot([], [], color=INK, lw=1.4, zorder=3)
    turn_dots, = ax_map.plot([], [], "o", color=ORANGE, ms=5, mec="white", mew=0.5, zorder=4)
    body, = ax_map.plot([], [], "o", color=INK, ms=9, mec="white", zorder=6)
    nose, = ax_map.plot([], [], color=INK, lw=3, solid_capstyle="round", zorder=6)
    distance = ax_map.text(0.02, 0.03, "", transform=ax_map.transAxes, fontsize=10.5,
                           color=INK, bbox=dict(fc="white", ec="none", alpha=0.8))

    # ---- Neurons --------------------------------------------------------
    strip_dots, zoom_dots, zoom_halos, in_head = _build_neuron_panel(fig, ax_net, neuron_pos)
    dim = np.array(to_rgba(DIM))

    # ---- Network activity over time --------------------------------------
    ax_act.set_title("Neurons firing per step", fontsize=10, color=INK, loc="left", pad=3)
    ax_act.set_xlim(0, steps)
    ax_act.set_ylim(-0.12 * firing_per_step.max(), firing_per_step.max() * 1.08)
    ax_act.set_yticks([])
    ax_act.set_xticks([])
    for side in ("top", "right"):
        ax_act.spines[side].set_visible(False)
    act_line, = ax_act.plot([], [], color=GREEN, lw=1.3)
    act_turns, = ax_act.plot([], [], "|", color=ORANGE, ms=8, mew=1.5)
    now = ax_act.axvline(0, color=INK, lw=1, alpha=0.4)
    ax_act.text(0.995, 0.95, "orange ticks = turns", transform=ax_act.transAxes, ha="right", va="top",
                fontsize=8, color=ORANGE)

    every = max(1, steps // TARGET_FRAMES)

    def draw(i):
        j = min(i, steps - 1)

        trail.set_data(path[: i + 1, 0], path[: i + 1, 1])
        done_turns = turn_steps[turn_steps <= i]
        turn_dots.set_data(path[done_turns, 0], path[done_turns, 1])
        x, y = path[i]
        body.set_data([x], [y])
        dx, dy = path[j + 1] - path[j]
        heading = np.arctan2(dy, dx)
        nose.set_data([x, x + 3.5 * np.cos(heading)], [y, y + 3.5 * np.sin(heading)])
        distance.set_text(f"distance to food: {np.hypot(x - env.food_pos[0], y - env.food_pos[1]):.0f}"
                          f"     turns: {len(done_turns)}")

        recent = history[max(0, j - every + 1): j + 1]
        if i >= steps:
            text, color = (("Reached the food!  Stops and eats." if ate else
                            f"Time is up: still {end_d:.0f} units from the food"), GREEN if ate else GREY)
        elif any(h["turned"] for h in recent):
            text, color = "Backward command neurons out-vote forward  ->  TURN", ORANGE
        elif firing_per_step[j] >= 15:
            text, color = f"Network active: {firing_per_step[j]} of 302 neurons fired this step", GREEN
        else:
            text, color = "Network quiet  ->  keep going straight", GREY
        badge.set_text(text)
        badge.set_color(color)

        g = _glow(act, j)[:, None]
        colours = dim + (lit - dim) * g
        strip_dots.set_facecolors(colours)
        zoom_dots.set_facecolors(colours[in_head])
        zoom_halos.set_facecolors(np.column_stack([lit[in_head, :3], 0.35 * g[in_head, 0]]))

        act_line.set_data(np.arange(j + 1), firing_per_step[: j + 1])
        act_turns.set_data(done_turns, np.full(len(done_turns), -0.06 * firing_per_step.max()))
        now.set_xdata([j, j])

    frames = list(range(0, steps, every)) + [steps] * (2 * FPS)
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / FPS)

    out = os.path.abspath(OUT_PATH)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=FPS))
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(frames)} frames)")
    print(f"Turns: {turns}   distance to food: {start_d:.1f} -> {end_d:.1f}   steps: {steps}")


if __name__ == "__main__":
    main()
