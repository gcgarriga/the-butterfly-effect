"""Strange attractor — the butterfly effect, literally.

Two initial conditions a hair apart are integrated (RK4) through a chaotic ODE; they
trace the same shape, then diverge. Drawn as glowing ribbons on black. Procedural
(no model needed) — the project's namesake from chaos theory.

Examples:
    python generate.py --size wallpaper_4k --attractor lorenz --scheme ice_fire
    python generate.py --size 2048x2048 --attractor aizawa --scheme aurora
"""

import os
import sys

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap as LSC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from common import presets  # noqa: E402

BG = "#04060d"


def lorenz(s):
    x, y, z = s
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z])


def aizawa(s):
    x, y, z = s
    a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
    return np.array([(z - b) * x - d * y, d * x + (z - b) * y,
                     c + a * z - z**3 / 3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3])


def thomas(s):
    x, y, z = s
    b = 0.208
    return np.array([np.sin(y) - b * x, np.sin(z) - b * y, np.sin(x) - b * z])


def halvorsen(s):
    x, y, z = s
    a = 1.4
    return np.array([-a * x - 4 * y - 4 * z - y**2, -a * y - 4 * z - 4 * x - z**2,
                     -a * z - 4 * x - 4 * y - x**2])


ATTRACTORS = {  # deriv, dt, steps, initial condition, projection axes
    "lorenz": (lorenz, 0.005, 16000, (0.0, 1.0, 1.05), (0, 2)),
    "aizawa": (aizawa, 0.01, 22000, (0.1, 0.0, 0.0), (0, 2)),
    "thomas": (thomas, 0.02, 22000, (0.1, 0.0, 0.0), (0, 1)),
    "halvorsen": (halvorsen, 0.004, 18000, (-5.0, 0.0, 0.0), (0, 1)),
}

C = {
    "cyan": ["#0b2545", "#3aa0ff", "#9be7ff"], "magenta": ["#3a0826", "#ff2d95", "#ff8fce"],
    "teal": ["#02110d", "#0fbf8f", "#aeffe6"], "violet": ["#1a0938", "#7b2ff7", "#d6b8ff"],
    "gold": ["#2a1c02", "#ffae00", "#ffe9a8"], "ember": ["#2a0606", "#ff5a36", "#ffd0a8"],
    "ice": ["#06122a", "#1b8fd0", "#cdf3ff"], "pink": ["#2a0820", "#ff5ca8", "#ffd6f2"],
}
SCHEMES = {
    "ice_fire": ("cyan", "magenta"), "aurora": ("teal", "violet"),
    "gold_ice": ("gold", "ice"), "ember_ice": ("ember", "cyan"), "pink_teal": ("pink", "teal"),
}


def integrate(deriv, dt, steps, ic):
    pts = np.empty((steps, 3))
    s = np.array(ic)
    for i in range(steps):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        pts[i] = s
    return pts


def glow(ax, x, y, cmap, lw):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    base = cmap(np.linspace(0.15, 1.0, len(segs)))
    for width, a in [(6, 0.035), (3, 0.09), (1.3, 0.38), (0.6, 0.62)]:
        col = base.copy()
        col[:, 3] = a
        ax.add_collection(LineCollection(segs, colors=col, linewidths=width * lw, capstyle="round"))


def main():
    p = presets.base_parser(__doc__)
    p.add_argument("--attractor", default="lorenz", choices=list(ATTRACTORS))
    p.add_argument("--scheme", default="ice_fire", choices=list(SCHEMES))
    args = p.parse_args()

    w, h = presets.resolve(args.size)
    lw = (w**2 + h**2) ** 0.5 / 1632
    deriv, dt, steps, ic, (i0, i1) = ATTRACTORS[args.attractor]
    cmaps = [LSC.from_list(n, C[n]) for n in SCHEMES[args.scheme]]

    fig, ax = presets.new_fig(w, h, BG, args.dpi)
    allpts = []
    for state, cm in zip((ic, (ic[0] + 1e-3, ic[1], ic[2])), cmaps):
        pp = integrate(deriv, dt, steps, state)[:, [i0, i1]]
        glow(ax, pp[:, 0], pp[:, 1], cm, lw)
        allpts.append(pp)
    P = np.vstack(allpts)
    # centre + fit, preserving the attractor's true proportions (dark margins fill the rest)
    x0, x1 = np.percentile(P[:, 0], [0.3, 99.7])
    y0, y1 = np.percentile(P[:, 1], [0.3, 99.7])
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.56
    ax.set_aspect("equal")
    ax.set_xlim(cx - half * presets.aspect(w, h), cx + half * presets.aspect(w, h))
    ax.set_ylim(cy - half, cy + half)
    ax.axis("off")

    out = args.out or f"{args.attractor}_{args.scheme}_{w}x{h}.png"
    fig.savefig(out, dpi=args.dpi, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
