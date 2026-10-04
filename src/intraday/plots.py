"""Figure style and shared plotting helpers (static, publication output).

Palette: the validated reference palette of the dataviz method. Assets take
categorical slots in a FIXED order (never by rank), so EUR/USD is always blue,
S&P 500 always orange and BTC always aqua across every figure. Magnitude uses
the one-hue blue ramp; signed quantities use the blue<->red diverging pair with
a neutral grey midpoint.
"""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .config import FIGURES  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ASSET_COLOR = {"eurusd": SERIES[0], "spx": SERIES[1], "btc": SERIES[2]}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a12b2a"])


def setup() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "lines.linewidth": 1.5, "legend.frameon": False, "legend.fontsize": 8.5,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)


def note(ax, text: str) -> None:
    """Small source/sample note under an axis (muted ink)."""
    ax.annotate(text, (0, -0.16), xycoords="axes fraction", fontsize=7.5, color=MUTED, va="top")
