"""Shared matplotlib style + colors + save helper for poster figures."""

from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

POSTER_DIR = Path(__file__).resolve().parents[3] / "results"
POSTER_DIR.mkdir(exist_ok=True)

ALGO_COLORS = {
    "BC":     "#7f7f7f",
    "SAC":    "#1f77b4",
    "PPO":    "#d62728",
    "TD3+BC": "#2ca02c",
}

BC_SEED_FULL110 = 0.6238532110091743


def apply_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save_fig(fig, name):
    pdf = POSTER_DIR / f"{name}.pdf"
    png = POSTER_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f"  -> {pdf}")
    print(f"  -> {png}")
