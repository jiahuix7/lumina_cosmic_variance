"""Minimal publication figure style (role-mapped 3-size ladder, clean frame)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE, SMALL, TINY = 9, 8, 7
META_GREY = "#6b6b6b"


def apply_figure_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": BASE,
        "axes.titlesize": BASE, "axes.labelsize": BASE,
        "axes.titleweight": "regular", "axes.titlelocation": "left",
        "axes.linewidth": 0.8, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.fontsize": SMALL, "legend.frameon": False,
        "xtick.labelsize": TINY, "ytick.labelsize": TINY,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
    })


def panel_letter(ax, letter, case="lower"):
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(-0.02, 1.06, s, transform=ax.transAxes,
            fontsize=BASE + 2, fontweight="bold", va="bottom", ha="right")


def panel_crops(fig, pad=6):
    r = fig.canvas.get_renderer()
    H = fig.bbox.height
    out = {}
    for i, ax in enumerate(fig.axes):
        b = ax.get_tightbbox(r)
        out[chr(97 + i)] = (max(b.x0 - pad, 0), max(H - b.y1 - pad, 0),
                            b.width + 2 * pad, b.height + 2 * pad)
    return out


def check_overlaps(fig):
    """Geometric check per figure-style 9.1. Returns list of offending pairs."""
    r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip() and t.get_visible()]
    spines = [(s, s.get_window_extent(r)) for ax in fig.axes
              for s in ax.spines.values() if s.get_visible()]
    tl = {ax: set(ax.get_xticklabels(which="both") +
                  ax.get_yticklabels(which="both")) for ax in fig.axes}
    ov = [(a.get_text(), b.get_text())
          for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
          if ba.overlaps(bb)]
    ov += [(t.get_text(), "spine")
           for t, bt in texts for s, bs in spines
           if bt.overlaps(bs) and t not in tl.get(s.axes, set())]
    outside = [t.get_text() for t, bt in texts
               if not fig.bbox.contains(bt.x0, bt.y0)
               or not fig.bbox.contains(bt.x1, bt.y1)]
    return ov, outside
