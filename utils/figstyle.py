"""figstyle.py - paper-scale figure styling.

Figures are authored at the width they will occupy in the compiled paper,
so a font size set in points renders at that same size on the page. The
manuscript body text is 9 pt and the text block is 5.95 inches wide, so
every label in a figure is set at 9 pt or larger and no figure is scaled
down by LaTeX.

Usage:
    from utils.figstyle import use_paper_style, TEXT_WIDTH_IN
    use_paper_style()
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.6))
    ...
    save_paper_figure(fig, path)
"""

TEXT_WIDTH_IN = 5.95     # \textwidth of the acmart manuscript layout
BODY_PT = 9.0            # body text size of the manuscript

# Colorblind-safe palette (Okabe-Ito) used across every figure.
PALETTE = {
    'blue': '#0072B2',
    'vermilion': '#D55E00',
    'green': '#009E73',
    'purple': '#CC79A7',
    'orange': '#E69F00',
    'sky': '#56B4E9',
    'gray': '#555555',
}


def use_paper_style(base=BODY_PT):
    """Set matplotlib defaults so figure text matches the paper body size."""
    import matplotlib
    matplotlib.rcParams.update({
        'font.size': base,
        'axes.titlesize': base + 1,
        'axes.labelsize': base,
        'xtick.labelsize': base,
        'ytick.labelsize': base,
        'legend.fontsize': base,
        'figure.titlesize': base + 1,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 4.5,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'savefig.dpi': 400,
    })


def save_paper_figure(fig, path):
    """Save at high resolution with minimal margins."""
    fig.savefig(path, dpi=400, bbox_inches='tight', facecolor='white')
