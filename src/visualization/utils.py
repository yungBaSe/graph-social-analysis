import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import hashlib
import json
from typing import Any
from IPython.display import display

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })


def _get_figure_path(name: str, plot_type: str, **kwargs) -> Path:
    param_str = json.dumps(kwargs, sort_keys=True)
    key = hashlib.md5(param_str.encode()).hexdigest()[:8]
    return FIGURES_DIR / f"{name}_{plot_type}_{key}.png"


def show_and_save(fig, name, plot_type, show=True, save=True, **kwargs):
    if save:
        path = _get_figure_path(name, plot_type, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"✅ Saved → {path.name}")

    if show:
        display(fig)
    
    plt.close(fig)
    plt.clf()
    import gc
    gc.collect() 


def list_figures(limit: int = 20) -> None:
    files = sorted(FIGURES_DIR.glob("*.png"), reverse=True)
    print(f"Found {len(files)} figures (showing last {limit}):")
    for f in files[:limit]:
        print(f"  • {f.name}")