"""
Combined slopegraph: Dice (NoTTA → TTA) across testing versions (Dataset 1/2/3 blocks).
- Not bold fonts
- Thick lines
- Legend included
- 500 DPI

Dependencies:
  pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

test_sets = ["Kvasir-SEG", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-Larib", "CVC-T"]
versions = ["Dataset 1", "Dataset 2", "Dataset 3"]

tta_dice = {
    "Dataset 1": [0.9168, 0.9406, 0.8066, 0.8114, 0.8991],
    "Dataset 2": [0.9190, 0.9470, 0.8277, 0.8076, 0.9199],
    "Dataset 3": [0.9082, 0.9527, 0.8204, 0.7951, 0.8924],
}
notta_dice = {
    "Dataset 1": [0.9062, 0.9305, 0.7935, 0.7807, 0.8987],
    "Dataset 2": [0.9194, 0.9454, 0.8227, 0.8061, 0.9109],
    "Dataset 3": [0.9071, 0.9525, 0.8236, 0.7933, 0.8862],
}

plt.rcParams.update({
    "font.size": 16,
    "font.weight": "normal",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

def main():
    out_path = Path("Slopegraph_Dice_AllTestingVersions.png")

    x_blocks = {"Dataset 1": (0, 1), "Dataset 2": (3, 4), "Dataset 3": (6, 7)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(test_sets):
        x, y = [], []
        for v in versions:
            x0, x1 = x_blocks[v]
            x.extend([x0, x1, np.nan])
            y.extend([notta_dice[v][i], tta_dice[v][i], np.nan])
        ax.plot(x, y, marker="o", linewidth=3.0, markersize=7, markeredgewidth=2.0, label=name)

    ax.set_xticks([0, 1, 3, 4, 6, 7], ["NoTTA", "TTA", "NoTTA", "TTA", "NoTTA", "TTA"])
    ax.set_xlabel("Testing version blocks: Dataset 1 | Dataset 2 | Dataset 3")
    ax.set_ylabel("Dice")
    ax.set_title("Slopegraph: Dice (NoTTA → TTA) across testing versions")
    ax.grid(True, axis="y", linewidth=1.5, alpha=0.35)

    for v in [2, 5]:
        ax.axvline(v, linewidth=2.0, alpha=0.5)

    ax.legend(loc="best", frameon=True, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
