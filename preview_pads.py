import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from padopt.common import candidate_dirs


def main():
    p = argparse.ArgumentParser(description="Plot pad thickness maps; no Shap-E or MuJoCo dependency.")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    candidates = candidate_dirs(args.round_dir)[:args.count]
    if not candidates:
        raise ValueError("No candidates found")
    cols = min(4, len(candidates))
    rows = (len(candidates) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows), squeeze=False, constrained_layout=True)
    for ax, candidate in zip(axes.ravel(), candidates):
        with np.load(candidate / "surface.npz", allow_pickle=False) as g:
            image = ax.pcolormesh(g["s"] * 1000, g["z"] * 1000, g["thickness"].T * 1000, shading="auto", cmap="viridis", vmin=1.5, vmax=5.5)
        source = json.loads((candidate / "metadata.json").read_text())["source"]
        ax.set_title(candidate.name + " (" + source + ")", fontsize=9)
        ax.set_xlabel("Distance from distal joint [mm]")
        ax.set_ylabel("Width [mm]")
        ax.set_aspect("equal")
    for ax in axes.ravel()[len(candidates):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=list(axes.ravel()), label="Pad thickness [mm]", shrink=.7)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
