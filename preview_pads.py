"""Plot pad shapes in one PNG; no Shap-E or MuJoCo dependency.

--round-dir: the first --count completed candidates of a round (by candidate number).
--elites:    the candidates of an elites.json in rank order, titled with rank and score.
--view solid (default): the real 3D pad solid (as in the STL) sitting on the distal link.
--view surface:        3D surface plot of the thickness over the pad footprint.
--view map:            thickness maps (top view of the contact surface).
"""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from padopt.common import candidate_dirs, read_config, parse_gripper
from padopt.geometry import grid_vertices


def draw_solid(ax, s, z, radius, thickness, embed, link_length, vmin, vmax):
    """Pad solid on the distal link. Display axes: along the link from the joint, width (+ = up), height toward the object."""
    bottom, top = grid_vertices(s, z, radius, thickness, embed, 1)
    to_mm = lambda v: np.stack([-v[..., 0], v[..., 2], v[..., 1]], axis=-1) * 1000
    bottom, top = to_mm(bottom), to_mm(top)
    ni, nj = thickness.shape
    cmap = plt.get_cmap("viridis")
    light = np.array([-0.35, -0.45, 1.0])
    light /= np.linalg.norm(light)
    quads, colors = [], []

    def add(quad, rgb):
        normal = np.cross(quad[2] - quad[0], quad[3] - quad[1])
        normal /= max(np.linalg.norm(normal), 1e-12)
        shade = 0.5 + 0.5 * abs(float(normal @ light))
        quads.append(quad)
        colors.append((*(np.asarray(rgb) * shade), 1.0))

    t_mm = thickness * 1000
    for i in range(ni - 1):  # contact surface, coloured by thickness
        for j in range(nj - 1):
            quad = np.array([top[i, j], top[i + 1, j], top[i + 1, j + 1], top[i, j + 1]])
            value = (t_mm[i, j] + t_mm[i + 1, j] + t_mm[i + 1, j + 1] + t_mm[i, j + 1]) / 4
            add(quad, cmap((value - vmin) / max(vmax - vmin, 1e-9))[:3])
    ring = [(i, 0) for i in range(ni)] + [(ni - 1, j) for j in range(1, nj)] + \
           [(i, nj - 1) for i in range(ni - 2, -1, -1)] + [(0, j) for j in range(nj - 2, 0, -1)]
    for (i0, j0), (i1, j1) in zip(ring, ring[1:] + ring[:1]):  # side walls
        add(np.array([bottom[i0, j0], bottom[i1, j1], top[i1, j1], top[i0, j0]]), (0.66, 0.69, 0.73))
    ax.add_collection3d(Poly3DCollection(quads, facecolors=colors, edgecolors=(0, 0, 0, 0.12), linewidths=0.3))
    # Distal link (capsule body) as a light outline for context.
    L, r = link_length * 1000, radius * 1000
    u = np.linspace(0, np.pi, 40)
    for x in (0.0, L):
        ax.plot(np.full_like(u, x), r * np.cos(u), r * np.sin(u), color=(0.9, 0.3, 0.25), linewidth=0.8, alpha=0.6)
    for angle in (0.0, np.pi / 2, np.pi):
        ax.plot([0, L], [r * np.cos(angle)] * 2, [r * np.sin(angle)] * 2, color=(0.9, 0.3, 0.25), linewidth=0.6, alpha=0.5)
    ax.set_xlim(0, L)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, r + vmax)
    ax.set_box_aspect((L, 2 * r, r + vmax))
    ax.view_init(elev=38, azim=-62)
    ax.set_xlabel("from joint [mm]", fontsize=7, labelpad=-2)
    ax.set_ylabel("width [mm]", fontsize=7, labelpad=-4)
    ax.set_zlabel("height [mm]", fontsize=7, labelpad=-6)
    ax.tick_params(labelsize=6, pad=0)


def load_candidates(args):
    """List of (folder, title)."""
    if args.elites:
        selection = json.loads(Path(args.elites).read_text(encoding="utf-8"))
        items = []
        for k, elite in enumerate(selection["elites"][:args.count], 1):
            metrics = elite.get("metrics", {})
            title = "#" + str(elite.get("rank", k)) + "  " + elite["candidate_id"]
            if "mean_score" in metrics:
                title += "\nscore " + format(float(metrics["mean_score"]), ".2f") + " / " + format(float(metrics["max_score"]), "g")
                title += "   worst obj " + format(float(metrics["worst_object_score"]), ".1f")
            items.append((Path(elite["candidate_dir"]), title))
        return items
    items = []
    for folder in candidate_dirs(args.round_dir)[:args.count]:
        source = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))["source"]
        items.append((folder, folder.name + " (" + source + ")"))
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.json", help="Used for the thickness colour range")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--round-dir")
    group.add_argument("--elites", help="elites.json from select_elites.py (plotted in rank order)")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--view", choices=["solid", "surface", "map"], default="solid",
                   help="solid: 3D pad solid on the link (default); surface: 3D thickness plot; map: thickness map")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = read_config(args.config)
    pad = cfg["pad"]
    link_length = parse_gripper(cfg["base_xml"])["upper"]["length_m"]
    vmin, vmax = pad["thickness_min_mm"], pad["thickness_max_mm"]
    items = load_candidates(args)
    if not items:
        raise ValueError("No candidates found")
    missing = [str(folder) for folder, _ in items if not (folder / "surface.npz").is_file()]
    if missing:
        raise FileNotFoundError("surface.npz not found in: " + ", ".join(missing))
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    three_d = args.view in ("solid", "surface")
    fig = plt.figure(figsize=(4.2 * cols, (3.6 if three_d else 3.3) * rows), constrained_layout=True)
    image = None
    for k, (folder, title) in enumerate(items):
        with np.load(folder / "surface.npz", allow_pickle=False) as g:
            s_m, z_m, radius, thickness_m = g["s"], g["z"], float(g["radius"]), g["thickness"]
        s, z, thickness = s_m * 1000, z_m * 1000, thickness_m * 1000
        if args.view == "solid":
            ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
            draw_solid(ax, s_m, z_m, radius, thickness_m, pad["embed_mm"] * .001, link_length, vmin, vmax)
        elif three_d:
            ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
            ss, zz = np.meshgrid(s, z, indexing="ij")
            image = ax.plot_surface(ss, zz, thickness, cmap="viridis", vmin=vmin, vmax=vmax, edgecolor="k", linewidth=0.2)
            ax.set_zlim(0, vmax)
            ax.set_box_aspect((np.ptp(s), np.ptp(z), 2.5 * vmax))
            ax.view_init(elev=32, azim=-60)
            ax.set_xlabel("from joint [mm]", fontsize=7, labelpad=-2)
            ax.set_ylabel("width [mm]", fontsize=7, labelpad=-2)
            ax.set_zlabel("thick. [mm]", fontsize=7, labelpad=-4)
            ax.tick_params(labelsize=6, pad=0)
        else:
            ax = fig.add_subplot(rows, cols, k + 1)
            image = ax.pcolormesh(s, z, thickness.T, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Distance from distal joint [mm]", fontsize=8)
            ax.set_ylabel("Width [mm] (+ = up)", fontsize=8)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=9)
    if image is None:
        image = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax))
    fig.colorbar(image, ax=fig.axes, label="Pad thickness [mm]", shrink=.6)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print("Saved " + str(len(items)) + " pads -> " + str(args.out))


if __name__ == "__main__":
    main()
