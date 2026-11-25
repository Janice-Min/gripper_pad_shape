"""Exact open-keyframe XY placement preview of the capsule links and pad section."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from padopt.common import read_config
from padopt.evaluation import compile_case
from padopt.geometry import grid_vertices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = read_config(args.config)
    import mujoco as mj
    model, data = compile_case(cfg, args.candidate, cfg["evaluation"]["train"][2])
    mj.mj_resetDataKeyframe(model, data, mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "open"))
    mj.mj_forward(model, data)
    root = ET.parse(cfg["base_xml"]).getroot()
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    def world(body, points):
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body)
        return (np.asarray(points) @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid])[:, :2] * 1000
    for body, geom, color in [("upper_link1", "upper_green_link", "#929aa5"), ("lower_link1", "lower_green_link", "#929aa5"),
                              ("upper_link2", "upper_red_link", "#df8d86"), ("lower_link2", "lower_red_link", "#df8d86")]:
        element = root.find(".//geom[@name='" + geom + "']")
        endpoints = world(body, np.fromstring(element.get("fromto"), sep=" ").reshape(2, 3))
        r = float(element.get("size")) * 1000
        a, b = endpoints
        angle = np.arctan2(*(b - a)[::-1])
        angles0 = angle + np.linspace(np.pi / 2, 3 * np.pi / 2, 40)
        angles1 = angle + np.linspace(-np.pi / 2, np.pi / 2, 40)
        polygon = np.concatenate([a + r * np.c_[np.cos(angles0), np.sin(angles0)], b + r * np.c_[np.cos(angles1), np.sin(angles1)]])
        ax.add_patch(Polygon(polygon, facecolor=color, edgecolor="#374151", linewidth=1.1))
        ax.plot(endpoints[:, 0], endpoints[:, 1], "o", color="#374151", markersize=4)
    with np.load(Path(args.candidate) / "surface.npz", allow_pickle=False) as g:
        for body, sign in [("upper_link2", -1), ("lower_link2", 1)]:
            bottom, top = grid_vertices(g["s"], g["z"], float(g["radius"]), g["thickness"], cfg["pad"]["embed_mm"] * .001, sign)
            middle = len(g["z"]) // 2
            contour = np.concatenate([bottom[:, middle], top[::-1, middle]])
            ax.add_patch(Polygon(world(body, contour), facecolor="#16a394", edgecolor="#006b63", linewidth=1.2))
    x, y = cfg["evaluation"]["position_x_m"] * 1000, 0
    ax.add_patch(Circle((x, y), 30, facecolor="#bdd3ef", edgecolor="#4e729c", alpha=.7))
    ax.text(x, y, "Example object\nr = 30 mm", ha="center", va="center", color="#28496b", fontsize=10)
    ax.text(10, 0, "Palm", ha="center", va="center", color="#4b5563")
    ax.annotate("Upper distal pad\nlocal -Y inward", xy=(-73, 48), xytext=(-112, 81), arrowprops={"arrowstyle": "->", "color": "#006b63"}, color="#006b63", fontsize=10)
    ax.annotate("Lower distal pad\nlocal +Y inward", xy=(-73, -48), xytext=(-112, -83), arrowprops={"arrowstyle": "->", "color": "#006b63"}, color="#006b63", fontsize=10)
    ax.set(xlim=(-120, 35), ylim=(-90, 90), xlabel="World X [mm]", ylabel="World Y [mm]",
           title="Flat control pad: placement preview\nOpen keyframe, viewed from +Z (not optimized)")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
