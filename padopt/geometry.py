"""Project Shap-E geometry into a fixed-footprint, capsule-backed solid pad.

Coordinates of the internal grid: s along distal -X, z across its width,
n toward the opposing finger. All stored/exported coordinates use metres.
"""
import struct
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import ConvexHull
from .common import parse_gripper


def pad_axes(cfg):
    link = parse_gripper(cfg["base_xml"])["upper"]
    p = cfg["pad"]
    start = p["joint_margin_mm"] * .001
    end = link["length_m"] - p["tip_margin_mm"] * .001
    half_width = p["width_mm"] * .0005
    if not (0 < start < end < link["length_m"] and 0 < half_width < link["radius_m"]):
        raise ValueError("Invalid pad footprint: check margins, width, and capsule radius.")
    if not (0 < p["embed_mm"] < p["thickness_min_mm"] <= p["thickness_max_mm"]):
        raise ValueError("Require 0 < embed < min thickness <= max thickness.")
    if int(p["cells_length"]) < 2 or int(p["cells_width"]) < 2:
        raise ValueError("At least two collision cells along each axis are required.")
    s = np.linspace(start, end, int(p["cells_length"]) + 1)
    z = np.linspace(-half_width, half_width, int(p["cells_width"]) + 1)
    return s, z, link["radius_m"]


def projected_height(vertices, faces, shape):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if len(vertices) < 4 or len(faces) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Invalid decoded mesh")
    used = np.unique(faces.ravel())
    vertices = vertices - vertices[used].mean(axis=0)
    _, _, axes = np.linalg.svd(vertices[used], full_matrices=False)
    for row in axes:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1
    v = vertices @ axes.T
    span = np.ptp(v[used, :2], axis=0)
    if np.min(span) <= 1e-8:
        raise ValueError("Degenerate mesh projection")
    lo, hi = v[used, :2].min(axis=0), v[used, :2].max(axis=0)
    # A modest crop avoids basing the whole surface on silhouette-edge holes.
    u = np.linspace(lo[0] + .05 * span[0], hi[0] - .05 * span[0], shape[0])
    w = np.linspace(lo[1] + .05 * span[1], hi[1] - .05 * span[1], shape[1])
    heights = np.full(shape, -np.inf)
    for face in faces:
        a, b, c = v[face]
        imin, imax = np.searchsorted(u, [min(a[0], b[0], c[0]), max(a[0], b[0], c[0])], side="left")
        jmin, jmax = np.searchsorted(w, [min(a[1], b[1], c[1]), max(a[1], b[1], c[1])], side="left")
        imin, imax = max(0, imin), min(len(u), imax + 1)
        jmin, jmax = max(0, jmin), min(len(w), jmax + 1)
        if imin >= imax or jmin >= jmax:
            continue
        xx, yy = np.meshgrid(u[imin:imax], w[jmin:jmax], indexing="ij")
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-14:
            continue
        aa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
        bb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        cc = 1 - aa - bb
        zz = aa * a[2] + bb * b[2] + cc * c[2]
        mask = (aa >= -1e-7) & (bb >= -1e-7) & (cc >= -1e-7)
        block = heights[imin:imax, jmin:jmax]
        np.maximum(block, np.where(mask, zz, -np.inf), out=block)
    valid = np.isfinite(heights)
    coverage = float(valid.mean())
    if coverage < .25:
        raise ValueError("Projected surface coverage below 25 percent")
    if not valid.all():
        nearest = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        heights = heights[tuple(nearest)]
    return heights, coverage


def from_shap_e(vertices, faces, cfg):
    s, z, radius = pad_axes(cfg)
    h, coverage = projected_height(vertices, faces, (len(s), len(z)))
    h = gaussian_filter(h, cfg["pad"]["smoothing_sigma"], mode="nearest")
    low, high = np.percentile(h, [2, 98])
    if high - low < 1e-8:
        raise ValueError("Decoded shape projects to an almost flat surface")
    h = np.clip((h - low) / (high - low), 0, 1)
    p = cfg["pad"]
    thickness = (p["thickness_min_mm"] + h * (p["thickness_max_mm"] - p["thickness_min_mm"])) * .001
    # Reduce amplitude about the mean if needed; this preserves the slope bound
    # without manufacturing a different random surface or inserting a fallback.
    gs, gz = np.gradient(thickness, s, z, edge_order=1)
    max_slope = float(np.sqrt(gs * gs + gz * gz).max())
    if max_slope > p["max_slope"]:
        thickness = thickness.mean() + (thickness - thickness.mean()) * p["max_slope"] / max_slope
    return s, z, radius, thickness, coverage


def flat_pad(cfg, thickness_mm=3.5):
    s, z, radius = pad_axes(cfg)
    return s, z, radius, np.full((len(s), len(z)), thickness_mm * .001)


def grid_vertices(s, z, radius, thickness, embed_m, sign):
    ss, zz = np.meshgrid(s, z, indexing="ij")
    base = np.sqrt(radius * radius - zz * zz) - embed_m
    bottom = np.stack([-ss, sign * base, zz], axis=-1)
    top = np.stack([-ss, sign * (base + thickness), zz], axis=-1)
    return bottom, top


def surface_mesh(bottom, top):
    ni, nj = bottom.shape[:2]
    k = ni * nj
    verts = np.concatenate([bottom.reshape(-1, 3), top.reshape(-1, 3)])
    faces = []
    def index(i, j): return i * nj + j
    # The upper (-Y) side has outward winding for these index patterns.
    for i in range(ni - 1):
        for j in range(nj - 1):
            a, b, c, d = index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)
            faces.extend([(a, c, b), (a, d, c), (a + k, b + k, c + k), (a + k, c + k, d + k)])
    boundary = ([index(i, 0) for i in range(ni)] + [index(ni - 1, j) for j in range(1, nj)]
                + [index(i, nj - 1) for i in range(ni - 2, -1, -1)] + [index(0, j) for j in range(nj - 2, 0, -1)])
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        faces.extend([(a, b, b + k), (a, b + k, a + k)])
    faces = np.asarray(faces, dtype=np.int32)
    # Orient globally by signed volume (mirroring changes winding).
    tri = verts[faces]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    if volume < 0:
        faces = faces[:, [0, 2, 1]]
    return verts, faces


def convex_parts(bottom, top):
    # Two triangular prisms per grid cell: union preserves contact valleys.
    # Each prism is a proper convex hull; no full-pad hull is used for collision.
    for i in range(bottom.shape[0] - 1):
        for j in range(bottom.shape[1] - 1):
            for inds in [((i, j), (i + 1, j), (i + 1, j + 1)), ((i, j), (i + 1, j + 1), (i, j + 1))]:
                verts = np.array([arr[a, b] for arr in (bottom, top) for a, b in inds])
                hull = ConvexHull(verts)
                faces = hull.simplices.copy()
                for f, eq in zip(faces, hull.equations):
                    normal = np.cross(verts[f[1]] - verts[f[0]], verts[f[2]] - verts[f[0]])
                    if normal @ eq[:3] < 0:
                        f[1], f[2] = f[2], f[1]
                yield verts, faces, float(hull.volume)


def write_stl(path, vertices, faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = np.asarray(vertices)[np.asarray(faces)]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal /= np.maximum(norm, 1e-30)
    with path.open("wb") as f:
        f.write(b"padopt SI metres".ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tri)))
        for n, t in zip(normal, tri):
            f.write(struct.pack("<12fH", *n, *t.ravel(), 0))


def save_pad(candidate, cfg, s, z, radius, thickness):
    candidate = Path(candidate)
    candidate.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(candidate / "surface.npz", s=s, z=z, radius=radius, thickness=thickness)
    volume = 0
    for side, sign in [("upper", -1), ("lower", 1)]:
        bottom, top = grid_vertices(s, z, radius, thickness, cfg["pad"]["embed_mm"] * .001, sign)
        v, f = surface_mesh(bottom, top)
        write_stl(candidate / ("pad_" + side + ".stl"), v, f)
        if side == "upper":
            volume = sum(p[2] for p in convex_parts(bottom, top))
    return {"one_pad_volume_m3": volume, "one_pad_mass_kg": volume * cfg["pad"]["density"],
            "min_thickness_mm": float(thickness.min() * 1000), "max_thickness_mm": float(thickness.max() * 1000),
            "pad_length_mm": float(np.ptp(s) * 1000), "pad_width_mm": float(np.ptp(z) * 1000),
            "collision_parts_per_pad": 2 * (len(s) - 1) * (len(z) - 1)}
