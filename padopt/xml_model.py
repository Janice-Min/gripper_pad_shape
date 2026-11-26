from pathlib import Path
import os
import xml.etree.ElementTree as ET
import numpy as np
from .common import assert_candidate_compatible, parse_gripper
from .geometry import grid_vertices, convex_parts


def numbers(values):
    return " ".join(format(float(x), ".10g") for x in np.asarray(values).ravel())


def model_tree(cfg, candidate=None, asset_relative_to=None, object_case=None, check_metadata=True):
    root = ET.parse(cfg["base_xml"]).getroot()
    if candidate is not None:
        candidate = Path(candidate).resolve()
        if check_metadata:
            assert_candidate_compatible(cfg, candidate)
        with np.load(candidate / "surface.npz", allow_pickle=False) as grid:
            s, z, radius, thickness = grid["s"], grid["z"], float(grid["radius"]), grid["thickness"]
        asset = root.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            root.insert(list(root).index(root.find("worldbody")), asset)
        for side, link in parse_gripper(cfg["base_xml"]).items():
            body = root.find(".//body[@name='" + link["body"] + "']")
            mesh_file = candidate / ("pad_" + side + ".stl")
            mesh_path = mesh_file.as_posix() if asset_relative_to is None else Path(os.path.relpath(mesh_file, asset_relative_to)).as_posix()
            ET.SubElement(asset, "mesh", name="pad_visual_" + side, file=mesh_path)
            ET.SubElement(body, "geom", name="pad_visual_" + side, type="mesh", mesh="pad_visual_" + side,
                          contype="0", conaffinity="0", mass="0", group="2", rgba="0.12 0.72 0.65 1")
            bottom, top = grid_vertices(s, z, radius, thickness, cfg["pad"]["embed_mm"] * .001, link["inward_sign"])
            for k, (vertices, faces, volume) in enumerate(convex_parts(bottom, top)):
                name = "pad_collision_" + side + "_" + str(k)
                ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices), face=" ".join(map(str, faces.ravel())))
                ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, mass=format(volume * cfg["pad"]["density"], ".12g"),
                              friction=numbers(cfg["pad"]["friction"]), condim="4", group="3", rgba="0.12 0.72 0.65 0")
    if object_case is not None:
        add_test_rig(root, cfg, object_case)
    return root


HALF_HEIGHT = {
    "sphere": lambda size: size[0],
    "cylinder": lambda size: size[1],
    "capsule": lambda size: size[0] + size[1],
    "box": lambda size: size[2],
    "ellipsoid": lambda size: size[2],
}


def object_half_height(geom_type, size):
    """Vertical half extent of an upright test object (z axis = object axis)."""
    if geom_type not in HALF_HEIGHT:
        raise ValueError("Unsupported test object type for the support: " + str(geom_type))
    return float(HALF_HEIGHT[geom_type](list(size)))


def add_test_rig(root, cfg, object_case):
    """Replace the XML object by a test object resting on a movable support plate.

    No force holds the object: from t = 0 it is free under gravity and only the
    support plate (a mocap body that is lowered later) keeps it from falling.
    """
    e = cfg["evaluation"]
    obj = root.find(".//body[@name='object_body']")
    geom = obj.find("geom[@name='object']")
    for key in ("type", "size", "mesh", "fromto", "quat", "euler", "axisangle", "pos", "density"):
        geom.attrib.pop(key, None)
    geom.set("type", object_case["type"])
    geom.set("size", numbers(object_case["size"]))
    geom.set("mass", str(e["object_mass_kg"]))
    geom.set("friction", numbers(e["object_friction"]))
    x = object_case.get("position_x_m", e["position_x_m"])
    pos = [x, 0, e["position_z_m"]]
    obj.set("pos", numbers(pos))
    sx, sy, sz = (v * .001 for v in e["support_size_mm"])
    top = e["position_z_m"] - object_half_height(object_case["type"], object_case["size"])
    support = ET.SubElement(root.find("worldbody"), "body", name="object_support", mocap="true",
                            pos=numbers([x, 0, top - sz / 2]))
    ET.SubElement(support, "geom", name="support", type="box", size=numbers([sx / 2, sy / 2, sz / 2]),
                  friction=numbers(e["support_friction"]), rgba="0.62 0.5 0.36 0.55")


def gripper_root_body(root, cfg):
    """The worldbody child that carries both fingers (``palm`` in design0009.xml)."""
    parents = {child: parent for parent in root.iter() for child in parent}
    worldbody = root.find("worldbody")
    body = root.find(".//body[@name='" + parse_gripper(cfg["base_xml"])["upper"]["body"] + "']")
    while parents[body] is not worldbody:
        body = parents[body]
    return body


def add_lift_axis(root, cfg):
    """Give the gripper base one vertical slide joint driven by a stiff position actuator.

    Used only for evaluation models, so the gripper can pick the object up. The
    original joints, tendon and motor are untouched; keyframes get the new slide
    coordinate (0) inserted at its qpos/qvel address and a 0 ctrl for the new actuator.
    """
    import mujoco as mj
    e = cfg["evaluation"]
    before = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    body = gripper_root_body(root, cfg)
    bid = mj.mj_name2id(before, mj.mjtObj.mjOBJ_BODY, body.get("name"))
    later = [j for j in range(before.njnt) if before.jnt_bodyid[j] >= bid]
    qadr = int(before.jnt_qposadr[later[0]]) if later else before.nq
    vadr = int(before.jnt_dofadr[later[0]]) if later else before.nv
    body.insert(0, ET.Element("joint", name="lift_slide", type="slide", axis="0 0 1", damping="0", armature="0"))
    actuators = root.find("actuator")
    if actuators is None:
        actuators = ET.SubElement(root, "actuator")
    kp, kv = float(e["lift_kp"]), float(e["lift_kv"])
    ET.SubElement(actuators, "general", name="lift_actuator", joint="lift_slide", gaintype="fixed", biastype="affine",
                  gainprm=numbers([kp]), biasprm=numbers([0, -kp, -kv]))
    for key in root.iter("key"):
        for attribute, address in (("qpos", qadr), ("qvel", vadr)):
            if key.get(attribute):
                values = key.get(attribute).split()
                values.insert(address, "0")
                key.set(attribute, " ".join(values))
        if key.get("ctrl"):
            key.set("ctrl", key.get("ctrl") + " 0")
    return root


def build_xml(cfg, candidate, output=None, check_metadata=True):
    candidate = Path(candidate).resolve()
    output = Path(output).resolve() if output else candidate / "gripper_with_pad.xml"
    root = model_tree(cfg, candidate, asset_relative_to=output.parent, check_metadata=check_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output
