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
        obj = root.find(".//body[@name='object_body']")
        geom = obj.find("geom[@name='object']")
        for key in ("type", "size", "mesh", "fromto", "quat", "euler", "axisangle", "pos", "density"):
            geom.attrib.pop(key, None)
        geom.set("type", object_case["type"])
        geom.set("size", numbers(object_case["size"]))
        geom.set("mass", str(cfg["evaluation"]["object_mass_kg"]))
        geom.set("friction", numbers(cfg["evaluation"]["object_friction"]))
        # Hold the object only while the gripper closes. Disable this equality
        # before every measurement: no object support remains during hold tests.
        pos = [cfg["evaluation"]["position_x_m"], 0, cfg["evaluation"]["position_z_m"]]
        obj.set("pos", numbers(pos))
        ET.SubElement(root.find("worldbody"), "body", name="object_fixture", mocap="true", pos=numbers(pos))
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(equality, "weld", name="temporary_object_hold", body1="object_body", body2="object_fixture",
                      relpose="0 0 0 1 0 0 0", active="true", solref="0.002 1")
    return root


def build_xml(cfg, candidate, output=None, check_metadata=True):
    candidate = Path(candidate).resolve()
    output = Path(output).resolve() if output else candidate / "gripper_with_pad.xml"
    root = model_tree(cfg, candidate, asset_relative_to=output.parent, check_metadata=check_metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output
