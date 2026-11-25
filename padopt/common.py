from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np


def read_config(path="config.json"):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("base_xml", "output_dir", "shap_e_dir", "cache_dir"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
    cfg["config_path"] = str(path)
    return cfg


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def parse_gripper(path):
    root = ET.parse(path).getroot()
    result = {}
    for side, body_name, geom_name, sign in (
        ("upper", "upper_link2", "upper_red_link", -1),
        ("lower", "lower_link2", "lower_red_link", 1),
    ):
        body = root.find(".//body[@name='" + body_name + "']")
        geom = None if body is None else body.find("geom[@name='" + geom_name + "']")
        if geom is None or geom.get("type") != "capsule":
            raise ValueError("Expected capsule geom in " + body_name)
        endpoints = np.fromstring(geom.get("fromto", ""), sep=" ")
        radius = float(geom.get("size", "0").split()[0])
        if endpoints.shape != (6,) or not np.allclose(endpoints[:3], 0) or not np.allclose(endpoints[4:], 0) or endpoints[3] >= 0:
            raise ValueError("This adapter expects capsules from the body origin along local -X.")
        result[side] = {"body": body_name, "geom": geom_name, "length_m": float(-endpoints[3]), "radius_m": radius, "inward_sign": sign}
    if not np.isclose(result["upper"]["length_m"], result["lower"]["length_m"]) or not np.isclose(result["upper"]["radius_m"], result["lower"]["radius_m"]):
        raise ValueError("This implementation uses the same mirrored pad on symmetric distal links.")
    return result


def source_signature(cfg):
    return digest({"xml_sha256": file_sha(cfg["base_xml"]), "pad": cfg["pad"]})


def assert_candidate_compatible(cfg, candidate):
    metadata = json.loads((Path(candidate) / "metadata.json").read_text(encoding="utf-8"))
    if metadata["source_signature"] != source_signature(cfg):
        raise ValueError("Candidate geometry/base XML differs from this config. Use its original config.")
    return metadata


def candidate_dirs(round_dir):
    return [p for p in sorted(Path(round_dir).glob("pad_*")) if (p / "metadata.json").is_file()]
