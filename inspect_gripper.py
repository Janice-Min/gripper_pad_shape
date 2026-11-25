import argparse
import json
from padopt.common import read_config, parse_gripper
from padopt.geometry import pad_axes


def main():
    p = argparse.ArgumentParser(description="Inspect distal geometry without loading Shap-E or MuJoCo.")
    p.add_argument("--config", default="config.json")
    args = p.parse_args()
    cfg = read_config(args.config)
    s, z, radius = pad_axes(cfg)
    print(json.dumps({"distal_links": parse_gripper(cfg["base_xml"]), "pad_length_mm": float((s[-1] - s[0]) * 1000),
                      "pad_width_mm": float((z[-1] - z[0]) * 1000), "capsule_radius_mm": radius * 1000,
                      "collision_parts_per_pad": 2 * (len(s) - 1) * (len(z) - 1),
                      "units": "All meshes and XML use metres; *_mm config fields use millimetres."}, indent=2))


if __name__ == "__main__":
    main()
