import argparse
from padopt.common import read_config
from padopt.xml_model import build_xml


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rebuild mounted XML from a candidate surface.npz.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--candidate", required=True)
    p.add_argument("--out")
    args = p.parse_args()
    print(build_xml(read_config(args.config), args.candidate, args.out))
