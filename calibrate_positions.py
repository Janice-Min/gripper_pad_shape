"""Place every test object where the PADS make the grasp.

For each object the nominal x-position is swept. At each position the fingers close on
the object standing free on its support, and the finger parts touching the object are
recorded. The chosen position is the middle of the longest run of positions where both
pads touch the object and no other finger part does (fallback: both pads plus links).
Success is NOT used, so the placement does not favour any pad design.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from padopt.common import read_config
from padopt.evaluation import compile_case, single_trial

CODES = {"B": "both pads only", "b": "both pads + other finger parts", "p": "one pad",
         "L": "finger links only (no pad)", "-": "no contact", "o": "initial overlap"}


def classify(contacts):
    contacts = set(contacts)
    if {"upper_pad", "lower_pad"} <= contacts:
        return "B" if "link" not in contacts else "b"
    if contacts & {"upper_pad", "lower_pad"}:
        return "p"
    return "L" if "link" in contacts else "-"


def longest_run(xs, codes, accepted):
    best, current = [], []
    for x, code in zip(xs, codes):
        current = current + [x] if code in accepted else []
        if len(current) > len(best):
            best = current
    return best


def write_positions(config_path, chosen):
    text = Path(config_path).read_text(encoding="utf-8")
    for name, x in chosen.items():
        pattern = re.compile(r'(\{"name": "' + re.escape(name) + r'"[^\n]*?)(, "position_x_m": [-0-9.eE]+)?\}')
        text, count = pattern.subn(lambda m: m.group(1) + ', "position_x_m": ' + format(x, ".4f") + "}", text, count=1)
        if count != 1:
            raise ValueError("Could not find a one-line case entry for " + name + " in " + str(config_path))
    json.loads(text)
    Path(config_path).write_text(text, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.json")
    p.add_argument("--candidate", default="runs/baselines/pad_flat", help="Pad used to locate the pad region (default: flat control pad)")
    p.add_argument("--suite", choices=["train", "validation", "both"], default="both")
    p.add_argument("--x-near-mm", type=float, default=-25.0, help="Sweep start (closest to the palm)")
    p.add_argument("--x-far-mm", type=float, default=-100.0, help="Sweep end (farthest from the palm)")
    p.add_argument("--step-mm", type=float, default=2.5)
    p.add_argument("--write", action="store_true", help="Write the chosen positions into --config")
    p.add_argument("--also-write", nargs="*", default=[], help="Other config files that must use the same positions (e.g. config_stress.json)")
    args = p.parse_args()
    cfg = read_config(args.config)
    e = cfg["evaluation"]
    xs = [float(x) / 1000 for x in np.arange(args.x_near_mm, args.x_far_mm - 1e-9, -abs(args.step_mm))]
    suites = ["train", "validation"] if args.suite == "both" else [args.suite]
    print("Codes: " + ", ".join(k + " = " + v for k, v in CODES.items()))
    print("x [mm]".ljust(26) + "".join(format(x * 1000, ">6.1f") for x in xs))
    chosen, report = {}, []
    for suite in suites:
        for case in e[suite]:
            model, data = compile_case(cfg, args.candidate, case)
            codes = []
            for x in xs:
                pos, quat = np.array([x, 0.0, e["position_z_m"]]), np.array([1.0, 0, 0, 0])
                r = single_trial(cfg, model, data, pos, quat, until="closing")
                codes.append("o" if r["status"] == "initial_overlap" else classify(r["closing_contacts"]))
            run = longest_run(xs, codes, {"B"}) or longest_run(xs, codes, {"B", "b"})
            x_mid = round(float(np.mean(run)) * 2000) / 2000 if run else None
            if x_mid is not None:
                chosen[case["name"]] = x_mid
            label = suite[0] + ":" + case["name"]
            print(label.ljust(26) + "".join(format(c, ">6s") for c in codes)
                  + ("   -> " + format(x_mid * 1000, ".1f") + " mm" if x_mid is not None else "   -> no pad-only position"), flush=True)
            report.append({"suite": suite, "name": case["name"], "codes": codes, "chosen_x_m": x_mid})
    missing = [r["name"] for r in report if r["chosen_x_m"] is None]
    if missing:
        print("No position where both pads touch: " + ", ".join(missing) + ". These keep their current position.")
    if args.write:
        for path in [cfg["config_path"], *args.also_write]:
            write_positions(path, chosen)
            print("Wrote position_x_m for " + str(len(chosen)) + " objects into " + str(path))
    else:
        print("Dry run. Add --write to store these positions in " + cfg["config_path"])


if __name__ == "__main__":
    main()
