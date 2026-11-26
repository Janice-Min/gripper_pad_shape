"""Choose, for every test object, the x-position at which the grasp has the wanted type.

For each object the nominal x-position is swept. At each position the fingers close on
the object standing free on its support, and the finger parts touching the object are
recorded. The chosen position is the swept position nearest the middle of the longest run
of the wanted grasp. Success is NOT used, so the placement does not favour any pad design.

--grasp envelop (default): middle of the longest run of positions where all four links
    (both proximal and both distal links) touch the object and both pads touch it
    (fallback: all four links, pads not required).
--grasp pad: middle of the longest run where both pads and no other finger part touch
    (fallback: both pads plus other finger parts).
Objects that cannot reach the wanted grasp anywhere in the sweep fall back to the pad
rule (--fallback pad) or keep their current position (--fallback keep).
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from padopt.common import read_config
from padopt.evaluation import compile_case, single_trial

LINKS = ("upper_link1", "upper_link2", "lower_link1", "lower_link2")
CODES = {"E": "all 4 links + both pads (enveloping)", "e": "all 4 links, pads not both",
         "B": "both pads only", "b": "both pads + other finger parts", "p": "one pad",
         "L": "finger links only (no pad)", "-": "no contact", "o": "initial overlap"}


def classify(contacts):
    """Pad-grasp code from the pad/other-finger contact labels."""
    contacts = set(contacts)
    if {"upper_pad", "lower_pad"} <= contacts:
        return "B" if "link" not in contacts else "b"
    if contacts & {"upper_pad", "lower_pad"}:
        return "p"
    return "L" if "link" in contacts else "-"


def classify_envelop(contacts, bodies):
    """'E'/'e' when all four links touch the object, otherwise the pad-grasp code."""
    if all(link in bodies for link in LINKS):
        return "E" if {"upper_pad", "lower_pad"} <= set(contacts) else "e"
    return classify(contacts)


def longest_run(xs, codes, accepted):
    best, current = [], []
    for x, code in zip(xs, codes):
        current = current + [x] if code in accepted else []
        if len(current) > len(best):
            best = current
    return best


def choose(xs, codes, grasp, fallback):
    """Return (x or None, rule used)."""
    rules = [("envelop", {"E"}), ("envelop", {"E", "e"})] if grasp == "envelop" else []
    if grasp == "pad" or fallback == "pad":
        rules += [("pad", {"B"}), ("pad", {"B", "b"})]
    for name, accepted in rules:
        run = longest_run(xs, codes, accepted)
        if run:
            # The swept position closest to the middle of the run (always a position that was observed).
            middle = float(np.mean(run))
            return min(run, key=lambda x: abs(x - middle)), name
    return None, "keep"


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
    p.add_argument("--candidate", default="runs/baselines/pad_flat", help="Pad used for the sweep (default: flat control pad)")
    p.add_argument("--suite", choices=["train", "validation", "both"], default="both")
    p.add_argument("--grasp", choices=["envelop", "pad"], default="envelop", help="Wanted grasp type")
    p.add_argument("--fallback", choices=["pad", "keep"], default="pad",
                   help="For objects that never reach the wanted grasp: use the pad rule, or keep the current position")
    p.add_argument("--x-near-mm", type=float, default=-25.0, help="Sweep start (closest to the palm)")
    p.add_argument("--x-far-mm", type=float, default=-100.0, help="Sweep end (farthest from the palm)")
    p.add_argument("--step-mm", type=float, default=1.5)
    p.add_argument("--write", action="store_true", help="Write the chosen positions into --config")
    p.add_argument("--also-write", nargs="*", default=[], help="Other config files that must use the same positions (e.g. config_light.json)")
    args = p.parse_args()
    cfg = read_config(args.config)
    e = cfg["evaluation"]
    xs = [float(x) / 1000 for x in np.arange(args.x_near_mm, args.x_far_mm - 1e-9, -abs(args.step_mm))]
    suites = ["train", "validation"] if args.suite == "both" else [args.suite]
    shown = CODES if args.grasp == "envelop" else {k: v for k, v in CODES.items() if k not in "Ee"}
    print("Codes: " + ", ".join(k + " = " + v for k, v in shown.items()))
    print("x [mm]".ljust(26) + "".join(format(x * 1000, ">6.1f") for x in xs))
    chosen, report = {}, []
    for suite in suites:
        for case in e[suite]:
            model, data = compile_case(cfg, args.candidate, case)
            codes = []
            for x in xs:
                pos, quat = np.array([x, 0.0, e["position_z_m"]]), np.array([1.0, 0, 0, 0])
                r = single_trial(cfg, model, data, pos, quat, until="closing")
                if r["status"] == "initial_overlap":
                    codes.append("o")
                elif args.grasp == "envelop":
                    codes.append(classify_envelop(r["closing_contacts"], r["closing_bodies"]))
                else:
                    codes.append(classify(r["closing_contacts"]))
            x_mid, rule = choose(xs, codes, args.grasp, args.fallback)
            if x_mid is not None:
                chosen[case["name"]] = x_mid
            label = suite[0] + ":" + case["name"]
            note = ("   -> " + format(x_mid * 1000, ".1f") + " mm (" + rule + ")") if x_mid is not None else "   -> keep current position"
            print(label.ljust(26) + "".join(format(c, ">6s") for c in codes) + note, flush=True)
            report.append({"suite": suite, "name": case["name"], "codes": codes, "chosen_x_m": x_mid, "rule": rule})
    other = [r["name"] for r in report if r["rule"] != args.grasp]
    if other:
        print("Never reached the '" + args.grasp + "' grasp: " + ", ".join(other) + " (see the rule in brackets).")
    if args.write:
        for path in [cfg["config_path"], *args.also_write]:
            write_positions(path, chosen)
            print("Wrote position_x_m for " + str(len(chosen)) + " objects into " + str(path))
    else:
        print("Dry run. Add --write to store these positions in " + cfg["config_path"])


if __name__ == "__main__":
    main()
