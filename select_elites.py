import argparse
import csv
import json
from pathlib import Path
from padopt.common import read_config, save_json


def main():
    p = argparse.ArgumentParser(description="Select elites from TRAIN results only, using a documented lexicographic ranking.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--summary", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = read_config(args.config)
    with Path(args.summary).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["suite"] != "train" for r in rows):
        raise ValueError("Select with train results only. Keep validation data out of optimization.")
    count = args.count or cfg["optimization"]["elite_count"]
    eligible = []
    for row in rows:
        candidate = (Path(args.summary).resolve().parent / row["candidate_dir"]).resolve()
        meta_path = candidate / "metadata.json"
        if not row["candidate_dir"] or not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source") != "shap_e":
            continue
        eligible.append((row, candidate))
    if len(eligible) < count:
        raise ValueError("Need at least " + str(count) + " evaluated Shap-E candidates")
    eligible.sort(key=lambda item: (-float(item[0]["success_rate"]), -float(item[0]["worst_object_success"]), int(item[0]["invalid_trials"]),
                                   -float(item[0]["contact_fraction"]), float(item[0]["mean_drift_mm"]),
                                   float(item[0]["one_pad_volume_m3"]), item[0]["candidate_id"]))
    if float(eligible[0][0]["success_rate"]) <= 0:
        raise RuntimeError("Every design failed. Check object placement, actuator authority and protocol before optimizing.")
    selected = [{"candidate_dir": str(path), "candidate_id": row["candidate_id"], "metrics": row} for row, path in eligible[:count]]
    save_json(args.out, {"suite": "train", "ranking": ["success_rate desc", "worst_object_success desc", "invalid_trials asc", "contact_fraction desc", "mean_drift_mm asc", "pad_volume asc"],
                         "summary_path": str(Path(args.summary).resolve()), "elites": selected})
    print("Selected " + str(len(selected)) + " elites -> " + str(args.out))


if __name__ == "__main__":
    main()
