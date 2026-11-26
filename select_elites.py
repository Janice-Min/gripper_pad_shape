import argparse
import csv
import json
from pathlib import Path
from padopt.common import read_config, save_json

RANKING = ["mean_score desc", "worst_object_score desc", "pad_contact_fraction desc", "invalid_trials asc",
           "mean_drift_mm asc", "pad_volume asc"]
COLUMNS = ["rank", "candidate_id", "mean_score", "max_score", "score_rate", "worst_object_score", "held_after_lift_rate",
           "pad_contact_fraction", "success_rate", "pad_success_rate", "invalid_trials", "mean_drift_mm", "one_pad_volume_m3",
           "candidate_dir"]


def sort_key(row):
    return (-float(row["mean_score"]), -float(row["worst_object_score"]), -float(row["pad_contact_fraction"]),
            int(row["invalid_trials"]), float(row["mean_drift_mm"]), float(row["one_pad_volume_m3"]), row["candidate_id"])


def main():
    p = argparse.ArgumentParser(description="Rank Shap-E candidates by their TRAIN score and select the elites.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--summary", required=True)
    p.add_argument("--count", type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--ranking-out", help="CSV with the rank of every candidate (default: ranking.csv next to --out)")
    p.add_argument("--show", type=int, default=20, help="How many top ranks to print")
    args = p.parse_args()
    cfg = read_config(args.config)
    with Path(args.summary).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["suite"] != "train" for r in rows):
        raise ValueError("Select with train results only. Keep validation data out of optimization.")
    if not {"mean_score", "worst_object_score"} <= set(rows[0]):
        raise ValueError("This summary.csv was made before scoring was added (no mean_score). Re-evaluate with evaluate_pads.py into a new --out folder.")
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
    eligible.sort(key=lambda item: sort_key(item[0]))
    if float(eligible[0][0]["mean_score"]) <= 0:
        raise RuntimeError("No design scored any points (every object dropped before the first disturbance). "
                           "Check trials.csv (status, closing_shift_mm, drop_time_s) and the protocol before optimizing.")

    ranking_path = Path(args.ranking_out) if args.ranking_out else Path(args.out).resolve().parent / "ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    with ranking_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for rank, (row, candidate) in enumerate(eligible, 1):
            writer.writerow({**row, "rank": rank, "candidate_dir": str(candidate)})

    selected = [{"rank": rank, "candidate_dir": str(path), "candidate_id": row["candidate_id"], "metrics": row}
                for rank, (row, path) in enumerate(eligible[:count], 1)]
    save_json(args.out, {"suite": "train", "ranking": RANKING, "summary_path": str(Path(args.summary).resolve()),
                         "ranking_csv": str(ranking_path.resolve()), "elites": selected})

    shown = eligible[:max(args.show, 0)]
    width = max([len("candidate")] + [len(row["candidate_id"]) for row, _ in shown])
    print(" rank  " + "candidate".ljust(width) + "  score (mean / max)  worst object  pad contact")
    for rank, (row, _) in enumerate(shown, 1):
        mark = "*" if rank <= count else " "
        print("%s%4d  %s  %8.2f / %-8g  %12.2f  %11.2f" % (mark, rank, row["candidate_id"].ljust(width), float(row["mean_score"]),
              float(row["max_score"]), float(row["worst_object_score"]), float(row["pad_contact_fraction"])))
    print("Selected " + str(len(selected)) + " elites (*) -> " + str(args.out))
    print("Full ranking of " + str(len(eligible)) + " candidates -> " + str(ranking_path))


if __name__ == "__main__":
    main()
