import argparse
import csv
import json
from pathlib import Path
import os
from padopt.common import read_config, save_json, digest, source_signature, candidate_dirs, file_sha
from padopt.evaluation import compile_case, case_pose, single_trial, summarize


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description="Evaluate multi-object gripping under gravity and a shared disturbance.")
    p.add_argument("--config", default="config.json")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--round-dir")
    group.add_argument("--candidate")
    group.add_argument("--bare", action="store_true")
    p.add_argument("--suite", choices=["train", "validation"], default="train")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    cfg = read_config(args.config)
    candidates = [None] if args.bare else ([Path(args.candidate).resolve()] if args.candidate else candidate_dirs(args.round_dir))
    if args.limit:
        candidates = candidates[:args.limit]
    if not candidates:
        raise ValueError("No completed candidates found")
    out = Path(args.out).resolve()
    records = out / "records"
    records.mkdir(parents=True, exist_ok=True)
    signature = digest({"source": source_signature(cfg), "evaluation": cfg["evaluation"], "suite": args.suite,
                        "repeats": args.repeats, "code_sha": [file_sha(Path(__file__).parent / "padopt" / name) for name in ("evaluation.py", "xml_model.py")]})
    if (out / "evaluation.json").exists() and json.loads((out / "evaluation.json").read_text())["signature"] != signature:
        raise RuntimeError("Evaluation protocol changed. Choose a new --out directory.")
    save_json(out / "evaluation.json", {"signature": signature, "suite": args.suite, "repeats": args.repeats,
                                       "config": cfg, "note": "Object fixture disabled before all hold/disturbance measurements."})
    aggregates, all_trials = [], []
    for i, candidate in enumerate(candidates):
        cid = candidate.name if candidate else "bare_gripper"
        identity = digest({"candidate": str(candidate), "surface": file_sha(candidate / "surface.npz") if candidate else None})
        record = records / (cid + "_" + identity[:12] + ".json")
        if record.exists():
            value = json.loads(record.read_text(encoding="utf-8"))
            if value["signature"] != signature:
                raise RuntimeError("Cached evaluation signature mismatch")
        else:
            rows = []
            for ci, case in enumerate(cfg["evaluation"][args.suite]):
                model, data = compile_case(cfg, candidate, case)
                for repeat in range(args.repeats):
                    pos, quat = case_pose(cfg, ci, repeat, args.suite)
                    r = single_trial(cfg, model, data, pos, quat)
                    rows.append({"candidate_id": cid, "suite": args.suite, "object_name": case["name"], "repeat": repeat, **r})
            volume = json.loads((candidate / "metadata.json").read_text())["one_pad_volume_m3"] if candidate else 0.0
            summary = {"candidate_id": cid, "candidate_dir": Path(os.path.relpath(candidate, out)).as_posix() if candidate else "",
                       "suite": args.suite, "one_pad_volume_m3": volume, **summarize(rows)}
            value = {"signature": signature, "summary": summary, "trials": rows}
            save_json(record, value)
        aggregates.append(value["summary"])
        all_trials.extend(value["trials"])
        write_csv(out / "summary.csv", aggregates)
        write_csv(out / "trials.csv", all_trials)
        summary = value["summary"]
        print("[" + str(i + 1) + "/" + str(len(candidates)) + "] " + cid + " score=" + format(summary["mean_score"], ".2f") + "/"
              + format(summary["max_score"], "g") + " success=" + format(summary["success_rate"], ".3f"), flush=True)


if __name__ == "__main__":
    main()
