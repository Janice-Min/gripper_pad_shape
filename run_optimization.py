"""Run complete generation -> evaluation -> selection -> embedding-fit rounds."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from padopt.common import read_config, source_signature, file_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--start-round", type=int, default=0)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--shap-e-dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-modified-mc", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    if args.count < cfg["optimization"]["elite_count"] or args.rounds < 1 or args.start_round < 0:
        p.error("Need count >= elite_count, rounds >= 1 and start-round >= 0")
    root = Path(__file__).resolve().parent
    base_args = ["--config", cfg["config_path"]]
    model_args = ["--device", args.device]
    if args.shap_e_dir:
        model_args += ["--shap-e-dir", str(Path(args.shap_e_dir).resolve())]
    if args.allow_modified_mc:
        model_args += ["--allow-modified-mc"]
    def run(script, *flags):
        command = [sys.executable, str(root / script), *base_args, *flags]
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=root, check=True)
    for iteration in range(args.start_round, args.start_round + args.rounds):
        folder = Path(cfg["output_dir"]) / ("round_" + str(iteration).zfill(3))
        gen = ["--round", str(iteration), "--count", str(args.count), *model_args]
        if iteration > 0:
            previous = Path(cfg["output_dir"]) / ("round_" + str(iteration - 1).zfill(3)) / "next_embeddings.pt"
            gen += ["--embedding", str(previous)]
        run("generate_pads.py", *gen)
        evaluation = folder / "eval_train"
        run("evaluate_pads.py", "--round-dir", str(folder), "--suite", "train", "--repeats", str(args.repeats), "--out", str(evaluation))
        elites = folder / "elites.json"
        run("select_elites.py", "--summary", str(evaluation / "summary.csv"), "--out", str(elites))
        bank = folder / "next_embeddings.pt"
        if not bank.exists():
            run("optimize_embeddings.py", "--elites", str(elites), "--out", str(bank), *model_args)
        else:
            manifest = json.loads(bank.with_suffix(".json").read_text(encoding="utf-8"))
            if (manifest["config"] != cfg["optimization"] or manifest["source_signature"] != source_signature(cfg)
                    or manifest["elite_selection_sha256"] != file_sha(elites)):
                raise RuntimeError("Existing embedding bank has different training settings/elites. Choose a new experiment output directory.")
    print("Training rounds complete. Choose the final design using train results, then evaluate it once on validation cases.")


if __name__ == "__main__":
    main()
