import argparse
from padopt.common import read_config
from padopt.evaluation import compile_case, case_pose, single_trial


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replay the same evaluation protocol in the MuJoCo viewer.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--candidate")
    p.add_argument("--suite", choices=["train", "validation"], default="train")
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--repeat", type=int, default=0)
    args = p.parse_args()
    cfg = read_config(args.config)
    case = cfg["evaluation"][args.suite][args.case]
    model, data = compile_case(cfg, args.candidate, case)
    print(single_trial(cfg, model, data, *case_pose(cfg, args.case, args.repeat, args.suite), visualize=True))
