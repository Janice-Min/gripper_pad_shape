import argparse
import json
from pathlib import Path
from padopt.common import read_config, save_json, assert_candidate_compatible, source_signature, file_sha
from padopt.shap_e_runtime import initialize, load_tensor, seed_everything


def main():
    p = argparse.ArgumentParser(description="Freeze Shap-E and fit one condition embedding per elite latent with diffusion training loss.")
    p.add_argument("--config", default="config.json")
    p.add_argument("--elites", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shap-e-dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-modified-mc", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    target = Path(args.out)
    if target.exists():
        raise FileExistsError("Refusing to overwrite an existing embedding bank: " + str(target))
    selection = json.loads(Path(args.elites).read_text(encoding="utf-8"))
    if selection["suite"] != "train":
        raise ValueError("Training elites required")
    rt = initialize(cfg, args.shap_e_dir, args.device, args.allow_modified_mc, decoder=False)
    torch, opt = rt["torch"], cfg["optimization"]
    seed_everything(torch, opt["seed"])
    embeddings, histories, gradient_histories, candidates = [], [], [], []
    # Sequential fitting keeps the diffusion backprop batch at one to reduce VRAM.
    for elite in selection["elites"]:
        folder = Path(elite["candidate_dir"])
        meta = assert_candidate_compatible(cfg, folder)
        if meta["source"] != "shap_e":
            raise ValueError("Synthetic/baseline pads cannot be used as Shap-E training targets")
        latent = load_tensor(torch, folder / "latent.pt").to(rt["device"]).float()[None]
        condition = load_tensor(torch, folder / "condition.pt").to(rt["device"]).float()
        embed = torch.nn.Parameter(condition.clone())
        anchor = condition.detach()
        optimizer = torch.optim.AdamW([embed], lr=opt["learning_rate"], weight_decay=opt["weight_decay"])
        history = []
        gradient_history = []
        for iteration in range(opt["steps_per_embedding"]):
            optimizer.zero_grad(set_to_none=True)
            ts = torch.randint(0, rt["diffusion"].num_timesteps, (1,), device=rt["device"])
            terms = rt["diffusion"].training_losses(model=rt["model"], x_start=latent, t=ts, model_kwargs={"embeddings": embed})
            loss = terms["loss"].mean() + opt["anchor_weight"] * (embed - anchor).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diffusion loss")
            loss.backward()
            if embed.grad is None or not torch.isfinite(embed.grad).all():
                raise RuntimeError("Embedding gradients are missing/nonfinite. Check the selected Shap-E version.")
            gradient_history.append(float(embed.grad.norm().detach().cpu()))
            torch.nn.utils.clip_grad_norm_([embed], opt["gradient_clip"])
            optimizer.step()
            history.append(float(loss.detach().cpu()))
            if iteration % 10 == 0:
                print(folder.name, iteration, history[-1], flush=True)
        embeddings.append(embed.detach().cpu())
        histories.append(history)
        gradient_histories.append(gradient_history)
        candidates.append(str(folder))
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": torch.cat(embeddings), "source_signature": source_signature(cfg), "elite_dirs": candidates,
                "elite_selection_sha256": file_sha(args.elites)}, target)
    save_json(target.with_suffix(".json"), {"source_signature": source_signature(cfg), "config": opt, "elite_dirs": candidates,
                                          "elite_selection_sha256": file_sha(args.elites),
                                          "loss_histories": histories, "gradient_norm_histories": gradient_histories, "shap_e": rt["provenance"]})
    print("Saved embedding bank:", target)


if __name__ == "__main__":
    main()
