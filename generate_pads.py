"""Generate exactly N valid, distinct pad candidates with Shap-E (default 500)."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
from padopt.common import read_config, save_json, source_signature, file_sha, digest
from padopt.geometry import from_shap_e, save_pad, write_stl
from padopt.xml_model import build_xml
from padopt.shap_e_runtime import initialize, load_tensor, seed_everything


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.json")
    p.add_argument("--count", type=int)
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--embedding", help="An embedding bank made by optimize_embeddings.py")
    p.add_argument("--shap-e-dir")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-modified-mc", action="store_true")
    args = p.parse_args()
    cfg = read_config(args.config)
    count = args.count if args.count is not None else cfg["generation"]["count"]
    if count < 1 or args.round < 0:
        p.error("count must be positive and round must be non-negative")
    if args.round > 0 and not args.embedding:
        p.error("Later rounds require --embedding; this prevents accidentally starting a new random round.")
    rt = initialize(cfg, args.shap_e_dir, args.device, args.allow_modified_mc)
    torch = rt["torch"]
    from shap_e.diffusion.sample import sample_latents
    from shap_e.util.notebooks import decode_latent_mesh
    bank = None
    if args.embedding:
        obj = load_tensor(torch, args.embedding)
        if isinstance(obj, dict) and obj.get("source_signature") != source_signature(cfg):
            raise ValueError("Embedding bank was fitted for a different pad domain/base gripper")
        bank = obj["embeddings"] if isinstance(obj, dict) else obj
        if bank.ndim != 2 or bank.shape[1] != 768 or len(bank) == 0 or not torch.isfinite(bank).all():
            raise ValueError("Expected a finite K x 768 embedding bank")
    folder = Path(cfg["output_dir"]) / ("round_" + str(args.round).zfill(3))
    folder.mkdir(parents=True, exist_ok=True)
    signature = digest({"source": source_signature(cfg), "generation": cfg["generation"], "round": args.round,
                        "embedding_sha": file_sha(args.embedding) if args.embedding else None, "shap_e": rt["provenance"]})
    if (folder / "run.json").exists():
        previous = json.loads((folder / "run.json").read_text())
        if previous["signature"] != signature:
            raise RuntimeError("This round already has a different configuration/source. Use a new output_dir.")
    save_json(folder / "run.json", {"signature": signature, "config": cfg, "round": args.round,
                                  "embedding_path": str(Path(args.embedding).resolve()) if args.embedding else None,
                                  "shap_e": rt["provenance"]})
    seen = set()
    successes = 0
    for index in range(count):
        candidate = folder / ("pad_" + str(index).zfill(4))
        metadata_file = candidate / "metadata.json"
        if metadata_file.exists():
            previous = json.loads(metadata_file.read_text())
            if previous.get("run_signature") != signature or not (candidate / "latent.pt").exists() or not (candidate / "gripper_with_pad.xml").exists():
                raise RuntimeError("Incomplete or incompatible committed candidate: " + str(candidate))
            seen.add(previous["surface_hash"])
            successes += 1
            continue
        candidate.mkdir(parents=True, exist_ok=True)
        for attempt in range(cfg["generation"]["max_attempts_per_candidate"]):
            seed = cfg["generation"]["seed"] + args.round * 10000000 + index * 100 + attempt
            seed_everything(torch, seed)
            if bank is None:
                prompt = cfg["generation"]["prompts"][index % len(cfg["generation"]["prompts"])]
                kwargs = {"texts": [prompt]}
                with torch.no_grad():
                    condition = rt["model"].wrapped.clip(batch_size=1, texts=[prompt]).detach().cpu()
                conditioning_index = index % len(cfg["generation"]["prompts"])
            else:
                prompt = None
                conditioning_index = index % len(bank)
                condition = bank[conditioning_index:conditioning_index + 1].detach().cpu()
                kwargs = {"embeddings": condition.to(rt["device"])}
            # Dependency/model failures abort instead of silently becoming fake pads.
            with torch.no_grad():
                latent = sample_latents(batch_size=1, model=rt["model"], diffusion=rt["diffusion"], model_kwargs=kwargs,
                    guidance_scale=cfg["generation"]["guidance_scale"], progress=True, clip_denoised=True,
                    use_fp16=rt["device"].type == "cuda", use_karras=True, karras_steps=cfg["generation"]["karras_steps"],
                    sigma_min=1e-3, sigma_max=160, s_churn=0)[0]
                decoded = decode_latent_mesh(rt["xm"], latent).tri_mesh()
            try:
                s, z, radius, thickness, coverage = from_shap_e(decoded.verts, decoded.faces, cfg)
                surface_hash = hashlib.sha256(np.rint(thickness * 1e6 / 20).astype(np.int32).tobytes()).hexdigest()
                if surface_hash in seen:
                    raise ValueError("Duplicate processed pad surface at 20 micrometre quantization")
                stats = save_pad(candidate, cfg, s, z, radius, thickness)
            except ValueError as e:
                with (folder / "rejected.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"index": index, "attempt": attempt, "seed": seed, "reason": str(e)}) + "\n")
                continue
            torch.save(latent.detach().cpu(), candidate / "latent.pt")
            torch.save(condition, candidate / "condition.pt")
            if cfg["generation"]["save_raw_mesh"]:
                write_stl(candidate / "raw_shap_e.stl", decoded.verts, decoded.faces)
            metadata = {"candidate_id": candidate.name, "round": args.round, "seed": seed, "attempt": attempt,
                        "source": "shap_e", "prompt": prompt, "conditioning_index": conditioning_index,
                        "projection_coverage": coverage, "surface_hash": surface_hash, "source_signature": source_signature(cfg),
                        "run_signature": signature, **stats}
            # Metadata is the completion marker; interruptions before this point
            # can be retried deterministically without committing partial results.
            build_xml(cfg, candidate, check_metadata=False)
            save_json(metadata_file, metadata)
            seen.add(surface_hash)
            successes += 1
            print("[" + str(successes) + "/" + str(count) + "] " + str(candidate), flush=True)
            del latent, decoded
            if rt["device"].type == "cuda":
                torch.cuda.empty_cache()
            break
        else:
            raise RuntimeError("Could not obtain a valid distinct pad for index " + str(index) + ". See rejected.jsonl; no substitute geometry was inserted.")
    save_json(folder / "generation_summary.json", {"requested": count, "completed": successes, "source": "shap_e"})
    print("Complete: " + str(successes) + " Shap-E pads (one mirrored pair per candidate).")


if __name__ == "__main__":
    main()
