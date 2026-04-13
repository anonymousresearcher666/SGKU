import argparse
import math
import os
import shutil
import sys
import time
import datetime
from typing import Any, Dict, List, Tuple

# Enable CPU fallback for unsupported MPS ops (users can override by exporting the var explicitly).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.loading.KG import KGBaseTrainingData
from src.utilities.kge_factory import get_kge_model_class
from src.utilities.utilities import resolve_device, set_seeds


DEFAULT_DATASET = "FB15k-237-10"
DEFAULT_KGE = "transe"
DEFAULT_TIMESTEPS = 3
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "configs", f"pretrain_{DEFAULT_DATASET}_{DEFAULT_KGE}.yaml")
DEFAULT_FORGET_CONFIG = os.path.join(SCRIPT_DIR, "configs", f"forget_{DEFAULT_DATASET}_{DEFAULT_KGE}.yaml")


def load_config(config_path: str) -> Dict:
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_args(config: Dict) -> argparse.Namespace:
    args = argparse.Namespace()

    defaults = config.get("defaults", {})
    for key, value in defaults.items():
        setattr(args, key, value)

    run_cfg = config.get("run", {})
    for key, value in run_cfg.items():
        if key == "dataset":
            setattr(args, "data_name", value)
        elif key == "method":
            setattr(args, "unlearning_method", value)
        else:
            setattr(args, key, value)

    hyper_cfg = config.get("hyperparameters", {})
    for key, value in hyper_cfg.items():
        setattr(args, key, value)

    if hasattr(args, "data_path") and not os.path.isabs(args.data_path):
        args.data_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.data_path))
    if hasattr(args, "pretrain_save_path") and not os.path.isabs(args.pretrain_save_path):
        args.pretrain_save_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.pretrain_save_path))
    if hasattr(args, "log_path") and not os.path.isabs(args.log_path):
        args.log_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.log_path))
    if hasattr(args, "unlearning_save_path") and not os.path.isabs(args.unlearning_save_path):
        args.unlearning_save_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.unlearning_save_path))
    return args


def load_pretrained_model(args, kg):
    device = resolve_device(getattr(args, "device", None))
    set_seeds(getattr(args, "seed", 1234))
    model_class = get_kge_model_class(getattr(args, "kge", "transe"))
    model = model_class(args, kg)
    model.to(device)

    pretrain_base = getattr(args, "pretrain_save_path", os.path.join(PROJECT_ROOT, "checkpoint_pretrain"))
    if not os.path.isabs(pretrain_base):
        pretrain_base = os.path.join(PROJECT_ROOT, pretrain_base)

    candidate_dirs = [
        os.path.join(pretrain_base, args.data_name, args.kge),
        os.path.join(pretrain_base, args.data_name),
        pretrain_base,
    ]

    for directory in candidate_dirs:
        candidate_file = os.path.join(directory, "model_best.tar")
        if os.path.isfile(candidate_file):
            checkpoint = torch.load(candidate_file, map_location=device, weights_only=True)
            state_dict = checkpoint["state_dict"]
            model_state = model.state_dict()
            for key, value in list(state_dict.items()):
                if key in model_state and model_state[key].shape != value.shape:
                    if model_state[key].dim() == 2 and model_state[key].shape[0] >= value.shape[0]:
                        pad_rows = model_state[key].shape[0] - value.shape[0]
                        if pad_rows > 0:
                            pad = torch.zeros((pad_rows,) + value.shape[1:], device=value.device, dtype=value.dtype)
                            value = torch.cat([value, pad], dim=0)
                            state_dict[key] = value
                    else:
                        del state_dict[key]
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return model

    raise FileNotFoundError(
        f"Could not locate pretrained checkpoint for {args.data_name} ({args.kge}). "
        f"Searched: {candidate_dirs}"
    )


def compute_ranks(model, device, kg, triples: List[Tuple[int, int, int]], batch_size: int = 512) -> torch.Tensor:
    ranks = []
    total = len(triples)
    iterator = range(0, total, batch_size)
    if tqdm:
        iterator = tqdm(iterator, desc="Evaluating triples", unit="batch")

    for start in iterator:
        end = min(start + batch_size, total)
        batch = triples[start:end]
        heads = torch.tensor([h for h, _, _ in batch], device=device)
        relations = torch.tensor([r for _, r, _ in batch], device=device)
        tails = torch.tensor([t for _, _, t in batch], device=device)

        with torch.no_grad():
            scores = model.predict(heads, relations)

        target_scores = scores[torch.arange(scores.size(0), device=device), tails]
        compare = scores >= target_scores.unsqueeze(1)
        rank = compare.sum(dim=1).to(torch.int64)
        ranks.append(rank.cpu())

    if tqdm:
        iterator.close()
    return torch.cat(ranks, dim=0)


def main():
    started_at = datetime.datetime.now().isoformat()
    run_start = time.time()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    parser = argparse.ArgumentParser(description="Build model-specific forget sets based on learned knowledge")
    parser.add_argument("--config", help="Path to dataset/model YAML configuration",
                        default=DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    parser.add_argument("--forget-config", help="Optional YAML containing forget-specific parameters",
                        default=DEFAULT_FORGET_CONFIG if os.path.exists(DEFAULT_FORGET_CONFIG) else None)
    parser.add_argument("--rank-threshold", type=int, default=None,
                        help="Maximum rank to keep a triple (lower is better)")
    parser.add_argument("--keep-top", type=int, default=None,
                        help="Keep the top-N triples per timestep regardless of threshold")
    parser.add_argument("--keep-fraction", type=float, default=None,
                        help="Keep the best fraction (0-1] of triples per timestep")
    parser.add_argument("--per-step-count", type=int, default=None,
                        help="Number of triples to forget at each timestep")
    parser.add_argument("--per-step-fraction", type=float, default=None,
                        help="Fraction of triples to forget at each timestep")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size used for scoring triples")
    parser.add_argument("--suffix", default=None,
                        help="Suffix for the output timesteps directory (defaults to <kge>)")
    parser.add_argument("--input-dir", default=None,
                        help="Optional existing timesteps directory to filter (relative to dataset root allowed)")
    parser.add_argument("--output-dir", default=None,
                        help="Custom output directory for filtered timesteps (relative allowed)")
    args_cli = parser.parse_args()

    if not args_cli.config and not args_cli.forget_config:
        parser.error("Provide at least --config or --forget-config")

    config = load_config(args_cli.config) if args_cli.config else {}
    forget_cfg = load_config(args_cli.forget_config) if args_cli.forget_config else {}
    if not config and forget_cfg:
        config = forget_cfg
    if not config:
        parser.error("Configuration file missing required sections (defaults/run/...)")

    args = build_args(config)

    dataset_dir = os.path.join(args.data_path, args.data_name)

    def resolve_relative(path):
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(dataset_dir, path)

    forget_params: Dict[str, Any] = {}
    forget_params.update(config.get("forget", {}))
    if forget_cfg:
        forget_params.update(forget_cfg.get("forget", forget_cfg))

    if args_cli.rank_threshold is not None:
        forget_params["rank_threshold"] = args_cli.rank_threshold
    if args_cli.keep_top is not None:
        forget_params["keep_top"] = args_cli.keep_top
    if args_cli.keep_fraction is not None:
        forget_params["keep_fraction"] = args_cli.keep_fraction
    if args_cli.per_step_count is not None:
        forget_params["per_step_count"] = args_cli.per_step_count
    if args_cli.per_step_fraction is not None:
        forget_params["per_step_fraction"] = args_cli.per_step_fraction
    if args_cli.input_dir:
        forget_params["input_dir"] = args_cli.input_dir
    if args_cli.output_dir:
        forget_params["output_dir"] = args_cli.output_dir
    if args_cli.suffix:
        forget_params["suffix"] = args_cli.suffix

    kge_suffix = getattr(args, "kge", "").lower()

    rank_threshold = forget_params.get("rank_threshold")
    keep_top = forget_params.get("keep_top")
    keep_fraction = forget_params.get("keep_fraction")
    per_step_count = forget_params.get("per_step_count")
    per_step_fraction = forget_params.get("per_step_fraction")

    if rank_threshold is not None:
        rank_threshold = int(rank_threshold)
    if keep_top is not None:
        keep_top = int(keep_top)
    if keep_fraction is not None:
        keep_fraction = float(keep_fraction)
        if not (0.0 < keep_fraction <= 1.0):
            raise ValueError("keep_fraction must be in (0, 1].")

    if rank_threshold is None and keep_top is None and keep_fraction is None:
        rank_threshold = 2  # Equivalent to MRR > 0.5

    resolved_input_dir = resolve_relative(forget_params.get("input_dir"))
    use_existing_timesteps = resolved_input_dir is not None and os.path.isdir(resolved_input_dir)
    if use_existing_timesteps:
        input_dir = resolved_input_dir
    else:
        input_dir = None

    suffix = forget_params.get("suffix") or kge_suffix or "model"
    output_dir = resolve_relative(forget_params.get("output_dir"))
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "forget_sets", suffix, "timesteps")
    output_dir = os.path.normpath(output_dir)
    parent_dir = os.path.dirname(output_dir)
    if parent_dir and not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    kg = KGBaseTrainingData(args)
    model = load_pretrained_model(args, kg)
    device = resolve_device(getattr(args, "device", None))

    timesteps_num = int(getattr(args, "timesteps_num", DEFAULT_TIMESTEPS))

    print(f"Building model-specific forget sets for {args.data_name} ({args.kge})")
    print(
        f"Selection criteria -> rank_threshold: {rank_threshold}, keep_top: {keep_top}, "
        f"keep_fraction: {keep_fraction}, per_step_count: {per_step_count}, per_step_fraction: {per_step_fraction}"
    )
    print(f"Output directory: {output_dir}")

    metadata_steps: List[Dict[str, Any]] = []
    metadata_info: Dict[str, Any] = {
        "dataset": args.data_name,
        "kge": args.kge,
        "timesteps_num": timesteps_num,
        "criteria": {
            "rank_threshold": rank_threshold,
            "keep_top": keep_top,
            "keep_fraction": keep_fraction,
            "per_step_count": per_step_count,
            "per_step_fraction": per_step_fraction,
        },
    }

    if use_existing_timesteps:
        print(f"Input timesteps: {input_dir}")
        for timestep_idx in range(timesteps_num):
            input_file = os.path.join(input_dir, f"{timestep_idx}.txt")
            if not os.path.isfile(input_file):
                raise FileNotFoundError(f"Missing timestep file: {input_file}")

            with open(input_file, "r", encoding="utf-8") as f:
                raw_triples = [line.strip().split("\t") for line in f if line.strip()]

            triple_ids = []
            for triple in raw_triples:
                if len(triple) < 3:
                    continue
                h, r, t = triple[:3]
                if h not in kg.entity2id or r not in kg.relation2id or t not in kg.entity2id:
                    continue
                triple_ids.append((kg.entity2id[h], kg.relation2id[r], kg.entity2id[t]))

            if not triple_ids:
                print(f"Timestep {timestep_idx}: no valid triples found, skipping.")
                continue

            ranks = compute_ranks(model, device, kg, triple_ids, batch_size=args_cli.batch_size)

            mask = torch.zeros(len(triple_ids), dtype=torch.bool)
            if rank_threshold is not None:
                mask |= ranks <= rank_threshold
            if keep_top is not None and keep_top > 0:
                top = min(keep_top, len(triple_ids))
                top_idx = torch.argsort(ranks)[:top]
                mask[top_idx] = True
            if keep_fraction is not None and keep_fraction > 0:
                top = max(1, math.ceil(keep_fraction * len(triple_ids)))
                frac_idx = torch.argsort(ranks)[:top]
                mask[frac_idx] = True
            if not mask.any():
                fallback_idx = torch.argsort(ranks)[:1]
                mask[fallback_idx] = True
            kept_indices = mask.nonzero(as_tuple=True)[0]

            output_file = os.path.join(output_dir, f"{timestep_idx}.txt")
            with open(output_file, "w", encoding="utf-8") as outf:
                for idx in kept_indices.tolist():
                    h_id, r_id, t_id = triple_ids[idx]
                    outf.write(f"{kg.id2entity[h_id]}\t{kg.id2relation[r_id]}\t{kg.id2entity[t_id]}\n")

            retained = len(kept_indices)
            selected_ranks = ranks[kept_indices] if retained > 0 else None
            metadata_steps.append({
                "timestep": timestep_idx,
                "input_triples": len(triple_ids),
                "kept": retained,
                "min_rank": int(selected_ranks.min().item()) if selected_ranks is not None else None,
                "max_rank": int(selected_ranks.max().item()) if selected_ranks is not None else None,
            })
            print(
                f"Timestep {timestep_idx}: kept {retained} of {len(triple_ids)} triples "
                f"({retained/len(triple_ids):.1%})"
            )
    else:
        train_file = os.path.join(dataset_dir, "triples.txt")
        if not os.path.isfile(train_file):
            raise FileNotFoundError(f"Could not find training triples file: {train_file}")

        raw_triples = []
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    raw_triples.append(parts[:3])

        triple_ids: List[Tuple[int, int, int]] = []
        for h_str, r_str, t_str in raw_triples:
            try:
                triple_ids.append(kg.fact2id(h_str, r_str, t_str))
            except KeyError:
                continue

        if not triple_ids:
            raise RuntimeError("No valid triples loaded from training set.")

        print(f"Scoring {len(triple_ids)} triples to build sequential forget sets...")
        ranks = compute_ranks(model, device, kg, triple_ids, batch_size=args_cli.batch_size)
        order = torch.argsort(ranks)

        if per_step_count is None:
            if per_step_fraction is not None and per_step_fraction > 0:
                per_step_count = max(1, int(len(triple_ids) * per_step_fraction))
            else:
                raise ValueError("Specify per_step_count or per_step_fraction when generating timesteps from scratch.")
        else:
            per_step_count = int(per_step_count)

        for timestep_idx in range(timesteps_num):
            start = timestep_idx * per_step_count
            end = min(start + per_step_count, len(order))
            if start >= len(order):
                print(f"Timestep {timestep_idx}: no more triples available, stopping.")
                break

            selected_idx = order[start:end]
            if len(selected_idx) > 0:
                output_file = os.path.join(output_dir, f"{timestep_idx}.txt")
                with open(output_file, "w", encoding="utf-8") as outf:
                    for idx in selected_idx.tolist():
                        h_id, r_id, t_id = triple_ids[idx]
                        outf.write(f"{kg.id2entity[h_id]}\t{kg.id2relation[r_id]}\t{kg.id2entity[t_id]}\n")

                chosen_ranks = ranks[selected_idx]
                metadata_steps.append({
                    "timestep": timestep_idx,
                    "input_triples": len(triple_ids),
                    "kept": len(selected_idx),
                    "min_rank": int(chosen_ranks.min().item()),
                    "max_rank": int(chosen_ranks.max().item()),
                })
                print(
                    f"Timestep {timestep_idx}: selected {len(selected_idx)} triples "
                    f"(rank range {chosen_ranks.min().item()} - {chosen_ranks.max().item()})"
                )

    metadata_info["criteria"]["per_step_count"] = per_step_count
    metadata_info["criteria"]["per_step_fraction"] = per_step_fraction
    metadata_info["steps"] = metadata_steps
    metadata_info["source"] = "existing" if use_existing_timesteps else "generated"
    metadata_info["timesteps_dir"] = output_dir
    metadata_info["timing"] = {
        "started_at": started_at,
        "duration_sec": round(time.time() - run_start, 6),
    }
    metadata_dir = os.path.dirname(output_dir)
    metadata_path = os.path.join(metadata_dir, "metadata.yaml")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata_info, f, sort_keys=False)

    print(f"Done. Metadata saved to {metadata_path}.")


if __name__ == "__main__":
    main()
