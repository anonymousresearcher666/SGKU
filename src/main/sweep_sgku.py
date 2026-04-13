import argparse
import datetime
import json
import os
import random
import subprocess
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _deep_set(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Any = d
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _find_results_json(run_dir: str) -> Optional[str]:
    if not os.path.isdir(run_dir):
        return None
    candidates = []
    for name in os.listdir(run_dir):
        if name.endswith("_results.json"):
            candidates.append(os.path.join(run_dir, name))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _objective_from_results(results_path: str, objective: str) -> Optional[float]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    timesteps = data.get("timesteps") or []
    if not isinstance(timesteps, list) or not timesteps:
        return None

    def score_for(ts: Dict[str, Any]) -> Optional[float]:
        comp = ts.get("composite") or {}
        if objective == "mrr_avg":
            return comp.get("mrr_avg")
        if objective == "mrr_f1":
            return comp.get("mrr_f1")
        if objective == "retain_mrr":
            return (ts.get("retain") or {}).get("mrr")
        if objective == "forget_mrr":
            return (ts.get("forget") or {}).get("mrr")
        return None

    vals: List[float] = []
    if objective.endswith("_final"):
        ts = timesteps[-1]
        v = score_for(ts)
        return None if v is None else float(v)

    for ts in timesteps:
        v = score_for(ts)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _random_grid(rng: random.Random) -> Dict[str, Any]:
    # Wider, high-leverage search space (still safe for long unattended sweeps).
    epsilon_clip = rng.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
    lambda_kl = rng.choice([0.0, 0.0025, 0.005, 0.01, 0.02, 0.05])
    lambda_boundary = rng.choice([0.0, 0.2, 0.4, 0.6, 0.9])
    lambda_neg = rng.choice([0.0, 0.02, 0.05, 0.1, 0.2])
    temperature_min = rng.choice([0.01, 0.02, 0.05, 0.1])
    projection_every_k = rng.choice([10, 20, 50, 100])
    projection_lambda = rng.choice([0.2, 0.4, 0.6, 0.8])
    projection_top_n = rng.choice([5, 10, 20])
    reference_refresh_m = rng.choice([5, 10, 20])
    group_weight_alpha = rng.choice([0.2, 0.5, 0.8])
    return {
        "epsilon_clip": float(epsilon_clip),
        "lambda_kl": float(lambda_kl),
        "lambda_boundary": float(lambda_boundary),
        "lambda_neg": float(lambda_neg),
        "temperature_min": float(temperature_min),
        "projection_every_k": int(projection_every_k),
        "projection_lambda": float(projection_lambda),
        "projection_top_n": int(projection_top_n),
        "reference_refresh_m": int(reference_refresh_m),
        "group_weight_alpha": float(group_weight_alpha),
    }


def _build_run_config(
    base_cfg: Dict[str, Any],
    *,
    run_name: str,
    out_dir: str,
    retain_eval_sample_frac: float,
    epoch_num: int,
    patience: int,
    valid_gap: int,
    timesteps_num: Optional[int],
    stop_on_success: bool,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("defaults", {})
    cfg.setdefault("hyperparameters", {})

    # Save path must be unique per run to preserve checkpoints+artifacts.
    _deep_set(cfg, ("defaults", "unlearning_save_path"), out_dir)
    _deep_set(cfg, ("defaults", "overwrite_unlearning_dir"), True)

    # Fast tuning loop controls.
    _deep_set(cfg, ("defaults", "retain_eval_sample_frac"), float(retain_eval_sample_frac))
    _deep_set(cfg, ("defaults", "epoch_num"), int(epoch_num))
    if timesteps_num is not None:
        _deep_set(cfg, ("defaults", "timesteps_num"), int(timesteps_num))
    _deep_set(cfg, ("defaults", "patience"), int(patience))
    _deep_set(cfg, ("defaults", "valid_gap"), int(valid_gap))
    _deep_set(cfg, ("defaults", "stop_on_success"), bool(stop_on_success))

    # Apply SGKU hyperparameter overrides.
    for k, v in overrides.items():
        _deep_set(cfg, ("hyperparameters", k), v)

    # Name is used for logging and for results filenames.
    cfg["name"] = str(run_name)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Unattended SGKU hyperparameter sweep (saves best configs/checkpoints).")
    parser.add_argument("--base-config", required=True, help="Path to a base SGKU YAML config.")
    parser.add_argument("--out-root", default="checkpoint_sweeps", help="Root directory for sweep runs.")
    parser.add_argument("--runs", type=int, default=12, help="Number of runs to try.")
    parser.add_argument("--objective", default="mrr_avg_final", choices=["mrr_avg", "mrr_f1", "mrr_avg_final", "mrr_f1_final", "retain_mrr", "forget_mrr"], help="Which metric to maximize.")
    parser.add_argument("--retain-eval-frac", type=float, default=0.05, help="Fraction of retain eval triples to use for tuning (0 disables sampling).")
    parser.add_argument("--epoch-num", type=int, default=40, help="Max epochs per timestep during tuning.")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (in validation steps).")
    parser.add_argument("--valid-gap", type=int, default=10, help="Validate every N epochs.")
    parser.add_argument("--timesteps-num", type=int, default=1, help="How many timesteps to run per trial (1 is best for fast hyperparameter search).")
    parser.add_argument("--include-base", action="store_true", help="Include the base config (no hyperparameter overrides) as the first trial.")
    parser.add_argument("--stop-on-success", action="store_true", help="Stop a timestep early once success criteria are met.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)
    out_root = args.out_root
    if not os.path.isabs(out_root):
        out_root = os.path.join(PROJECT_ROOT, out_root)
    os.makedirs(out_root, exist_ok=True)

    sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_out_dir = os.path.join(PROJECT_ROOT, "src", "main", "configs", "sweeps", sweep_id)
    os.makedirs(cfg_out_dir, exist_ok=True)

    rng = random.Random(int(args.seed))
    leaderboard: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    total_runs = int(args.runs)
    trial_specs: List[Dict[str, Any]] = []
    if bool(args.include_base):
        trial_specs.append({"hp": {}, "label": "base"})
    while len(trial_specs) < total_runs:
        trial_specs.append({"hp": _random_grid(rng), "label": "sampled"})

    for idx, spec in enumerate(trial_specs[:total_runs]):
        hp = spec["hp"]
        run_id = f"sweep_{sweep_id}_{idx:03d}"
        run_dir = os.path.join(out_root, run_id)
        run_cfg = _build_run_config(
            base_cfg,
            run_name=run_id,
            out_dir=run_dir,
            retain_eval_sample_frac=float(args.retain_eval_frac),
            epoch_num=int(args.epoch_num),
            patience=int(args.patience),
            valid_gap=int(args.valid_gap),
            timesteps_num=int(args.timesteps_num) if args.timesteps_num else None,
            stop_on_success=bool(args.stop_on_success),
            overrides=hp,
        )
        cfg_path = os.path.join(cfg_out_dir, f"{run_id}.yaml")
        _write_yaml(cfg_path, run_cfg)

        print(f"\n=== [{idx+1}/{total_runs}] RUN {run_id} ({spec['label']}) ===")
        print(f"out_dir: {run_dir}")
        print(f"hyperparameters: {hp}")

        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "main", "main.py"), "--config", cfg_path]
        ret = subprocess.call(cmd, cwd=PROJECT_ROOT)

        results_path = _find_results_json(run_dir)
        score = None if results_path is None else _objective_from_results(results_path, args.objective)
        entry = {
            "run_id": run_id,
            "return_code": int(ret),
            "config": cfg_path,
            "run_dir": run_dir,
            "results": results_path,
            "objective": args.objective,
            "score": score,
            "hyperparameters": hp,
        }
        leaderboard.append(entry)
        if score is not None and (best is None or (best.get("score") is None) or float(score) > float(best["score"])):
            best = entry
            print(f"NEW BEST: {args.objective}={float(score):.6f} ({run_id})")

        # Persist sweep progress after each run.
        with open(os.path.join(out_root, f"sweep_{sweep_id}_leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump({"sweep_id": sweep_id, "objective": args.objective, "runs": leaderboard, "best": best}, f, indent=2)

    print("\n=== SWEEP COMPLETE ===")
    if best:
        print(f"best_run: {best['run_id']}  {args.objective}={best['score']}")
        print(f"best_config: {best['config']}")
        print(f"best_run_dir: {best['run_dir']}")
        print(f"best_results: {best['results']}")
    else:
        print("No successful runs produced a score; check logs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
