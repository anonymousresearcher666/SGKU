import argparse
import datetime
import json
import os
import subprocess
import sys
from copy import deepcopy
from itertools import product
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
    candidates = [os.path.join(run_dir, name) for name in os.listdir(run_dir) if name.endswith("_results.json")]
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

    base_objective = objective[: -len("_final")] if objective.endswith("_final") else objective

    def score_for(ts: Dict[str, Any]) -> Optional[float]:
        comp = ts.get("composite") or {}
        if base_objective == "mrr_avg":
            return comp.get("mrr_avg")
        if base_objective == "mrr_f1":
            return comp.get("mrr_f1")
        if base_objective == "retain_mrr":
            return (ts.get("retain") or {}).get("mrr")
        if base_objective == "forget_mrr":
            return (ts.get("forget") or {}).get("mrr")
        return None

    if objective.endswith("_final"):
        ts = timesteps[-1]
        v = score_for(ts)
        return None if v is None else float(v)

    vals: List[float] = []
    for ts in timesteps:
        v = score_for(ts)
        if v is None:
            continue
        vals.append(float(v))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


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

    _deep_set(cfg, ("defaults", "unlearning_save_path"), out_dir)
    _deep_set(cfg, ("defaults", "overwrite_unlearning_dir"), True)
    _deep_set(cfg, ("defaults", "retain_eval_sample_frac"), float(retain_eval_sample_frac))
    _deep_set(cfg, ("defaults", "epoch_num"), int(epoch_num))
    if timesteps_num is not None:
        _deep_set(cfg, ("defaults", "timesteps_num"), int(timesteps_num))
    _deep_set(cfg, ("defaults", "patience"), int(patience))
    _deep_set(cfg, ("defaults", "valid_gap"), int(valid_gap))
    _deep_set(cfg, ("defaults", "stop_on_success"), bool(stop_on_success))

    for k, v in overrides.items():
        _deep_set(cfg, ("hyperparameters", k), v)

    cfg["name"] = str(run_name)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search for SDKU hyperparameters.")
    parser.add_argument("--base-config", required=True, help="Path to a base SDKU YAML config.")
    parser.add_argument("--out-root", default="checkpoint_grid_sweep", help="Root directory for sweep runs.")
    parser.add_argument(
        "--objective",
        default="mrr_avg_final",
        choices=["mrr_avg", "mrr_f1", "mrr_avg_final", "mrr_f1_final", "retain_mrr", "forget_mrr"],
        help="Which metric to maximize.",
    )
    parser.add_argument("--retain-eval-frac", type=float, default=0.05, help="Retain eval sampling fraction.")
    parser.add_argument("--epoch-num", type=int, default=40, help="Max epochs per timestep during tuning.")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (validation steps).")
    parser.add_argument("--valid-gap", type=int, default=10, help="Validate every N epochs.")
    parser.add_argument("--timesteps-num", type=int, default=1, help="Timesteps to run per trial.")
    parser.add_argument("--stop-on-success", action="store_true", help="Stop a timestep early when success criteria hit.")
    parser.add_argument("--max-runs", type=int, default=0, help="Optional cap on number of grid runs (0 = no cap).")

    # SDKU grid (keep default small; override via CLI as needed).
    parser.add_argument("--epsilon-clip", nargs="+", type=float, default=[0.05, 0.06])
    parser.add_argument("--lambda-kl", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--lambda-boundary", nargs="+", type=float, default=[1.1])
    parser.add_argument("--lambda-neg", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument("--group-weight-alpha", nargs="+", type=float, default=[0.6, 0.7, 0.8])
    parser.add_argument("--sdku-bias-scale", nargs="+", type=float, default=[0.2, 0.3])
    parser.add_argument("--sdku-bias-clip", nargs="+", type=float, default=[0.6, 0.9])
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)

    out_root = args.out_root
    if not os.path.isabs(out_root):
        out_root = os.path.join(PROJECT_ROOT, out_root)
    os.makedirs(out_root, exist_ok=True)

    sweep_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_out_dir = os.path.join(PROJECT_ROOT, "src", "main", "configs", "sweeps", "grid", sweep_id)
    os.makedirs(cfg_out_dir, exist_ok=True)

    combos = list(
        product(
            args.epsilon_clip,
            args.lambda_kl,
            args.lambda_boundary,
            args.lambda_neg,
            args.group_weight_alpha,
            args.sdku_bias_scale,
            args.sdku_bias_clip,
        )
    )
    if args.max_runs and args.max_runs > 0:
        combos = combos[: int(args.max_runs)]

    leaderboard: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for idx, (eps, lam_kl, lam_boundary, lam_neg, gw_alpha, bias_scale, bias_clip) in enumerate(combos):
        run_id = f"grid_{sweep_id}_{idx:03d}"
        run_dir = os.path.join(out_root, run_id)
        overrides = {
            "epsilon_clip": float(eps),
            "lambda_kl": float(lam_kl),
            "lambda_boundary": float(lam_boundary),
            "lambda_neg": float(lam_neg),
            "group_weight_alpha": float(gw_alpha),
            "sdku_bias_scale": float(bias_scale),
            "sdku_bias_clip": float(bias_clip),
        }
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
            overrides=overrides,
        )
        cfg_path = os.path.join(cfg_out_dir, f"{run_id}.yaml")
        _write_yaml(cfg_path, run_cfg)

        print(f"\n=== [{idx+1}/{len(combos)}] RUN {run_id} ===")
        print("config:", cfg_path)
        print("hyperparameters:", overrides)

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
            "hyperparameters": overrides,
        }
        leaderboard.append(entry)
        if score is not None and (best is None or (best.get("score") is None) or float(score) > float(best["score"])):
            best = entry
            print(f"NEW BEST: {args.objective}={float(score):.6f} ({run_id})")

        with open(os.path.join(out_root, f"grid_{sweep_id}_leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump({"sweep_id": sweep_id, "objective": args.objective, "runs": leaderboard, "best": best}, f, indent=2)

    print("\n=== GRID SEARCH COMPLETE ===")
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
