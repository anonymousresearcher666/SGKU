import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
from copy import deepcopy
from itertools import product

import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _find_results(run_dir):
    pattern = os.path.join(run_dir, "SGKU", "FB15k-237-10", "transe", "*_results.json")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _extract_mrr_avg(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    final = data.get("summary", {}).get("final")
    if not isinstance(final, dict):
        return None
    return final.get("composite", {}).get("mrr_avg")


def main():
    parser = argparse.ArgumentParser(description="Narrow hyperparameter sweep (epsilon vs lambda_neg).")
    parser.add_argument("--base-config", required=True, help="Base SGKU YAML to modify.")
    parser.add_argument("--out-root", default="checkpoint_narrow_sweep", help="Sweep output root.")
    parser.add_argument("--epsilon", nargs="+", type=float, default=[0.10, 0.11, 0.12], help="epsilon_clip values.")
    parser.add_argument("--lambda-neg", nargs="+", type=float, default=[0.22, 0.24, 0.26], help="lambda_neg values.")
    parser.add_argument("--retain-eval-frac", type=float, default=0.05, help="Retain eval sampling fraction.")
    parser.add_argument("--epoch-num", type=int, default=40, help="Epochs per trial.")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--valid-gap", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    base_cfg = _load_yaml(args.base_config)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_dir = os.path.join(PROJECT_ROOT, "src", "main", "configs", "sweeps", "narrow", timestamp)
    os.makedirs(cfg_dir, exist_ok=True)
    out_root = args.out_root
    if not os.path.isabs(out_root):
        out_root = os.path.join(PROJECT_ROOT, out_root)
    os.makedirs(out_root, exist_ok=True)

    combos = list(product(args.epsilon, args.lambda_neg))
    leaderboard = []

    for idx, (eps, lam_neg) in enumerate(combos):
        run_id = f"narrow_{timestamp}_{idx:02d}"
        run_dir = os.path.join(out_root, run_id)
        cfg = deepcopy(base_cfg)
        cfg["name"] = run_id
        cfg.setdefault("defaults", {})
        cfg.setdefault("hyperparameters", {})
        cfg["defaults"]["unlearning_save_path"] = run_dir
        cfg["defaults"]["retain_eval_sample_frac"] = float(args.retain_eval_frac)
        cfg["defaults"]["epoch_num"] = int(args.epoch_num)
        cfg["defaults"]["patience"] = int(args.patience)
        cfg["defaults"]["valid_gap"] = int(args.valid_gap)
        cfg["defaults"]["seed"] = args.seed
        cfg["hyperparameters"]["epsilon_clip"] = float(eps)
        cfg["hyperparameters"]["lambda_neg"] = float(lam_neg)
        cfg["hyperparameters"]["lambda_boundary"] = float(cfg["hyperparameters"].get("lambda_boundary", 0.4))
        cfg["hyperparameters"]["lambda_kl"] = float(cfg["hyperparameters"].get("lambda_kl", 0.0))
        cfg["hyperparameters"]["group_weight_alpha"] = float(cfg["hyperparameters"].get("group_weight_alpha", 0.8))
        cfg_path = os.path.join(cfg_dir, f"{run_id}.yaml")
        _write_yaml(cfg_path, cfg)

        print(f"\n=== RUN {run_id} ({idx+1}/{len(combos)}) eps={eps} lam_neg={lam_neg} ===")
        print("config:", cfg_path)
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "src", "main", "main.py"), "--config", cfg_path]
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("LIBOMP_USE_SHM", "0")
        env.setdefault("KMP_AFFINITY", "disabled")
        env.setdefault("OMP_WAIT_POLICY", "passive")
        env.setdefault("KMP_BLOCKTIME", "1")
        ret = subprocess.call(cmd, cwd=PROJECT_ROOT, env=env)

        results_path = _find_results(run_dir)
        mrr_avg = _extract_mrr_avg(results_path) if results_path else None
        leaderboard.append({
            "run_id": run_id,
            "config": cfg_path,
            "run_dir": run_dir,
            "epsilon_clip": eps,
            "lambda_neg": lam_neg,
            "return_code": int(ret),
            "results": results_path,
            "mrr_avg": mrr_avg,
        })

        with open(os.path.join(out_root, f"narrow_{timestamp}_leaderboard.json"), "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": timestamp,
                "args": vars(args),
                "runs": leaderboard
            }, f, indent=2)

    print("\n=== NARROW SWEEP COMPLETE ===")
    best = max([r for r in leaderboard if r["mrr_avg"] is not None], key=lambda r: r["mrr_avg"], default=None)
    if best:
        print(f"best run: {best['run_id']} (ε={best['epsilon_clip']}, λ_neg={best['lambda_neg']}) mrr_avg={best['mrr_avg']}")
        print("config:", best["config"])
        print("results:", best["results"])
    else:
        print("No scored runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
