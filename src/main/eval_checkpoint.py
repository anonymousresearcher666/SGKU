import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

# Enable CPU fallback for unsupported MPS ops (users can override by exporting the var explicitly).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.loading.KG import KGUnlearningData
from src.model.SGKU import SGKU
from src.model.SDKU import SDKU
from src.runners.tester import UnlearningTester
from src.utilities.kge_factory import get_kge_model_class
from src.utilities.utilities import resolve_device, set_seeds


def _map_run_config(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for key, value in run_cfg.items():
        if key == "dataset":
            mapped["data_name"] = value
        elif key == "method":
            mapped["unlearning_method"] = value
        else:
            mapped[key] = value
    return mapped


def load_args(config_path: str) -> argparse.Namespace:
    if not os.path.isabs(config_path):
        config_path = os.path.join(SCRIPT_DIR, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    defaults = config.get("defaults", {})
    run_cfg = config.get("run", {})
    hyper_cfg = config.get("hyperparameters", {})

    args_dict: Dict[str, Any] = {}
    args_dict.update(defaults)
    args_dict.update(_map_run_config(run_cfg))
    args_dict.update(hyper_cfg)

    args = argparse.Namespace()
    for key, value in args_dict.items():
        setattr(args, key, value)

    for attr in ("data_path", "log_path", "unlearning_save_path", "pretrain_save_path"):
        val = getattr(args, attr, None)
        if isinstance(val, str) and not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(PROJECT_ROOT, val)))

    return args


def load_baseline(baseline_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on retain/forget splits.")
    parser.add_argument("--config", required=True, help="Path to SGKU YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .tar checkpoint to evaluate.")
    parser.add_argument("--timestep", type=int, default=0, help="Timestep index to evaluate.")
    parser.add_argument("--device", default=None, help="Override device (cpu/cuda/mps).")
    parser.add_argument("--eval-sample-size", type=int, default=None, help="Optional sample size for faster eval.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional path to pretrain_unlearning_baseline.json (for deltas).",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
    cli = parser.parse_args()

    args = load_args(cli.config)
    args.begin_pretrain = False
    args.begin_unleanring = True

    if cli.eval_sample_size is not None:
        args.eval_sample_size = int(cli.eval_sample_size)

    if cli.device:
        args.device = resolve_device(cli.device)
    else:
        args.device = resolve_device(getattr(args, "device", None))

    set_seeds(int(getattr(args, "seed", 1234)))

    # Point evaluation to the requested timestep.
    args.timestep = int(cli.timestep)
    args.timestep_test = int(cli.timestep)
    args.timestep_validation = int(cli.timestep)

    # Load KG + model.
    kg = KGUnlearningData(args)
    model_class = get_kge_model_class(getattr(args, "kge", "transe"))
    if getattr(args, "unlearning_method", "SGKU") == "SDKU":
        model = SDKU(args=args, kg=kg, kge_model_class=model_class, schema_store=getattr(kg, "schema_store", None))
    else:
        model = SGKU(args=args, kg=kg, kge_model_class=model_class, schema_store=getattr(kg, "schema_store", None))
    model.to(args.device)

    ckpt = torch.load(cli.checkpoint, map_location=args.device, weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    # Be defensive: skip any mismatched tensors (e.g., when buffers changed between versions).
    if isinstance(state, dict):
        model_state = model.state_dict()
        filtered = {}
        skipped = 0
        for k, v in state.items():
            if k in model_state and hasattr(model_state[k], "shape") and hasattr(v, "shape"):
                if tuple(model_state[k].shape) != tuple(v.shape):
                    skipped += 1
                    continue
            filtered[k] = v
        if skipped:
            print(f"Warning: skipped {skipped} checkpoint tensors due to shape mismatch.")
        state = filtered
    model.load_state_dict(state, strict=False)

    tester = UnlearningTester(args, kg, model)
    forget_results, retain_results = tester.test()

    payload = {
        "checkpoint": os.path.abspath(cli.checkpoint),
        "timestep": int(cli.timestep),
        "device": str(args.device),
        "eval_sample_size": int(getattr(args, "eval_sample_size", 0) or 0),
        "retain": retain_results,
        "forget": forget_results,
    }

    # Optional delta vs pretrained baseline.
    baseline_path = cli.baseline
    if baseline_path is None:
        baseline_path = os.path.join(
            args.pretrain_save_path,
            args.data_name,
            str(args.kge).lower(),
            "pretrain_unlearning_baseline.json",
        )
    baseline = load_baseline(baseline_path)
    if baseline and isinstance(baseline.get("timesteps"), list):
        base = None
        for rec in baseline["timesteps"]:
            if int(rec.get("timestep", -1)) == int(cli.timestep):
                base = rec
                break
        if base:
            base_retain = base.get("retain", {}) if isinstance(base.get("retain"), dict) else {}
            base_forget = base.get("forget", {}) if isinstance(base.get("forget"), dict) else {}
            payload["delta_vs_pretrain"] = {
                "retain": {
                    "mrr": retain_results.get("mrr") - base_retain.get("mrr"),
                    "hits10": retain_results.get("hits10") - base_retain.get("hits10"),
                },
                "forget": {
                    "mrr": forget_results.get("mrr") - base_forget.get("mrr"),
                    "hits10": forget_results.get("hits10") - base_forget.get("hits10"),
                },
                "baseline_path": baseline_path,
            }
            print(
                "Δ vs pretrained (same split) -> "
                f"retain ΔMRR={payload['delta_vs_pretrain']['retain']['mrr']:+.4f}, "
                f"forget ΔMRR={payload['delta_vs_pretrain']['forget']['mrr']:+.4f}"
            )

    print(
        f"Retain: MRR={retain_results.get('mrr'):.4f} Hits@10={retain_results.get('hits10'):.4f} "
        f"| Forget: MRR={forget_results.get('mrr'):.4f} Hits@10={forget_results.get('hits10'):.4f}"
    )

    out_path = cli.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(cli.checkpoint))[0]
        out_path = os.path.join(os.path.dirname(cli.checkpoint), f"{stem}_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
