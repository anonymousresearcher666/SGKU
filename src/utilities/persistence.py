import datetime
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


EXCLUDED_ARG_KEYS = {
    "logger",
    "kg",
    "model",
    "optimizer",
    "train_processor",
    "valid_processor",
    "run_forget_valid_processor",
    "retain_valid_processor",
    "forget_dataset",
    "retain_dataset",
    "boundary_dataset",
    "forget_data_loader",
    "retain_data_loader",
    "boundary_data_loader",
    "schema_store",
    "entity_types",
    "global_pattern_cache",
}


def sanitize_component(value: Any) -> str:
    """Return a filesystem-friendly representation of a value."""
    text = str(value)
    sanitized = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text)
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def to_serializable(obj: Any) -> Any:
    """Convert complex objects (torch, numpy, etc.) into JSON-serializable structures."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, torch.device):
        return obj.type
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    return str(obj)


def sanitize_args_for_json(args: Any, extra_excluded: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Filter argparse.Namespace (or similar) to JSON-safe values."""
    excluded = set(EXCLUDED_ARG_KEYS)
    if extra_excluded:
        excluded.update(extra_excluded)

    sanitized: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in excluded or key.startswith("__"):
            continue
        sanitized[key] = to_serializable(value)
    return sanitized


def build_run_base_name(experiment_name: str, parameters: Dict[str, Any]) -> str:
    """Construct a readable file stem from experiment name and parameter map."""
    slug = sanitize_component(experiment_name) or "run"
    if parameters:
        parts: List[str] = []
        for key in sorted(parameters.keys()):
            part = f"{sanitize_component(key)}-{sanitize_component(parameters[key])}"
            parts.append(part)
        slug = f"{slug}__{'__'.join(parts)}"
    return slug[:180]


def determine_unique_paths(directory: str, base_name: str) -> Tuple[str, str]:
    """Return unique hyperparameters/metrics file paths by appending an index when needed."""
    idx = 0
    while True:
        suffix = "" if idx == 0 else f"_{idx}"
        hyper_path = os.path.join(directory, f"{base_name}_hyperparameters{suffix}.json")
        metrics_path = os.path.join(directory, f"{base_name}_metrics{suffix}.json")
        if not os.path.exists(hyper_path) and not os.path.exists(metrics_path):
            return hyper_path, metrics_path
        idx += 1


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _extract_kge_from_hyper(metrics_path: str) -> Optional[str]:
    hyper_path = metrics_path.replace("_metrics.json", "_hyperparameters.json")
    if not os.path.isfile(hyper_path):
        return None
    hyper = _read_json(hyper_path) or {}
    args = hyper.get("args")
    if isinstance(args, dict):
        kge = args.get("kge")
        return str(kge).lower() if kge is not None else None
    return None


def _timestamp_key(payload: Optional[Dict[str, Any]], path: str) -> Tuple[int, datetime.datetime]:
    ts = None
    if isinstance(payload, dict):
        ts = payload.get("timestamp")
    if isinstance(ts, str):
        try:
            return (1, datetime.datetime.fromisoformat(ts))
        except ValueError:
            pass
    try:
        return (0, datetime.datetime.fromtimestamp(os.path.getmtime(path)))
    except OSError:
        return (0, datetime.datetime.fromtimestamp(0))


def _load_retrain_reference(root_dir: str, dataset: str, kge: Optional[str]) -> Dict[str, Any]:
    base_dir = os.path.join(root_dir, "results", dataset, "retrain")
    if not os.path.isdir(base_dir):
        return {
            "available": False,
            "reason": f"No retrain results directory at {base_dir}",
        }
    metrics_files = glob.glob(os.path.join(base_dir, "*_metrics.json"))
    if not metrics_files:
        return {
            "available": False,
            "reason": f"No retrain metrics found in {base_dir}",
        }
    kge_norm = str(kge).lower() if kge is not None else None
    candidates: List[Tuple[Tuple[int, datetime.datetime], str, Optional[float], Optional[str]]] = []
    for metrics_path in metrics_files:
        payload = _read_json(metrics_path) or {}
        run_kge = _extract_kge_from_hyper(metrics_path)
        if kge_norm and run_kge and run_kge != kge_norm:
            continue
        training_time = payload.get("total_training_time", payload.get("training_time"))
        candidates.append((_timestamp_key(payload, metrics_path), metrics_path, _safe_float(training_time), run_kge))
    if not candidates:
        for metrics_path in metrics_files:
            payload = _read_json(metrics_path) or {}
            training_time = payload.get("total_training_time", payload.get("training_time"))
            candidates.append((_timestamp_key(payload, metrics_path), metrics_path, _safe_float(training_time), None))
    candidates.sort(key=lambda item: item[0])
    _, metrics_path, training_time, run_kge = candidates[-1]
    payload = _read_json(metrics_path) or {}
    return {
        "available": True,
        "path": metrics_path,
        "timestamp": payload.get("timestamp"),
        "training_time": training_time,
        "kge": run_kge,
        "source": "retrain",
    }


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _composite(retain_metric: Any, forget_metric: Any) -> Optional[float]:
    """Composite score for bounded metrics in [0, 1] where forgetting prefers smaller values."""
    retain_val = _safe_float(retain_metric)
    forget_val = _safe_float(forget_metric)
    if retain_val is None or forget_val is None:
        return None
    return round((retain_val + (1.0 - forget_val)) / 2.0, 6)


def _harmonic(retain_metric: Any, forget_metric: Any) -> Optional[float]:
    """Harmonic mean between retain score and forget-success (1 - forget_metric)."""
    retain_val = _safe_float(retain_metric)
    forget_val = _safe_float(forget_metric)
    if retain_val is None or forget_val is None:
        return None
    forget_success = 1.0 - forget_val
    denom = retain_val + forget_success
    if denom <= 0:
        return 0.0
    return round((2.0 * retain_val * forget_success) / denom, 6)


def _extract_standard_retain_metrics(retain_results: Dict[str, Any]) -> Dict[str, Any]:
    mean_rank = retain_results.get("mean_rank", retain_results.get("mr"))
    payload = {
        "mrr": retain_results.get("mrr"),
        "mar": mean_rank,
        "hits1": retain_results.get("hits1"),
        "hits10": retain_results.get("hits10"),
    }
    return {k: to_serializable(v) for k, v in payload.items() if v is not None}


def _extract_standard_forget_metrics(forget_results: Dict[str, Any]) -> Dict[str, Any]:
    pos_metrics = forget_results.get("pos_metrics") if isinstance(forget_results.get("pos_metrics"), dict) else {}
    mean_rank = pos_metrics.get("mr", forget_results.get("mean_rank"))
    payload = {
        "mrr": forget_results.get("mrr"),
        "mar": mean_rank,
        "hits1": forget_results.get("hits1"),
        "hits10": forget_results.get("hits10"),
    }
    return {k: to_serializable(v) for k, v in payload.items() if v is not None}


def _summarize_unlearning_timesteps(
    timestep_records: Sequence[Dict[str, Any]],
    pretrain_baseline: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    summarized: List[Dict[str, Any]] = []
    for record in timestep_records or []:
        timestep = int(record.get("timestep", 0))
        retain_raw = record.get("retain") if isinstance(record.get("retain"), dict) else {}
        forget_raw = record.get("forget") if isinstance(record.get("forget"), dict) else {}

        retain = _extract_standard_retain_metrics(retain_raw)
        forget = _extract_standard_forget_metrics(forget_raw)

        composite = {
            "mrr_avg": _composite(retain.get("mrr"), forget.get("mrr")),
            "mrr_f1": _harmonic(retain.get("mrr"), forget.get("mrr")),
            "hits1_avg": _composite(retain.get("hits1"), forget.get("hits1")),
            "hits10_avg": _composite(retain.get("hits10"), forget.get("hits10")),
        }
        composite = {k: v for k, v in composite.items() if v is not None}

        # Paper metrics (MRR-based):
        # MRR_Avg = (MRR_r + (1 - MRR_f)) / 2, MRR_F1 = 2*MRR_r*(1-MRR_f)/(MRR_r + (1-MRR_f)).
        paper_metrics_mrr = None
        try:
            mrr_r = _safe_float(retain.get("mrr"))
            mrr_f = _safe_float(forget.get("mrr"))
            if mrr_r is not None and mrr_f is not None:
                one_minus_mrr_f = 1.0 - float(mrr_f)
                mrr_avg = (float(mrr_r) + one_minus_mrr_f) / 2.0
                denom = float(mrr_r) + one_minus_mrr_f
                mrr_f1 = (2.0 * float(mrr_r) * one_minus_mrr_f / denom) if denom > 0 else 0.0
                paper_metrics_mrr = {
                    "MRR_r": round(float(mrr_r), 6),
                    "MRR_f": round(float(mrr_f), 6),
                    "MRR_Avg": round(float(mrr_avg), 6),
                    "MRR_F1": round(float(mrr_f1), 6),
                }
        except Exception:
            paper_metrics_mrr = None

        deltas: Dict[str, Any] = {}
        if isinstance(pretrain_baseline, dict) and timestep in pretrain_baseline:
            base = pretrain_baseline.get(timestep) or {}
            base_retain = base.get("retain") if isinstance(base.get("retain"), dict) else {}
            base_forget = base.get("forget") if isinstance(base.get("forget"), dict) else {}

            base_retain_metrics = _extract_standard_retain_metrics(base_retain)
            base_forget_metrics = _extract_standard_forget_metrics(base_forget)

            def delta(cur: Any, basev: Any) -> Optional[float]:
                cur_f = _safe_float(cur)
                base_f = _safe_float(basev)
                if cur_f is None or base_f is None:
                    return None
                return round(cur_f - base_f, 6)

            deltas = {
                "retain": {
                    "mrr": delta(retain.get("mrr"), base_retain_metrics.get("mrr")),
                    "hits1": delta(retain.get("hits1"), base_retain_metrics.get("hits1")),
                    "hits10": delta(retain.get("hits10"), base_retain_metrics.get("hits10")),
                },
                "forget": {
                    "mrr": delta(forget.get("mrr"), base_forget_metrics.get("mrr")),
                    "hits1": delta(forget.get("hits1"), base_forget_metrics.get("hits1")),
                    "hits10": delta(forget.get("hits10"), base_forget_metrics.get("hits10")),
                },
            }
            deltas["retain"] = {k: v for k, v in deltas["retain"].items() if v is not None}
            deltas["forget"] = {k: v for k, v in deltas["forget"].items() if v is not None}

            deltas["composite"] = {
                "mrr_avg": delta(composite.get("mrr_avg"), _composite(base_retain_metrics.get("mrr"), base_forget_metrics.get("mrr"))),
                "mrr_f1": delta(composite.get("mrr_f1"), _harmonic(base_retain_metrics.get("mrr"), base_forget_metrics.get("mrr"))),
            }
            deltas["composite"] = {k: v for k, v in deltas["composite"].items() if v is not None}

        summarized.append(
            {
                "timestep": timestep,
                "train_time_s": to_serializable(record.get("train_time")),
                "retain": retain,
                "forget": forget,
                "audit": to_serializable(record.get("audit")) if record.get("audit") is not None else None,
                "composite": to_serializable(composite),
                "paper_metrics_mrr": to_serializable(paper_metrics_mrr),
                "delta_vs_pretrain": to_serializable(deltas) if deltas else None,
            }
        )
    return summarized


def _build_experiment_verdict(
    summarized_timesteps: Sequence[Dict[str, Any]],
    *,
    retain_drop_max_frac: float = 0.05,
    forget_improve_min_frac: float = 0.10,
    metric_key: str = "mrr",
) -> Dict[str, Any]:
    """Create a concise human-readable verdict for an experiment."""
    if not summarized_timesteps:
        return {
            "successful": None,
            "score_0_100": None,
            "comment": "No timesteps recorded; unable to evaluate experiment.",
            "criteria": {
                "metric": metric_key,
                "retain_drop_max_frac": retain_drop_max_frac,
                "forget_improve_min_frac": forget_improve_min_frac,
            },
        }

    final = summarized_timesteps[-1]
    retain = final.get("retain") if isinstance(final.get("retain"), dict) else {}
    forget = final.get("forget") if isinstance(final.get("forget"), dict) else {}
    composite = final.get("composite") if isinstance(final.get("composite"), dict) else {}
    delta = final.get("delta_vs_pretrain") if isinstance(final.get("delta_vs_pretrain"), dict) else None

    cur_retain = _safe_float(retain.get(metric_key))
    cur_forget = _safe_float(forget.get(metric_key))
    cur_comp = _safe_float(composite.get(f"{metric_key}_avg")) if isinstance(composite, dict) else None

    if not delta:
        return {
            "successful": None,
            "score_0_100": None,
            "comment": (
                "No pretrained baseline on retain/forget splits available; "
                f"reporting absolute metrics only (retain {metric_key}={cur_retain}, forget {metric_key}={cur_forget}, "
                f"composite {metric_key}_avg={cur_comp})."
            ),
            "criteria": {
                "metric": metric_key,
                "retain_drop_max_frac": retain_drop_max_frac,
                "forget_improve_min_frac": forget_improve_min_frac,
            },
        }

    delta_retain = delta.get("retain") if isinstance(delta.get("retain"), dict) else {}
    delta_forget = delta.get("forget") if isinstance(delta.get("forget"), dict) else {}
    delta_cur_retain = _safe_float(delta_retain.get(metric_key))
    delta_cur_forget = _safe_float(delta_forget.get(metric_key))

    # Reconstruct baseline values (delta = current - baseline).
    base_retain = (cur_retain - delta_cur_retain) if (cur_retain is not None and delta_cur_retain is not None) else None
    base_forget = (cur_forget - delta_cur_forget) if (cur_forget is not None and delta_cur_forget is not None) else None

    eps = 1e-12
    retain_drop = None
    retain_drop_frac = None
    if base_retain is not None and cur_retain is not None:
        retain_drop = max(0.0, base_retain - cur_retain)
        retain_drop_frac = retain_drop / max(abs(base_retain), eps)

    forget_improve = None
    forget_improve_frac = None
    if base_forget is not None and cur_forget is not None:
        forget_improve = max(0.0, base_forget - cur_forget)
        forget_improve_frac = forget_improve / max(abs(base_forget), eps)

    # Scores clamp to [0, 1] then mapped to [0, 100].
    retention_score = None
    if retain_drop_frac is not None:
        retention_score = max(0.0, min(1.0, 1.0 - retain_drop_frac))
    forgetting_score = None
    if forget_improve_frac is not None:
        forgetting_score = max(0.0, min(1.0, forget_improve_frac))

    score = None
    if retention_score is not None and forgetting_score is not None:
        score = round(100.0 * (0.5 * retention_score + 0.5 * forgetting_score), 2)

    successful = None
    if retain_drop_frac is not None and forget_improve_frac is not None:
        successful = (retain_drop_frac <= retain_drop_max_frac) and (forget_improve_frac >= forget_improve_min_frac)

    if base_retain is None or base_forget is None:
        comment = (
            "Pretrained baseline deltas present but could not reconstruct baseline values for the chosen metric; "
            "check that the metric exists in both retain/forget results."
        )
    else:
        comment = (
            f"Retain preservation: {metric_key} {cur_retain:.6f} vs pretrained {base_retain:.6f} "
            f"(drop {retain_drop_frac * 100.0:.2f}%). "
            f"Forgetting: {metric_key} {cur_forget:.6f} vs pretrained {base_forget:.6f} "
            f"(improved {forget_improve_frac * 100.0:.2f}%). "
            f"Composite {metric_key}_avg={cur_comp:.6f}."
        )
        if successful is True:
            comment = "SUCCESS: " + comment
        elif successful is False:
            failures = []
            if retain_drop_frac is not None and retain_drop_frac > retain_drop_max_frac:
                failures.append(f"retain drop > {retain_drop_max_frac * 100:.1f}%")
            if forget_improve_frac is not None and forget_improve_frac < forget_improve_min_frac:
                failures.append(f"forget improve < {forget_improve_min_frac * 100:.1f}%")
            comment = "NOT SUCCESSFUL (" + ", ".join(failures) + "): " + comment

    return {
        "successful": successful,
        "score_0_100": score,
        "comment": comment,
        "criteria": {
            "metric": metric_key,
            "retain_drop_max_frac": retain_drop_max_frac,
            "forget_improve_min_frac": forget_improve_min_frac,
        },
        "details": {
            "baseline": {"retain": base_retain, "forget": base_forget},
            "current": {"retain": cur_retain, "forget": cur_forget},
            "fractions": {"retain_drop_frac": retain_drop_frac, "forget_improve_frac": forget_improve_frac},
        },
    }


def save_unlearning_results(
    args: Any,
    experiment_name: str,
    parameters: Dict[str, Any],
    training_times: Sequence[float],
    timestep_records: Sequence[Dict[str, Any]],
    report_results: Any,
    root_dir: str,
    run_wall_time: Optional[float] = None,
) -> None:
    """Persist SGKU experiment artefacts to disk."""
    dataset = getattr(args, "data_name", "unknown_dataset")
    method = getattr(args, "unlearning_method", "unknown_method")
    base_dir = os.path.join(root_dir, "results", dataset, method)
    os.makedirs(base_dir, exist_ok=True)

    base_name = build_run_base_name(experiment_name, parameters)
    hyper_path, metrics_path = determine_unique_paths(base_dir, base_name)

    timestamp = datetime.datetime.now().isoformat()
    parameters_serializable = {k: to_serializable(v) for k, v in parameters.items()}
    training_times_serialized = [round(float(t), 6) for t in training_times]

    serialized_timesteps: List[Dict[str, Any]] = []
    total_eval_time = 0.0
    for record in timestep_records:
        forget_res = to_serializable(record.get("forget", {}))
        retain_res = to_serializable(record.get("retain", {}))
        forget_dur = forget_res.get("duration", 0.0) if isinstance(forget_res, dict) else 0.0
        retain_dur = retain_res.get("duration", 0.0) if isinstance(retain_res, dict) else 0.0
        try:
            total_eval_time += float(forget_dur or 0.0) + float(retain_dur or 0.0)
        except (TypeError, ValueError):
            pass

        serialized_timesteps.append({
            "timestep": int(record.get("timestep", 0)),
            "train_time": round(float(record.get("train_time", 0.0)), 6),
            "forget": forget_res,
            "retain": retain_res,
            "audit": to_serializable(record.get("audit")) if record.get("audit") is not None else None,
        })

    hyper_payload = {
        "dataset": dataset,
        "method": method,
        "experiment": experiment_name,
        "parameters": parameters_serializable,
        "args": sanitize_args_for_json(args),
        "timestamp": timestamp,
        "run_identifier": os.path.splitext(os.path.basename(metrics_path))[0],
    }

    retrain_reference: Optional[Dict[str, Any]] = None
    if method == "retrain":
        retrain_reference = {
            "available": True,
            "path": metrics_path,
            "timestamp": timestamp,
            "training_time": round(sum(training_times_serialized), 6),
            "kge": str(getattr(args, "kge", "")).lower() or None,
            "source": "self",
        }
    else:
        retrain_reference = _load_retrain_reference(root_dir, dataset, getattr(args, "kge", None))

    metrics_payload = {
        "dataset": dataset,
        "method": method,
        "experiment": experiment_name,
        "parameters": parameters_serializable,
        "training_times": training_times_serialized,
        "total_training_time": round(sum(training_times_serialized), 6),
        "total_evaluation_time": round(total_eval_time, 6),
        "total_wall_time": None if run_wall_time is None else round(float(run_wall_time), 6),
        "timesteps": serialized_timesteps,
        "report_table": report_results.get_string() if hasattr(report_results, "get_string") else str(report_results),
        "timestamp": timestamp,
        "hyperparameters_file": os.path.basename(hyper_path),
        "run_identifier": hyper_payload["run_identifier"],
        "retrain_reference": retrain_reference,
    }

    with open(hyper_path, "w", encoding="utf-8") as hf:
        json.dump(hyper_payload, hf, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_payload, mf, indent=2)

    # Additional "at-a-glance" results file for each experiment.
    results_path = metrics_path.replace("_metrics", "_results")
    pretrain_baseline = getattr(args, "pretrain_unlearning_baseline", None)
    summarized_timesteps = _summarize_unlearning_timesteps(
        timestep_records=timestep_records,
        pretrain_baseline=pretrain_baseline if isinstance(pretrain_baseline, dict) else None,
    )
    final = summarized_timesteps[-1] if summarized_timesteps else None
    verdict = _build_experiment_verdict(
        summarized_timesteps,
        retain_drop_max_frac=float(getattr(args, "success_retain_drop_max_frac", 0.05)),
        forget_improve_min_frac=float(getattr(args, "success_forget_improve_min_frac", 0.10)),
        metric_key=str(getattr(args, "success_metric", "mrr")),
    )

    def mean_of(key_path: Tuple[str, ...]) -> Optional[float]:
        vals: List[float] = []
        for ts in summarized_timesteps:
            cur: Any = ts
            for k in key_path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(k)
            fval = _safe_float(cur)
            if fval is not None:
                vals.append(fval)
        if not vals:
            return None
        return round(sum(vals) / len(vals), 6)

    results_payload = {
        "dataset": dataset,
        "method": method,
        "experiment": experiment_name,
        "parameters": parameters_serializable,
        "timestamp": timestamp,
        "run_identifier": hyper_payload["run_identifier"],
        "summary": {
            "final": final,
            "verdict": to_serializable(verdict),
            "retrain_reference": retrain_reference,
            "paper_metrics_mrr": (
                None
                if not isinstance(final, dict)
                else to_serializable(
                    {
                        "MRR_f": _safe_float(((final.get("forget") or {}).get("mrr"))),
                        "MRR_r": _safe_float(((final.get("retain") or {}).get("mrr"))),
                        "MRR_Avg": _safe_float(((final.get("composite") or {}).get("mrr_avg"))),
                        "MRR_F1": _safe_float(((final.get("composite") or {}).get("mrr_f1"))),
                        "definition": {
                            "MRR_f": "MRR on accumulated forget split (lower is better)",
                            "MRR_r": "MRR on retain split (higher is better)",
                            "MRR_Avg": "(MRR_r + (1 - MRR_f)) / 2",
                            "MRR_F1": "2*MRR_r*(1-MRR_f)/(MRR_r + (1-MRR_f))",
                        },
                    }
                )
            ),
            "mean_over_timesteps": {
                "retain_mrr": mean_of(("retain", "mrr")),
                "retain_hits10": mean_of(("retain", "hits10")),
                "forget_mrr": mean_of(("forget", "mrr")),
                "forget_hits10": mean_of(("forget", "hits10")),
                "composite_mrr_avg": mean_of(("composite", "mrr_avg")),
                "composite_mrr_f1": mean_of(("composite", "mrr_f1")),
            },
            "composite_definition": {
                "avg": "(retain_metric + (1 - forget_metric)) / 2 for bounded metrics",
                "f1": "harmonic_mean(retain_metric, 1 - forget_metric) for bounded metrics",
            },
        },
        "timesteps": summarized_timesteps,
    }
    with open(results_path, "w", encoding="utf-8") as rf:
        json.dump(results_payload, rf, indent=2)

    # Also persist run artifacts next to checkpoints (user-facing convenience).
    checkpoint_dir = getattr(args, "unlearning_save_path", None)
    config_path = getattr(args, "config_path", None)
    if isinstance(checkpoint_dir, str) and checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            run_id = str(hyper_payload.get("run_identifier") or "run")
            base_id = run_id[:-7] if run_id.endswith("_metrics") else run_id
            checkpoint_results_path = os.path.join(checkpoint_dir, f"{base_id}_results.json")
            with open(checkpoint_results_path, "w", encoding="utf-8") as cf:
                json.dump(results_payload, cf, indent=2)
            if isinstance(config_path, str) and os.path.isfile(config_path):
                checkpoint_config_path = os.path.join(checkpoint_dir, f"{base_id}_config.yaml")
                with open(config_path, "r", encoding="utf-8") as srcf:
                    config_text = srcf.read()
                with open(checkpoint_config_path, "w", encoding="utf-8") as dstf:
                    dstf.write(config_text)
        except Exception:
            # Never fail training runs due to persistence issues.
            pass


def save_pretrain_results(
    args: Any,
    training_time: float,
    best_validation: float,
    test_results: Dict[str, Any],
    log_path: str,
    root_dir: str,
    run_wall_time: Optional[float] = None,
    test_duration: Optional[float] = None,
) -> None:
    """Persist pretraining artefacts to disk."""
    dataset = getattr(args, "data_name", "unknown_dataset")
    model = getattr(args, "kge", "unknown_model")
    base_dir = os.path.join(root_dir, "results", dataset, "pretrain")
    os.makedirs(base_dir, exist_ok=True)

    parameters = {
        "kge": model,
        "epoch_num": getattr(args, "epoch_num", None),
        "batch_size": getattr(args, "batch_size", None),
        "lr": getattr(args, "lr", None),
    }
    base_name = build_run_base_name("pretrain", parameters)
    hyper_path, metrics_path = determine_unique_paths(base_dir, base_name)

    timestamp = datetime.datetime.now().isoformat()

    hyper_payload = {
        "dataset": dataset,
        "method": "pretrain",
        "kge": model,
        "parameters": {k: to_serializable(v) for k, v in parameters.items()},
        "args": sanitize_args_for_json(args),
        "timestamp": timestamp,
        "run_identifier": os.path.splitext(os.path.basename(metrics_path))[0],
    }

    metrics_payload = {
        "dataset": dataset,
        "method": "pretrain",
        "kge": model,
        "training_time": round(float(training_time), 6),
        "total_wall_time": None if run_wall_time is None else round(float(run_wall_time), 6),
        "test_duration": None if test_duration is None else round(float(test_duration), 6),
        "best_validation": round(float(best_validation), 6),
        "test_metrics": to_serializable(test_results),
        "log_path": log_path,
        "timestamp": timestamp,
        "hyperparameters_file": os.path.basename(hyper_path),
        "run_identifier": hyper_payload["run_identifier"],
    }

    with open(hyper_path, "w", encoding="utf-8") as hf:
        json.dump(hyper_payload, hf, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as mf:
        json.dump(metrics_payload, mf, indent=2)

    # Also persist run artifacts next to checkpoints (user-facing convenience).
    checkpoint_dir = getattr(args, "pretrain_save_path", None)
    config_path = getattr(args, "config_path", None)
    if isinstance(checkpoint_dir, str) and checkpoint_dir:
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            run_id = str(hyper_payload.get("run_identifier") or "run")
            base_id = run_id[:-7] if run_id.endswith("_metrics") else run_id
            checkpoint_results_path = os.path.join(checkpoint_dir, f"{base_id}_results.json")
            with open(checkpoint_results_path, "w", encoding="utf-8") as cf:
                json.dump(metrics_payload, cf, indent=2)
            if isinstance(config_path, str) and os.path.isfile(config_path):
                checkpoint_config_path = os.path.join(checkpoint_dir, f"{base_id}_config.yaml")
                with open(config_path, "r", encoding="utf-8") as srcf:
                    config_text = srcf.read()
                with open(checkpoint_config_path, "w", encoding="utf-8") as dstf:
                    dstf.write(config_text)
        except Exception:
            # Never fail pretraining due to persistence issues.
            pass


def _extract_standard_kge_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standard KGE evaluation metrics from a results dictionary.

    Returns metrics in the expected form: MRR, MAR (mean rank), Hits@1, Hits@10.
    """
    mean_rank = results.get("mean_rank", results.get("mr"))
    metrics: Dict[str, Any] = {
        "mrr": results.get("mrr"),
        "mar": mean_rank,
        "hits1": results.get("hits1"),
        "hits10": results.get("hits10"),
    }
    return {k: to_serializable(v) for k, v in metrics.items() if v is not None}


def save_pretrain_checkpoint_metrics(
    args: Any,
    test_results: Dict[str, Any],
    checkpoint_dir: str,
    checkpoint_name: str = "model_best.tar",
    filename: str = "pretrain_metrics.json",
) -> str:
    """Write a JSON baseline next to the pretrained checkpoint for later comparison."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    metrics = _extract_standard_kge_metrics(test_results or {})
    payload = {
        "dataset": getattr(args, "data_name", "unknown_dataset"),
        "kge": getattr(args, "kge", "unknown_kge"),
        "checkpoint": checkpoint_name,
        "metrics": metrics,
        "duration_s": to_serializable((test_results or {}).get("duration")),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    out_path = os.path.join(checkpoint_dir, filename)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path


def save_pretrain_unlearning_baseline(
    args: Any,
    checkpoint_dir: str,
    timestep_records: Sequence[Dict[str, Any]],
    checkpoint_name: str = "model_best.tar",
    filename: str = "pretrain_unlearning_baseline.json",
) -> str:
    """Persist pretrain baseline on *the same retain/forget splits* used during SGKU."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "dataset": getattr(args, "data_name", "unknown_dataset"),
        "kge": getattr(args, "kge", "unknown_kge"),
        "checkpoint": checkpoint_name,
        "timesteps_num": int(getattr(args, "timesteps_num", len(timestep_records) or 0) or 0),
        "timesteps": [to_serializable(record) for record in (timestep_records or [])],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    out_path = os.path.join(checkpoint_dir, filename)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path
