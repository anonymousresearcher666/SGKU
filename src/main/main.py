import argparse
import csv
import datetime
import logging
import os
import shutil
import sys
import time
import yaml
from typing import Any, Dict

# Enable CPU fallback for unsupported MPS ops (users can override by exporting the var explicitly).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("LIBOMP_USE_SHM", "0")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("KMP_BLOCKTIME", "1")
try:
    from prettytable import PrettyTable
except ImportError:
    class PrettyTable:
        def __init__(self):
            self.field_names = []
            self._rows = []

        def add_row(self, row):
            self._rows.append(list(row))

        def __str__(self):
            columns = [list(map(str, col)) for col in zip(*([self.field_names] + self._rows))] if self.field_names else []
            widths = [max(len(item) for item in col) for col in columns] if columns else []
            header = " | ".join(
                name.ljust(width) for name, width in zip(self.field_names, widths)
            ) if self.field_names else ""
            separator = "-+-".join("-" * width for width in widths) if widths else ""
            rows = [
                " | ".join(str(item).ljust(width) for item, width in zip(row, widths))
                for row in self._rows
            ]
            table_lines = []
            if header:
                table_lines.append(header)
                table_lines.append(separator)
            table_lines.extend(rows)
            return "\n".join(table_lines)
from torch.optim import Adam
import torch
import torch.multiprocessing as mp

try:
    mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from src.loading.KG import KGUnlearningData
from src.model.Retrain import Retrain
from src.model.SGKU import SGKU
from src.model.SGKUIntuitor import SGKUIntuitor
from src.model.SDKU import SDKU

from src.runners.tester import *
from src.runners.trainer import *
from src.utilities.utilities import resolve_device, set_seeds
from src.utilities.persistence import save_unlearning_results, save_pretrain_unlearning_baseline
from src.utilities.kge_factory import get_kge_model_class


class Runner():
    def __init__(self, args) -> None:
        """ 1. Set parameters, seeds, logger, paths and device """
        """ Set parameters """
        self.args = args

        """ Important: Set unlearning parameters """
        self.args.begin_pretrain = False
        self.args.begin_unleanring = True

        """ Set seeds """
        set_seeds(self.args.seed)
        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        base_log_dir = getattr(self.args, "log_path", os.path.join(PROJECT_ROOT, "logs"))
        if not os.path.isabs(base_log_dir):
            base_log_dir = os.path.join(PROJECT_ROOT, base_log_dir)
        os.makedirs(base_log_dir, exist_ok=True)
        self.args.log_path = os.path.join(base_log_dir, f"{now_time}_{self.args.unlearning_method}")

        logging_file_name = f'{self.args.log_path}.log'
        os.makedirs(os.path.dirname(logging_file_name), exist_ok=True)
        os.makedirs(os.path.dirname(logging_file_name), exist_ok=True)
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        # Set output directory for this run
        base_unlearning_dir = getattr(self.args, "unlearning_save_path", os.path.join(PROJECT_ROOT, "checkpoint_unlearning"))
        if not os.path.isabs(base_unlearning_dir):
            base_unlearning_dir = os.path.join(PROJECT_ROOT, base_unlearning_dir)
        save_dir = os.path.join(
            base_unlearning_dir,
            self.args.unlearning_method,
            self.args.data_name,
            self.args.kge.lower()
        )
        overwrite = bool(getattr(self.args, "overwrite_unlearning_dir", True))
        if os.path.exists(save_dir) and overwrite:
            shutil.rmtree(save_dir, True)
        os.makedirs(save_dir, exist_ok=True)
        self.args.unlearning_save_path = save_dir

        """ Set device """
        device = resolve_device(getattr(self.args, "device", None))
        self.args.device = device

        # Ensure model-specific forget sets exist
        kge_suffix = getattr(self.args, "kge", "").lower()
        data_root = getattr(self.args, "data_path", os.path.join(PROJECT_ROOT, "data"))
        if not os.path.isabs(data_root):
            data_root = os.path.join(PROJECT_ROOT, data_root)
        dataset_dir = os.path.join(data_root, self.args.data_name)
        candidate_timesteps = []
        if kge_suffix:
            candidate_timesteps.append(os.path.join(dataset_dir, "forget_sets", kge_suffix, "timesteps"))
            candidate_timesteps.append(os.path.join(dataset_dir, f"timesteps_{kge_suffix}"))
        candidate_timesteps.append(os.path.join(dataset_dir, "timesteps"))

        resolved_timesteps = next((path for path in candidate_timesteps if os.path.isdir(path)), None)
        if resolved_timesteps is None:
            candidates_msg = "\n".join(f"  - {path}" for path in candidate_timesteps)
            raise FileNotFoundError(
                "Unable to locate model-specific timesteps directory. Checked:\n"
                f"{candidates_msg}\n"
                "Please run build_forget_set.py for the appropriate dataset/model pairing before launching SGKU."
            )
        self.args.timesteps_dir = resolved_timesteps

        """ 2. Define data """
        self.kg = KGUnlearningData(self.args)

        """ 3. Define model """
        self.model, self.optimizer = self.create_model()
        self.pretrain_loaded = False
        self.pretrain_checkpoint_dir = None
        self.pretrain_baseline_by_timestep = {}
        if self.args.unlearning_method in ["SGKU", "SDKU"]:
            self.load_pretrained_weights()
            self.write_pretrain_baseline_on_unlearning_splits()

    def _append_epoch_metrics(
        self,
        *,
        timestep: int,
        epoch: int,
        loss: float,
        train_time: float,
        eval_time: float,
        epoch_wall_time: float,
        forget_results: Dict[str, Any],
        retain_results: Dict[str, Any],
        valid_res: Dict[str, Any],
    ) -> None:
        if not bool(getattr(self.args, "save_epoch_metrics", False)):
            return

        def _f(val):
            try:
                return None if val is None else float(val)
            except (TypeError, ValueError):
                return None

        lr = None
        if hasattr(self, "optimizer") and self.optimizer is not None:
            try:
                lr = float(self.optimizer.param_groups[0].get("lr"))
            except Exception:
                lr = None

        mrr_f = _f(forget_results.get("mrr"))
        mrr_r = _f(retain_results.get("mrr"))
        one_minus_mrr_f = None if mrr_f is None else 1.0 - mrr_f
        mrr_avg = None
        mrr_f1 = None
        if mrr_r is not None and one_minus_mrr_f is not None:
            mrr_avg = 0.5 * (mrr_r + one_minus_mrr_f)
            denom = mrr_r + one_minus_mrr_f
            if denom > 0:
                mrr_f1 = (2.0 * mrr_r * one_minus_mrr_f) / denom

        row = {
            "timestep": int(timestep),
            "epoch": int(epoch),
            "train_loss": _f(loss),
            "lr": lr,
            "train_time": _f(train_time),
            "eval_time": _f(eval_time),
            "epoch_wall_time": _f(epoch_wall_time),
            "retain_mrr": mrr_r,
            "forget_mrr": mrr_f,
            "retain_hits10": _f(retain_results.get("hits10")),
            "forget_hits10": _f(forget_results.get("hits10")),
            "mrr_avg": mrr_avg,
            "mrr_f1": mrr_f1,
            "valid_score": _f(valid_res.get(self.args.valid_metrics)),
        }

        out_path = os.path.join(
            self.args.unlearning_save_path,
            f"epoch_metrics_t{int(timestep)}.csv",
        )
        file_exists = os.path.isfile(out_path)
        with open(out_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def load_pretrained_weights(self):
        pretrain_base = getattr(self.args, "pretrain_save_path", "checkpoint_pretrain")
        if not os.path.isabs(pretrain_base):
            pretrain_base = os.path.join(PROJECT_ROOT, pretrain_base)
        dataset = getattr(self.args, "data_name", "")
        kge = getattr(self.args, "kge", "")

        candidate_dirs = []
        if dataset and kge:
            candidate_dirs.append(os.path.join(pretrain_base, dataset, kge))
        if dataset:
            candidate_dirs.append(os.path.join(pretrain_base, dataset))
        candidate_dirs.append(pretrain_base)

        for directory in candidate_dirs:
            candidate_file = os.path.join(directory, "model_best.tar")
            if os.path.isfile(candidate_file):
                try:
                    checkpoint = torch.load(candidate_file, map_location=self.args.device, weights_only=True)
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    if not isinstance(state_dict, dict):
                        raise ValueError("Checkpoint does not contain a state_dict mapping.")

                    # Pretraining saves raw KGE model keys (e.g., 'ent_embeddings.weight'),
                    # while SGKU/Retrain wrap the KGE model under 'kge_model.'.
                    # Load into the correct target module to ensure the pretrained weights are actually used.
                    target = self.model
                    if any(str(k).startswith("kge_model.") for k in state_dict.keys()):
                        target = self.model
                    elif hasattr(self.model, "kge_model"):
                        target = self.model.kge_model

                    missing, unexpected = target.load_state_dict(state_dict, strict=False)
                    if missing:
                        self.args.logger.warning(
                            f"Pretrain load: missing keys ({len(missing)}). Example: {missing[:3]}"
                        )
                    if unexpected:
                        self.args.logger.warning(
                            f"Pretrain load: unexpected keys ({len(unexpected)}). Example: {unexpected[:3]}"
                        )
                    self.pretrain_loaded = True
                    self.pretrain_checkpoint_dir = directory
                    self.args.logger.info(f"Loaded pretrained checkpoint from {candidate_file}")
                    return
                except Exception as exc:
                    self.args.logger.warning(f"Failed to load pretrained checkpoint '{candidate_file}': {exc}")
        self.args.logger.warning("No pretrained checkpoint found; proceeding from scratch.")
        exit()

    def _standard_metrics_from_retain_results(self, retain_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mrr": retain_results.get("mrr"),
            "mar": retain_results.get("mean_rank", retain_results.get("mr")),
            "hits1": retain_results.get("hits1"),
            "hits10": retain_results.get("hits10"),
            "count": retain_results.get("count"),
            "duration": retain_results.get("duration"),
        }

    def _standard_metrics_from_forget_results(self, forget_results: Dict[str, Any]) -> Dict[str, Any]:
        pos = forget_results.get("pos_metrics") or {}
        return {
            "mrr": forget_results.get("mrr"),
            "mar": pos.get("mr", forget_results.get("mean_rank")),
            "hits1": forget_results.get("hits1"),
            "hits10": forget_results.get("hits10"),
            "count": forget_results.get("count"),
            "duration": forget_results.get("duration"),
        }

    def write_pretrain_baseline_on_unlearning_splits(self) -> None:
        """Ensure retention deltas are computed on the same split (retain set), not full-test."""
        if not self.pretrain_loaded or not self.pretrain_checkpoint_dir:
            return
        if getattr(self.args, "write_pretrain_baseline", True) is False:
            return

        timesteps_num = int(getattr(self.args, "timesteps_num", 0) or 0)
        if timesteps_num <= 0:
            return

        baseline_records = []
        self.pretrain_baseline_by_timestep = {}
        prev_valid = getattr(self.args, "valid", False)
        try:
            self.args.valid = False
            for timestep in range(timesteps_num):
                self.args.timestep = timestep
                self.args.timestep_test = timestep
                self.args.timestep_validation = timestep
                tester = UnlearningTester(self.args, self.kg, self.model)
                forget_results, retain_results = tester.test()
                retain_metrics = self._standard_metrics_from_retain_results(retain_results)
                forget_metrics = self._standard_metrics_from_forget_results(forget_results)
                record = {
                    "timestep": int(timestep),
                    "retain": retain_metrics,
                    "forget": forget_metrics,
                }
                baseline_records.append(record)
                self.pretrain_baseline_by_timestep[int(timestep)] = record
        finally:
            self.args.valid = prev_valid

        out_path = save_pretrain_unlearning_baseline(
            args=self.args,
            checkpoint_dir=self.pretrain_checkpoint_dir,
            timestep_records=baseline_records,
            checkpoint_name="model_best.tar",
        )
        # Expose baseline for trainers/processors so validation logs can compare on the same split.
        self.args.pretrain_unlearning_baseline = self.pretrain_baseline_by_timestep
        self.args.logger.info(
            f"Saved pretrained baseline on unlearning splits to {out_path} "
            f"(retain/forget per timestep; use this for retention comparisons)."
        )

    def _log_retain_delta_vs_pretrain(self, timestep: int, retain_results: Dict[str, Any]) -> None:
        baseline = self.pretrain_baseline_by_timestep.get(int(timestep), {}).get("retain")
        if not baseline:
            return
        base_mrr = baseline.get("mrr")
        base_hits10 = baseline.get("hits10")
        cur_mrr = retain_results.get("mrr")
        cur_hits10 = retain_results.get("hits10")
        if base_mrr is None or cur_mrr is None:
            return
        msg = (
            "Retain vs pretrained baseline (same split) -> "
            f"ΔMRR: {float(cur_mrr) - float(base_mrr):+.4f}"
        )
        if base_hits10 is not None and cur_hits10 is not None:
            msg += f", ΔHits@10: {float(cur_hits10) - float(base_hits10):+.4f}"
        self.args.logger.info(msg)
        print(msg)

    def _success_checkpoint_path(self, timestep: int) -> str:
        return os.path.join(self.args.unlearning_save_path, f"{int(timestep)}model_best_success.tar")

    def _best_checkpoint_path(self, timestep: int) -> str:
        return os.path.join(self.args.unlearning_save_path, f"{int(timestep)}model_best.tar")

    def _select_checkpoint_for_timestep(self, timestep: int) -> str:
        prefer_success = bool(getattr(self.args, "prefer_success_checkpoint", True))
        success_path = self._success_checkpoint_path(timestep)
        best_path = self._best_checkpoint_path(timestep)
        if prefer_success and os.path.isfile(success_path):
            return success_path
        return best_path

    def _get_pretrain_baseline_metric(self, timestep: int, which: str, metric_key: str):
        baseline = getattr(self.args, "pretrain_unlearning_baseline", None)
        if not isinstance(baseline, dict):
            baseline = getattr(self, "pretrain_baseline_by_timestep", None)
        if not isinstance(baseline, dict):
            return None
        entry = baseline.get(int(timestep))
        if not isinstance(entry, dict):
            return None
        sub = entry.get(which)
        if not isinstance(sub, dict):
            return None
        return sub.get(metric_key)

    def _evaluate_success_criteria(
        self,
        timestep: int,
        *,
        retain_results: Dict[str, Any],
        forget_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        metric_key = str(getattr(self.args, "success_metric", "mrr"))
        retain_drop_max_frac = float(getattr(self.args, "success_retain_drop_max_frac", 0.05))
        forget_improve_min_frac = float(getattr(self.args, "success_forget_improve_min_frac", 0.10))

        base_retain = self._get_pretrain_baseline_metric(timestep, "retain", metric_key)
        base_forget = self._get_pretrain_baseline_metric(timestep, "forget", metric_key)

        # Current metrics: retain set uses global metrics; forget set "mar" should use positive ranks.
        if metric_key in {"mar", "mean_rank", "mr"}:
            cur_retain = retain_results.get("mean_rank", retain_results.get("mr"))
            forget_pos = forget_results.get("pos_metrics") if isinstance(forget_results.get("pos_metrics"), dict) else {}
            cur_forget = forget_pos.get("mr", forget_results.get("mean_rank", forget_results.get("mr")))
        else:
            cur_retain = retain_results.get(metric_key)
            cur_forget = forget_results.get(metric_key)

        try:
            base_retain_f = None if base_retain is None else float(base_retain)
            base_forget_f = None if base_forget is None else float(base_forget)
            cur_retain_f = None if cur_retain is None else float(cur_retain)
            cur_forget_f = None if cur_forget is None else float(cur_forget)
        except Exception:
            return {
                "metric": metric_key,
                "successful": None,
                "score": None,
                "details": "Metric conversion failed (non-numeric values).",
            }

        if base_retain_f is None or base_forget_f is None or cur_retain_f is None or cur_forget_f is None:
            return {
                "metric": metric_key,
                "successful": None,
                "score": None,
                "details": "No pretrained baseline for this timestep/metric; cannot compute success criteria.",
            }

        eps = 1e-12
        # Retain: most metrics (MRR/Hits) are higher-is-better, but rank-based metrics are lower-is-better.
        retain_lower_is_better = metric_key in {"mar", "mean_rank", "mr"}
        if retain_lower_is_better:
            retain_degrade = max(0.0, cur_retain_f - base_retain_f)
        else:
            retain_degrade = max(0.0, base_retain_f - cur_retain_f)
        retain_drop_frac = retain_degrade / max(abs(base_retain_f), eps)

        forget_improve = max(0.0, base_forget_f - cur_forget_f)  # lower forget metric is better
        forget_improve_frac = forget_improve / max(abs(base_forget_f), eps)

        retention_score = max(0.0, min(1.0, 1.0 - retain_drop_frac))
        forgetting_score = max(0.0, min(1.0, forget_improve_frac))
        score = 0.5 * retention_score + 0.5 * forgetting_score

        successful = (retain_drop_frac <= retain_drop_max_frac) and (forget_improve_frac >= forget_improve_min_frac)

        return {
            "metric": metric_key,
            "successful": bool(successful),
            "score": float(score),
            "fractions": {
                "retain_drop_frac": float(retain_drop_frac),
                "forget_improve_frac": float(forget_improve_frac),
            },
            "thresholds": {
                "retain_drop_max_frac": float(retain_drop_max_frac),
                "forget_improve_min_frac": float(forget_improve_min_frac),
            },
            "baseline": {"retain": float(base_retain_f), "forget": float(base_forget_f)},
            "current": {"retain": float(cur_retain_f), "forget": float(cur_forget_f)},
        }

    def _predict_all_tails(self, head: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "predict"):
            return self.model.predict(head, relation)
        return self.model.kge_model.predict(head, relation)

    def _run_unlearning_audit(self) -> Dict[str, Any]:
        if not bool(getattr(self.args, "enable_unlearning_audit", True)):
            return {"enabled": False}

        rank_threshold = int(getattr(self.args, "audit_rank_threshold", 10) or 10)
        score_threshold = float(getattr(self.args, "audit_score_threshold", 0.0) or 0.0)
        max_samples = int(getattr(self.args, "audit_max_samples", 0) or 0)

        processor = ForgetDBatching(self.args, self.kg)
        self.model.eval()

        total = 0
        passed = 0
        tail_rank_sum = 0.0
        head_rank_sum = 0.0
        tail_score_sum = 0.0
        head_score_sum = 0.0
        start = time.time()

        with torch.no_grad():
            for head, relation, tail, label in processor.data_loader:
                if max_samples > 0 and total >= max_samples:
                    break
                if max_samples > 0:
                    remaining = max_samples - total
                    if head.size(0) > remaining:
                        head = head[:remaining]
                        relation = relation[:remaining]
                        tail = tail[:remaining]
                        label = label[:remaining]

                head = head.to(self.args.device)
                relation = relation.to(self.args.device)
                tail = tail.to(self.args.device)
                label = label.to(self.args.device)
                bs = head.size(0)
                row_idx = torch.arange(bs, device=self.args.device)

                # Tail mode.
                pred_tail = self._predict_all_tails(head, relation)
                tail_score = pred_tail[row_idx, tail]
                pred_tail = torch.where(label.bool(), -torch.ones_like(pred_tail) * 10000000, pred_tail)
                pred_tail[row_idx, tail] = tail_score
                tail_rank = 1.0 + (pred_tail > tail_score.unsqueeze(1)).sum(dim=1).float()

                # Head mode via inverse relation query (t, r^{-1}, ?).
                inv_relation = torch.where(relation % 2 == 0, relation + 1, relation - 1)
                pred_head = self._predict_all_tails(tail, inv_relation)
                head_score = pred_head[row_idx, head]
                pred_head_filtered = pred_head.clone()
                for i in range(bs):
                    key = (int(tail[i].item()), int(inv_relation[i].item()))
                    true_heads = self.kg.hr2t.get(key, set())
                    if true_heads:
                        idx = torch.tensor(list(true_heads), dtype=torch.long, device=self.args.device)
                        pred_head_filtered[i, idx] = -10000000
                    pred_head_filtered[i, head[i]] = head_score[i]
                head_rank = 1.0 + (pred_head_filtered > head_score.unsqueeze(1)).sum(dim=1).float()

                cond = (
                    (tail_rank > rank_threshold)
                    & (head_rank > rank_threshold)
                    & (tail_score < score_threshold)
                    & (head_score < score_threshold)
                )
                passed += int(cond.sum().item())
                total += int(bs)
                tail_rank_sum += float(tail_rank.sum().item())
                head_rank_sum += float(head_rank.sum().item())
                tail_score_sum += float(tail_score.sum().item())
                head_score_sum += float(head_score.sum().item())

        duration = time.time() - start
        if total <= 0:
            return {
                "enabled": True,
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "rank_threshold": rank_threshold,
                "score_threshold": score_threshold,
                "duration": round(float(duration), 4),
            }

        return {
            "enabled": True,
            "total": int(total),
            "passed": int(passed),
            "pass_rate": round(float(passed) / float(total), 6),
            "rank_threshold": int(rank_threshold),
            "score_threshold": float(score_threshold),
            "avg_tail_rank": round(tail_rank_sum / float(total), 6),
            "avg_head_rank": round(head_rank_sum / float(total), 6),
            "avg_tail_score": round(tail_score_sum / float(total), 6),
            "avg_head_score": round(head_score_sum / float(total), 6),
            "duration": round(float(duration), 4),
        }

    def create_model(self):
        if self.args.unlearning_method in ["pretrain", "finetune"]:
            model = Retrain(self.args, self.kg)
        elif self.args.unlearning_method == "SGKU":
            model_class = get_kge_model_class(self.args.kge)
            model = SGKU(args=self.args, kg=self.kg, kge_model_class=model_class, schema_store=self.kg.schema_store)
        elif self.args.unlearning_method == "SDKU":
            model_class = get_kge_model_class(self.args.kge)
            model = SDKU(args=self.args, kg=self.kg, kge_model_class=model_class, schema_store=self.kg.schema_store)
        elif self.args.unlearning_method == "SGKUIntuitor":
             model_class = get_kge_model_class(self.args.kge)
             model = SGKUIntuitor(args=self.args, kg=self.kg, kge_model_class=model_class, schema_store=self.kg.schema_store)
        else:
            model = Retrain(self.args, self.kg)

        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def reset_model(self):
        model = Retrain(self.args, self.kg)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def unlearning(self):
        report_results = PrettyTable()
        report_results.field_names = [
            'Timestep',
            'Time',
            'MRR+',
            'Hits@1+',
            'Hits@10+',
            'MRR-',
            'Hits@1-',
            'Hits@10-',
            'MRR_Avg',
            'MRR_F1',
        ]
        training_times = []
        timestep_records = []
        timesteps_num = int(self.args.timesteps_num)
        start_timestep = int(getattr(self.args, "start_timestep", 0) or 0)
        if start_timestep < 0 or start_timestep >= timesteps_num:
            start_timestep = 0

        # Resume support: if starting from a later timestep, load the previous best checkpoint so
        # sequential unlearning continues from the correct parameters.
        if start_timestep > 0:
            prev_best = self._select_checkpoint_for_timestep(start_timestep - 1)
            if os.path.isfile(prev_best):
                self.args.logger.info(f"Resuming: loading previous timestep checkpoint {prev_best}")
                self.load_checkpoint(prev_best)
            else:
                self.args.logger.warning(
                    f"Resuming from timestep {start_timestep} but previous checkpoint not found: {prev_best}. "
                    "Proceeding from current model parameters."
                )

        for ss_id in range(start_timestep, timesteps_num):
            self.args.timestep = ss_id
            self.args.timestep_test = ss_id
            self.args.timestep_validation = ss_id
            print(f"\n===== TIMESTEP {ss_id}/{timesteps_num - 1} =====")
            self.args.logger.info(f"===== TIMESTEP {ss_id}/{timesteps_num - 1} =====")

            if self.args.unlearning_method == "pretrain":
                self.model, self.optimizer = self.reset_model()
            if self.args.unlearning_method in ["SGKU", "SDKU"]:
                self.model.save_embeddings()
            """ training """

            print(f"Starting training for timestep {ss_id}...")

            training_time = self.train()

            """ save and load model """
            best_checkpoint = self._select_checkpoint_for_timestep(ss_id)
            self.args.logger.info(f"[timestep {ss_id}] Selecting checkpoint for test/next timestep: {best_checkpoint}")
            if self.args.unlearning_method == "SDKU":
                self.args.logger.info(
                    f"[timestep {ss_id}] SDKU safeguard: keeping rolling weights (skip checkpoint reload)."
                )
            else:
                self.load_checkpoint(best_checkpoint)

            """ predict """
            forget_results, retain_results = self.test()
            self._log_retain_delta_vs_pretrain(ss_id, retain_results)
            audit_results = self._run_unlearning_audit()
            self.args.logger.info(
                f"[timestep {ss_id}] Unlearning audit -> pass_rate={audit_results.get('pass_rate', 'NA')} "
                f"({audit_results.get('passed', 'NA')}/{audit_results.get('total', 'NA')}) "
                f"R={audit_results.get('rank_threshold', 'NA')} Δ={audit_results.get('score_threshold', 'NA')}"
            )

            retain_mar = retain_results.get("mean_rank", retain_results.get("mr"))
            forget_pos = forget_results.get("pos_metrics") if isinstance(forget_results.get("pos_metrics"), dict) else {}
            forget_mar = forget_pos.get("mr", forget_results.get("mean_rank"))
            mrr_r = retain_results.get("mrr")
            mrr_f = forget_results.get("mrr")
            mrr_avg = None
            mrr_f1 = None
            if mrr_r is not None and mrr_f is not None:
                one_minus_mrr_f = 1.0 - float(mrr_f)
                mrr_avg = (float(mrr_r) + one_minus_mrr_f) / 2.0
                denom = float(mrr_r) + one_minus_mrr_f
                if denom > 0:
                    mrr_f1 = (2.0 * float(mrr_r) * one_minus_mrr_f) / denom
                else:
                    mrr_f1 = 0.0
            def pct(val):
                try:
                    return "NA" if val is None else round(float(val) * 100.0, 3)
                except (TypeError, ValueError):
                    return "NA"
            test_full_log = (
                f"TEST Timestep:{ss_id}\t"
                f"retain(MRR%:{pct(retain_results.get('mrr'))},MR:{retain_mar if retain_mar is not None else 'NA'},"
                f"H@1%:{pct(retain_results.get('hits1'))},H@3%:{pct(retain_results.get('hits3'))},H@10%:{pct(retain_results.get('hits10'))})\t"
                f"forget(MRR%:{pct(forget_results.get('mrr'))},MR:{forget_mar if forget_mar is not None else 'NA'},"
                f"H@1%:{pct(forget_results.get('hits1'))},H@3%:{pct(forget_results.get('hits3'))},H@10%:{pct(forget_results.get('hits10'))})\t"
                f"composite(MRR_Avg%:{'NA' if mrr_avg is None else round(float(mrr_avg) * 100.0,3)},MRR_F1:{'NA' if mrr_f1 is None else round(float(mrr_f1),6)})"
            )
            self.args.logger.info(test_full_log)

            training_times.append(float(training_time))
            timestep_records.append({
                "timestep": int(ss_id),
                "train_time": round(float(training_time), 6),
                "forget": forget_results,
                "retain": retain_results,
                "audit": audit_results,
            })
            forget_test_time = forget_results.get('duration', 0.0)
            retain_test_time = retain_results.get('duration', 0.0)
            total_test_time = forget_test_time + retain_test_time
            test_time_message = (
                f"Timestep {ss_id} testing times -> Forget: {forget_test_time:.2f}s, "
                f"Retain: {retain_test_time:.2f}s, Total: {total_test_time:.2f}s"
            )
            print(test_time_message)
            self.args.logger.info(test_time_message)

            """ output """
            mrr_avg_for_table = None
            mrr_f1_for_table = None
            try:
                mrr_r_val = retain_results.get("mrr")
                mrr_f_val = forget_results.get("mrr")
                if mrr_r_val is not None and mrr_f_val is not None:
                    one_minus = 1.0 - float(mrr_f_val)
                    mrr_avg_for_table = (float(mrr_r_val) + one_minus) / 2.0
                    denom = float(mrr_r_val) + one_minus
                    mrr_f1_for_table = (2.0 * float(mrr_r_val) * one_minus / denom) if denom > 0 else 0.0
            except Exception:
                mrr_avg_for_table = None
                mrr_f1_for_table = None

            report_results.add_row([
                self.args.timestep,
                training_time,
                retain_results["mrr"],
                retain_results["hits1"],
                retain_results["hits10"],
                forget_results["mrr"],
                forget_results["hits1"],
                forget_results["hits10"],
                "NA" if mrr_avg_for_table is None else round(float(mrr_avg_for_table), 6),
                "NA" if mrr_f1_for_table is None else round(float(mrr_f1_for_table), 6),
            ])
            self.args.logger.info(f"\n{report_results}")
        return report_results, training_times, timestep_records

    def get_report_results(self, results):
        mrrs, hits1s, hits3s, hits10s, num_test = [], [], [], [], []
        for idx, result in enumerate(results):
            mrrs.append(result['mrr'])
            hits1s.append(result['hits1'])
            hits3s.append(result['hits3'])
            hits10s.append(result['hits10'])
            num_test.append(len(self.kg.train_data))
        whole_mrr = sum(
            mrr * num_test[i] for i, mrr in enumerate(mrrs)
        ) / sum(num_test)
        whole_hits1 = sum(
            hits1 * num_test[i] for i, hits1 in enumerate(hits1s)
        ) / sum(num_test)
        whole_hits3 = sum(
            hits3 * num_test[i] for i, hits3 in enumerate(hits3s)
        ) / sum(num_test)
        whole_hits10 = sum(
            hits10 * num_test[i] for i, hits10 in enumerate(hits10s)
        ) / sum(num_test)
        return round(whole_mrr, 3), round(whole_hits1, 3), round(whole_hits3, 3), round(whole_hits10, 3)

    def train(self):
        start_time = time.time()
        self.best_valid = 0.0
        self.stop_epoch = 0
        self.best_success_score = -1.0
        self.best_success_epoch = None
        self.success_patience_counter = 0


        trainer = UnlearningTrainer(self.args, self.kg, self.model, self.optimizer)

        print("\n========================================")
        print(f"STARTING TRAINING: {self.args.epoch_num} epochs")
        print(f"Model: {self.args.unlearning_method}")
        print(f"Device: {self.args.device}")
        print(f"Batch Size: {self.args.batch_size}")
        print("========================================")

        for epoch in range(int(self.args.epoch_num)):
            epoch_start_time = time.time()
            timestep = getattr(self.args, "timestep", None)
            prefix = f"[timestep {timestep}] " if timestep is not None else ""
            print(f"{prefix}=== EPOCH {epoch + 1}/{int(self.args.epoch_num)} STARTED ===")
            if getattr(self.args, "debug", False) and epoch > 0:
                print("DEBUG MODE: Stopping after first epoch")
                break

            self.args.epoch = epoch
            success_candidate = False
            success_candidate_score = None
            if self.args.unlearning_method in ["pretrain", "retrain", "finetune", "SGKU", "SDKU", "SGKUIntuitor"]:
                valid_res = dict()
                loss, train_time, eval_time, forget_results, retain_results = trainer.run_epoch()
                total_epoch_time = time.time() - epoch_start_time
                print(
                    f"Epoch ({epoch}/{int(self.args.epoch_num)}) timing -> "
                    f"train: {train_time:.2f}s, eval: {0.0 if eval_time is None else eval_time:.2f}s, "
                    f"wall: {total_epoch_time:.2f}s"
                )

                # Validation phase
                if self.args.epoch % self.args.valid_gap == 0:
                    print(f"\n--- VALIDATION PHASE FOR EPOCH {epoch + 1} ---")
                    print(f"Forget results: {forget_results}")
                    print(f"Retain results: {retain_results}")
                    # Calculate combined metrics
                    valid_res[self.args.valid_metrics] = 0.5 * (1 - forget_results[self.args.valid_metrics]) + 0.5 * \
                                                         retain_results[self.args.valid_metrics]
                    valid_res['hits3'] = 0.5 * (1 - forget_results['hits3']) + 0.5 * retain_results['hits3']
                    valid_res['hits10'] = 0.5 * (1 - forget_results['hits10']) + 0.5 * retain_results['hits10']
                    print(f"Combined validation results: {valid_res}")
                    print(f"--- VALIDATION PHASE COMPLETE ---\n")
                    self._append_epoch_metrics(
                        timestep=int(getattr(self.args, "timestep", 0) or 0),
                        epoch=int(self.args.epoch),
                        loss=float(loss),
                        train_time=float(train_time),
                        eval_time=0.0 if eval_time is None else float(eval_time),
                        epoch_wall_time=float(total_epoch_time),
                        forget_results=forget_results,
                        retain_results=retain_results,
                        valid_res=valid_res,
                    )

                    retain_mar = retain_results.get("mean_rank", retain_results.get("mr"))
                    forget_pos = forget_results.get("pos_metrics") if isinstance(forget_results.get("pos_metrics"), dict) else {}
                    forget_mar = forget_pos.get("mr", forget_results.get("mean_rank"))
                    mrr_r = retain_results.get("mrr")
                    mrr_f = forget_results.get("mrr")
                    mrr_f1 = None
                    if mrr_r is not None and mrr_f is not None:
                        one_minus_mrr_f = 1.0 - float(mrr_f)
                        denom = float(mrr_r) + one_minus_mrr_f
                        if denom > 0:
                            mrr_f1 = (2.0 * float(mrr_r) * one_minus_mrr_f) / denom
                        else:
                            mrr_f1 = 0.0

                    def pct(val):
                        try:
                            return "NA" if val is None else round(float(val) * 100.0, 3)
                        except (TypeError, ValueError):
                            return "NA"

                    full_log = (
                        f"VALID Epoch:{epoch + 1}\t"
                        f"retain(MRR%:{pct(retain_results.get('mrr'))},MR:{retain_mar if retain_mar is not None else 'NA'},"
                        f"H@1%:{pct(retain_results.get('hits1'))},H@3%:{pct(retain_results.get('hits3'))},H@10%:{pct(retain_results.get('hits10'))})\t"
                        f"forget(MRR%:{pct(forget_results.get('mrr'))},MR:{forget_mar if forget_mar is not None else 'NA'},"
                        f"H@1%:{pct(forget_results.get('hits1'))},H@3%:{pct(forget_results.get('hits3'))},H@10%:{pct(forget_results.get('hits10'))})\t"
                        f"composite(MRR_Avg%:{pct(valid_res.get('mrr'))},MRR_F1:{'NA' if mrr_f1 is None else round(float(mrr_f1),6)},"
                        f"H@3_Avg%:{pct(valid_res.get('hits3'))},H@10_Avg%:{pct(valid_res.get('hits10'))})"
                    )
                    self.args.logger.info(full_log)

                    # Track "best_success" checkpoint based on YAML success criteria vs pretrained baseline.
                    success_eval = self._evaluate_success_criteria(
                        int(getattr(self.args, "timestep", 0)),
                        retain_results=retain_results,
                        forget_results=forget_results,
                    )
                    stop_on_success = bool(getattr(self.args, "stop_on_success", False))
                    stop_success_mode = str(getattr(self.args, "stop_on_success_mode", "first")).lower()
                    stop_success_triggered = False
                    if success_eval.get("successful") is None:
                        self.args.logger.info(
                            f"[timestep {getattr(self.args, 'timestep', 'NA')}] Success criteria unavailable: {success_eval.get('details')}"
                        )
                    else:
                        frac = success_eval.get("fractions") or {}
                        thr = success_eval.get("thresholds") or {}
                        success_candidate_score = float(success_eval.get("score", -1.0))
                        success_candidate = bool(success_eval.get("successful")) and (
                            success_candidate_score > float(self.best_success_score)
                        )
                        self.args.logger.info(
                            f"[timestep {getattr(self.args, 'timestep', 'NA')}] Success metric={success_eval.get('metric')} "
                            f"retain_drop={float(frac.get('retain_drop_frac', 0.0)):.4f} (<= {float(thr.get('retain_drop_max_frac', 0.0)):.4f}), "
                            f"forget_improve={float(frac.get('forget_improve_frac', 0.0)):.4f} (>= {float(thr.get('forget_improve_min_frac', 0.0)):.4f}), "
                            f"score={success_candidate_score:.4f}, best_success_score={float(self.best_success_score):.4f}"
                        )
                        if stop_on_success:
                            if stop_success_mode in {"patience", "best", "max", "maximum"}:
                                patience = int(getattr(self.args, "stop_on_success_patience", 5) or 5)
                                if success_eval.get("successful") is True:
                                    if success_candidate:
                                        self.success_patience_counter = 0
                                    else:
                                        self.success_patience_counter += 1
                                else:
                                    # Only start counting once we've achieved at least one successful epoch.
                                    if float(self.best_success_score) >= 0.0:
                                        self.success_patience_counter += 1
                                if float(self.best_success_score) >= 0.0 and self.success_patience_counter >= patience:
                                    stop_success_triggered = True
                            else:
                                # "first" mode: stop at the first successful epoch.
                                stop_success_triggered = bool(success_eval.get("successful") is True)
            else:
                raise NotImplementedError

            if self.args.epoch % self.args.valid_gap != 0:
                print(f"Skipping validation (epoch {epoch + 1} % {self.args.valid_gap} != 0)")
                self.args.logger.info(
                    f"TRAIN Epoch:{epoch + 1}\tLoss:{round(loss, 3)}\tTrain(s):{train_time:.2f}\t(validation skipped; valid_gap={self.args.valid_gap})"
                )
                # Optional: save checkpoints even when validation is skipped (useful for mid-gap evaluation).
                if bool(getattr(self.args, "save_each_epoch", False)):
                    self.save_model(is_best=False)
                # Optional: mitigate MPS memory pressure (can reduce slowdowns over long runs).
                if str(getattr(self.args, "device", "")).startswith("mps") and bool(getattr(self.args, "mps_empty_cache_each_epoch", False)):
                    try:
                        torch.mps.empty_cache()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                continue

            # Model saving logic
            if valid_res[self.args.valid_metrics] > self.best_valid:
                prev_best = self.best_valid
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = 0
                print(f"NEW BEST MODEL! Metrics improved from {prev_best:.4f} to {self.best_valid:.4f}")
                out_tar = self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                print(f"No improvement. Current: {valid_res[self.args.valid_metrics]:.4f}, Best: {self.best_valid:.4f}")
                out_tar = self.save_model(is_best=False)
                if self.stop_epoch >= self.args.patience:
                    print(f"\n*** EARLY STOPPING! Epoch: {epoch + 1}, No improvement for {self.stop_epoch} epochs ***")
                    self.args.logger.info(
                        f'Early Stopping! Epoch: {epoch + 1} Best Results: {round(self.best_valid * 100, 3)}'
                    )
                    break

            # Success checkpoint copy (after saving current epoch checkpoint).
            if self.args.epoch % self.args.valid_gap == 0 and (
                success_candidate or (stop_success_triggered and stop_success_mode == "first")
            ):
                try:
                    success_path = self._success_checkpoint_path(int(getattr(self.args, "timestep", 0)))
                    shutil.copyfile(out_tar, success_path)
                    if success_candidate_score is not None:
                        self.best_success_score = float(success_candidate_score)
                        self.best_success_epoch = int(self.args.epoch)
                    self.args.logger.info(
                        f"[timestep {getattr(self.args, 'timestep', 'NA')}] NEW BEST SUCCESS checkpoint -> {success_path} "
                        f"(epoch={epoch + 1}, score={float(self.best_success_score):.4f})"
                    )
                except Exception as exc:
                    self.args.logger.warning(f"Failed to update best_success checkpoint: {exc}")
                if stop_on_success and stop_success_triggered and stop_success_mode == "first":
                    self.args.logger.info(
                        f"[timestep {getattr(self.args, 'timestep', 'NA')}] stop_on_success triggered (mode=first); stopping timestep training."
                    )
                    break
            if stop_on_success and stop_success_triggered and stop_success_mode in {"patience", "best", "max", "maximum"}:
                patience = int(getattr(self.args, "stop_on_success_patience", 5) or 5)
                self.args.logger.info(
                    f"[timestep {getattr(self.args, 'timestep', 'NA')}] stop_on_success triggered (mode={stop_success_mode}, "
                    f"patience={patience}); stopping timestep training."
                )
                break
                if stop_on_success and success_candidate and stop_success_mode == "best":
                    self.args.logger.info(
                        f"[timestep {getattr(self.args, 'timestep', 'NA')}] stop_on_success triggered (mode=best); stopping timestep training."
                    )
                    break

            # Log results
            if epoch % 1 == 0:
                eval_dur = 0.0
                if self.args.epoch % self.args.valid_gap == 0 and forget_results and retain_results:
                    eval_dur = (eval_time if eval_time is not None else 0.0)
                    # If eval_time wasn't returned, use durations from processors
                    if eval_dur == 0.0:
                        eval_dur = forget_results.get('duration', 0.0) + retain_results.get('duration', 0.0)
                log_message = (
                    f"Epoch:{epoch + 1}\tLoss:{round(loss, 3)}\tMRR-Avg:{round(valid_res['mrr'] * 100, 3)}\t"
                    f"Hits@10:{round(valid_res['hits10'] * 100, 3)}\t"
                    f"Train(s):{train_time:.2f}\tEval(s):{eval_dur:.2f}\t"
                    f"Best:{round(self.best_valid * 100, 3)}"
                )
                print(log_message)
                self.args.logger.info(log_message)

            # Calculate epoch time
            epoch_time = total_epoch_time
            print(f"--- EPOCH {epoch} COMPLETED in {epoch_time:.2f} seconds ---\n")

        total_time = time.time() - start_time
        print("\n========================================")
        print(f"TRAINING COMPLETED: {epoch} epochs")
        print(f"Best validation score: {round(self.best_valid * 100, 3)}")
        if self.best_success_epoch is not None and float(self.best_success_score) >= 0.0:
            print(
                f"Best success checkpoint: epoch {int(self.best_success_epoch) + 1} "
                f"(score={float(self.best_success_score) * 100:.2f}, "
                f"criteria metric={getattr(self.args, 'success_metric', 'mrr')})"
            )
        print(f"Total training time: {total_time:.2f} seconds")
        print("========================================\n")
        return total_time

    def test(self):
        print("========================= Start testing =========================")
        tester = UnlearningTester(self.args, self.kg, self.model)
        return tester.test_with_report()

    def save_model(self, is_best=False) -> str:
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(
            self.args.unlearning_save_path,
            f'{str(self.args.timestep)}checkpoint-{self.args.epoch}.tar',
        )
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(
                self.args.unlearning_save_path, f'{str(self.args.timestep)}model_best.tar'
            )
            shutil.copyfile(out_tar, best_path)
        return out_tar

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info(f"=> loading checkpoint \'{input_file}\'")
            checkpoint = torch.load(input_file, map_location=self.args.device, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info(f'=> no checking found at \'{input_file}\'')


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


def run_experiments(config_path: str = "hyperparameters.yaml"):

    if not os.path.isabs(config_path):
        config_path = os.path.join(BASE_DIR, config_path)

    print(f"Loading configuration: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return

    if not config:
        print(f"Error: Configuration file '{config_path}' is empty or invalid.")
        return

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
    setattr(args, "config_path", os.path.abspath(config_path))

    for attr in ("data_path", "log_path", "unlearning_save_path", "pretrain_save_path"):
        val = getattr(args, attr, None)
        if isinstance(val, str) and not os.path.isabs(val):
            setattr(args, attr, os.path.normpath(os.path.join(PROJECT_ROOT, val)))

    if not getattr(args, "data_name", None):
        print("Configuration must supply run.dataset (data_name).")
        return
    if not getattr(args, "unlearning_method", None):
        setattr(args, "unlearning_method", run_cfg.get("method", "SGKU"))

    # Provide safe defaults for optional knobs that are required by loaders/trainers.
    if not hasattr(args, "neg_ratio"):
        setattr(args, "neg_ratio", 15)
    if not hasattr(args, "random_policy"):
        setattr(args, "random_policy", "schema")
    if not hasattr(args, "valid_gap"):
        setattr(args, "valid_gap", 1)
    if not hasattr(args, "valid_metrics"):
        setattr(args, "valid_metrics", "mrr")
    if not hasattr(args, "patience"):
        setattr(args, "patience", 20)
    if not hasattr(args, "overwrite_unlearning_dir"):
        setattr(args, "overwrite_unlearning_dir", True)
    if not hasattr(args, "start_timestep"):
        setattr(args, "start_timestep", 0)
    if not hasattr(args, "prefer_success_checkpoint"):
        setattr(args, "prefer_success_checkpoint", True)

    experiment_name = config.get("name") or run_cfg.get("name") or f"{args.data_name}_{args.unlearning_method}"

    print(f"\n=== Running {experiment_name} ===")
    runner = Runner(args)
    run_start = time.time()
    report_results, training_times, timestep_records = runner.unlearning()
    run_wall_time = time.time() - run_start
    save_unlearning_results(
        args=args,
        experiment_name=experiment_name,
        parameters=hyper_cfg if isinstance(hyper_cfg, dict) else {},
        training_times=training_times,
        timestep_records=timestep_records,
        report_results=report_results,
        root_dir=PROJECT_ROOT,
        run_wall_time=run_wall_time,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SGKU experiments")
    parser.add_argument(
        "--config",
        default="/Volumes/DATI/GitHub/KGUNLEARNING/src/main/configs/sgku_fb15k-237-10_transe.yaml",
        help="Path to the experiment configuration YAML file",
    )
    cli_args = parser.parse_args()
    run_experiments(cli_args.config)
