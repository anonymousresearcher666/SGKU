import time

from src.model.model_training import *


class Trainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.logger = args.logger
        self.train_processor = UnlTrainBatch(args, kg)
        self.valid_processor = DBatching(args, kg)
        self.optimizer = optimizer


    def run_epoch(self, do_eval: bool = True):
        self.args.valid = True
        loss, train_time = self.train_processor.process_epoch(self.model, self.optimizer)
        valid_time = 0.0
        res = {}
        if do_eval:
            total_batches = None
            total_samples = None
            try:
                total_batches = len(self.valid_processor.data_loader)
            except Exception:
                total_batches = None
            try:
                total_samples = len(self.valid_processor.data_loader.dataset)
            except Exception:
                total_samples = None
            msg = "Validation starting"
            if total_samples is not None:
                msg += f" | samples={total_samples}"
            if total_batches is not None:
                msg += f" | batches={total_batches}"
            self.logger.info(msg)
            print(msg)
            valid_start = time.time()
            res = self.valid_processor.process_epoch(self.model)
            valid_time = time.time() - valid_start
            res['duration'] = valid_time
            done_msg = f"Validation finished in {valid_time:.2f}s"
            self.logger.info(done_msg)
            print(done_msg)
        self.args.valid = False
        return loss, train_time, res, valid_time


class UnlearningTrainer():
    def __init__(self, args, kg, model, optimizer) -> None:
        self.args = args
        self.kg = kg
        self.model = model
        self.optimizer = optimizer
        self.logger = args.logger
        if self.args.unlearning_method in ["pretrain","retrain", "finetune"]:
            self.train_processor = UnlTrainBatch(args, kg)
        elif self.args.unlearning_method in ["SGKU", "SDKU"]:
            # Paper-aligned SGKU/GRPO training: retain-vs-forget batches + group-softmax GRPO.
            self.train_processor = SGKUPaperBatching(args, kg)
        elif self.args.unlearning_method == "SGKUIntuitor":
            # Keep legacy batching for the intuitor variant (can be upgraded later).
            self.train_processor = SGKUBatching(args, kg)
        else:
            raise NotImplementedError
        #
        ##Test how well the model has **forgotten** knowledge
        self.run_forget_valid_processor = ForgetDBatching(args, kg)
        ##Test how well the model has **preserved** knowledge
        self.retain_valid_processor = RetainDBatching(args, kg)

        # print("forget_valid_processor ", len(self.run_forget_valid_processor.data_loader)) ###
        # print("reserve_valid_processor ", len(self.reserve_valid_processor.data_loader)) ##

    def evaluate_model(self, model, forget_processor, reserve_processor):
        """Evaluate model on both forget and reserve datasets."""
        # Run evaluations separately
        forget_results = forget_processor.process_epoch(model)
        retain_results = reserve_processor.process_epoch(model)

        # Calculate unlearning metrics
        unlearning_metrics = calculate_unlearning_metrics(
            forget_processor=forget_processor,
            reserve_processor=reserve_processor
        )

        # You can save or return the metrics as needed
        return {
            'forget_results': forget_results,
            'retain_results': retain_results,
            'unlearning_metrics': unlearning_metrics,
            'evaluation_time': forget_results.get('duration', 0.0) + retain_results.get('duration', 0.0)
        }

    def run_epoch(self):
        self.args.valid = True
        loss, train_time = self.train_processor.process_epoch(self.model, self.optimizer)
        eval_time = None
        forget_results = None
        retain_results = None
        if self.args.epoch % self.args.valid_gap == 0:
            # Always evaluate on the CURRENT timestep split.
            # The forget/retain datasets are constructed in the processor __init__ using
            # args.timestep_test/args.timestep_validation. If these drift (or if a previous
            # timestep left stale values), evaluation can silently use the wrong split.
            cur_ts = int(getattr(self.args, "timestep", 0) or 0)
            self.args.timestep_test = cur_ts
            self.args.timestep_validation = cur_ts
            # Rebuild eval processors so their datasets reflect the correct timestep.
            self.run_forget_valid_processor = ForgetDBatching(self.args, self.kg)
            self.retain_valid_processor = RetainDBatching(self.args, self.kg)
            self.logger.info(
                f"[timestep {cur_ts}] Validation split -> timestep_test={self.args.timestep_test}, "
                f"timestep_validation={self.args.timestep_validation}"
            )

            eval_start = time.time()
            results = self.evaluate_model(self.model, self.run_forget_valid_processor, self.retain_valid_processor)
            eval_time = results.get('evaluation_time', time.time() - eval_start)
            forget_results = results['forget_results']
            retain_results = results['retain_results']
            baseline = getattr(self.args, "pretrain_unlearning_baseline", None)
            if isinstance(baseline, dict):
                timestep_key = int(getattr(self.args, "timestep_test", getattr(self.args, "timestep", 0)) or 0)
                base_retain = (baseline.get(timestep_key) or {}).get("retain") or {}
                base_mrr = base_retain.get("mrr")
                base_hits10 = base_retain.get("hits10")
                cur_mrr = retain_results.get("mrr") if isinstance(retain_results, dict) else None
                cur_hits10 = retain_results.get("hits10") if isinstance(retain_results, dict) else None
                if base_mrr is not None and cur_mrr is not None:
                    msg = (
                        "Retain vs pretrained baseline (same split) -> "
                        f"ΔMRR: {float(cur_mrr) - float(base_mrr):+.4f}"
                    )
                    if base_hits10 is not None and cur_hits10 is not None:
                        msg += f", ΔHits@10: {float(cur_hits10) - float(base_hits10):+.4f}"
                    self.logger.info(msg)
        self.args.valid = False
        return loss, train_time, eval_time, forget_results, retain_results
