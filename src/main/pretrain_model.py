import argparse
import os
import logging
import sys
import shutil
import time

# Enable CPU fallback for unsupported MPS ops (users can override by exporting the var explicitly).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
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
from src.runners.trainer import *
from src.runners.tester import *
from src.utilities.utilities import get_datetime, resolve_device, set_seeds
from src.utilities.persistence import save_pretrain_results, save_pretrain_checkpoint_metrics
from src.utilities.kge_factory import get_kge_model_class
class PretrainRunner():
    def __init__(self, args) -> None:
        """ 1. Set parameters, seeds, logger, paths and device """
        """ Set parameters """
        self.args = args
        self.args.begin_pretrain = True
        self.args.begin_unleanring = False
        """ Set seeds """
        set_seeds(self.args.seed)
        """ Set logger """
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        now_time = get_datetime()
        base_log_dir = getattr(self.args, "log_path", os.path.join(PROJECT_ROOT, "logs"))
        if not os.path.isabs(base_log_dir):
            base_log_dir = os.path.join(PROJECT_ROOT, base_log_dir)
        os.makedirs(base_log_dir, exist_ok=True)
        self.args.log_path = os.path.join(base_log_dir, f"{now_time}_pretrain")
        logging_file_name = f'{self.args.log_path}.log'
        os.makedirs(os.path.dirname(logging_file_name), exist_ok=True)
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger
        self.log_file = logging_file_name
        """ Set paths """
        pretrain_base = getattr(self.args, "pretrain_save_path", os.path.join(PROJECT_ROOT, "checkpoint_pretrain"))
        if not os.path.isabs(pretrain_base):
            pretrain_base = os.path.join(PROJECT_ROOT, pretrain_base)
        self.args.pretrain_save_path = os.path.join(pretrain_base, self.args.data_name, self.args.kge)
        if os.path.exists(self.args.pretrain_save_path):
            shutil.rmtree(self.args.pretrain_save_path, True)
        os.makedirs(self.args.pretrain_save_path, exist_ok=True)
        """ Set device """
        device = resolve_device(getattr(self.args, "device", None))
        self.args.device = device

        """ 2. Define data """
        self.kg = KGBaseTrainingData(self.args)
        """ 3. Define model """
        self.model, self.optimizer = self.create_model()
        self.args.logger.info(args)

    def create_model(self):
        kge_name = getattr(self.args, "kge", "transe")
        kge_class = get_kge_model_class(kge_name)
        model = kge_class(self.args, self.kg)
        model.to(self.args.device)
        optimizer = Adam(model.parameters(), lr=self.args.lr)
        return model, optimizer

    def pretrain(self):
        run_start = time.time()
        report_results = PrettyTable()
        report_results.field_names = ['Time', 'MRR', 'Hits@1', 'Hits@3', 'Hits@10']
        test_results = []
        training_times = []
        training_time = self.train()
        """ prepare result table """
        test_res = PrettyTable()
        test_res.field_names = [
            'MRR',
            'Hits@1',
            'Hits@3',
            'Hits@5',
            'Hits@10',
        ]

        """ save and load model """
        best_checkpoint = os.path.join(
            self.args.pretrain_save_path, f'model_best.tar'
        )
        self.load_checkpoint(best_checkpoint)

        """ predict """
        res = self.test()
        test_duration = res.get('duration', 0.0)
        test_time_message = f"Pretrain evaluation completed in {test_duration:.2f}s"
        print(test_time_message)
        self.args.logger.info(test_time_message)
        mr = res.get("mr")
        self.args.logger.info(
            "Pretrain test metrics -> "
            f"MRR:{round(res['mrr'] * 100, 3)}\t"
            f"MR:{'NA' if mr is None else round(float(mr), 3)}\t"
            f"Hits@1:{round(res['hits1'] * 100, 3)}\t"
            f"Hits@3:{round(res['hits3'] * 100, 3)}\t"
            f"Hits@10:{round(res['hits10'] * 100, 3)}"
        )

        run_wall_time = time.time() - run_start
        save_pretrain_results(
            args=self.args,
            training_time=training_time,
            best_validation=self.best_valid,
            test_results=res,
            log_path=self.log_file,
            root_dir=PROJECT_ROOT,
            run_wall_time=run_wall_time,
            test_duration=test_duration,
        )
        metrics_path = save_pretrain_checkpoint_metrics(
            args=self.args,
            test_results=res,
            checkpoint_dir=self.args.pretrain_save_path,
            checkpoint_name="model_best.tar",
        )
        self.args.logger.info(f"Saved pretrained baseline metrics to {metrics_path}")

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
        print("Start training: PRETRAIN ")
        self.best_valid = 0.0
        self.stop_epoch = 0

        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)
        total_epochs = int(self.args.epoch_num)
        valid_gap = int(getattr(self.args, "valid_gap", 1) or 1)
        progress_iter = (
            tqdm(range(total_epochs), desc=f"Pretraining {self.args.kge}", unit="epoch")
            if tqdm
            else range(total_epochs)
        )
        for epoch in progress_iter:
            if getattr(self.args, "debug", False) and epoch > 0:
                break
            self.args.epoch = epoch
            do_eval = (epoch % valid_gap == 0) or (epoch == total_epochs - 1)
            loss, train_time, valid_res, valid_time = trainer.run_epoch(do_eval=do_eval)
            metric_key = getattr(self.args, "valid_metrics", "mrr")
            if do_eval and isinstance(valid_res, dict) and metric_key in valid_res:
                if valid_res[metric_key] > self.best_valid:
                    self.best_valid = valid_res[metric_key]
                    self.stop_epoch = 0
                    self.save_model(is_best=True)
                else:
                    self.stop_epoch += 1
                    self.save_model()
                    if self.stop_epoch >= self.args.patience:
                        self.args.logger.info(
                            f'Early Stopping! Epoch: {epoch} Best Results: {round(self.best_valid * 100, 3)}'
                        )
                        break
            else:
                self.save_model()
            if epoch % 1 == 0:
                if do_eval and isinstance(valid_res, dict) and 'mrr' in valid_res and 'hits10' in valid_res:
                    mr = valid_res.get("mr")
                    hits1 = valid_res.get("hits1")
                    hits3 = valid_res.get("hits3")
                    message = (
                        f"Epoch:{epoch}\tLoss:{round(loss, 3)}\t"
                        f"MRR:{round(valid_res['mrr'] * 100, 3)}\t"
                        f"MR:{'NA' if mr is None else round(float(mr), 3)}\t"
                        f"Hits@1:{'NA' if hits1 is None else round(float(hits1) * 100, 3)}\t"
                        f"Hits@3:{'NA' if hits3 is None else round(float(hits3) * 100, 3)}\t"
                        f"Hits@10:{round(valid_res['hits10'] * 100, 3)}\t"
                        f"Train(s):{train_time:.2f}\tValid(s):{valid_time:.2f}\t"
                        f"Best:{round(self.best_valid * 100, 3)}"
                    )
                else:
                    message = (
                        f"Epoch:{epoch}\tLoss:{round(loss, 3)}\tTrain(s):{train_time:.2f}\t"
                        f"(validation skipped; valid_gap={valid_gap})"
                    )
                print(message)
                self.args.logger.info(message)
            if tqdm:
                progress_iter.set_postfix(
                    {
                        "loss": f"{loss:.3f}",
                        "best%": f"{self.best_valid * 100:.2f}",
                        "train_s": f"{train_time:.2f}",
                        "valid_s": f"{valid_time:.2f}",
                    }
                )
        if tqdm:
            progress_iter.close()
        end_time = time.time()
        return end_time - start_time

    def test(self):
        tester = Tester(self.args, self.kg, self.model)
        return tester.test()

    def save_model(self, is_best=False):
        checkpoint_dict = {'state_dict': self.model.state_dict()}
        checkpoint_dict['epoch_id'] = self.args.epoch
        out_tar = os.path.join(
            self.args.pretrain_save_path,
            f'checkpoint-{self.args.epoch}.tar',
        )
        torch.save(checkpoint_dict, out_tar)
        if is_best:
            best_path = os.path.join(
                self.args.pretrain_save_path, f'model_best.tar'
            )
            shutil.copyfile(out_tar, best_path)

    def load_checkpoint(self, input_file):
        if os.path.isfile(input_file):
            logging.info(f"=> loading checkpoint \'{input_file}\'")
            checkpoint = torch.load(input_file, map_location=self.args.device, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            logging.info(f'=> no checking found at \'{input_file}\'')


def read_parameters(config_file="hyperparameters.yaml"):
    """
    Reads hyperparameters from a YAML file and runs experiments.

    Args:
        config_file (str): Path to the YAML configuration file.
    """
    try:
        config_path = config_file
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return

    args = argparse.Namespace()

    defaults = config.get("defaults", {})
    for key, value in defaults.items():
        setattr(args, key, value)

    run_cfg = config.get("run", {})
    for key, value in run_cfg.items():
        if key == "dataset":
            setattr(args, "data_name", value)
        else:
            setattr(args, key, value)

    hyper_cfg = config.get("hyperparameters", {})
    for key, value in hyper_cfg.items():
        setattr(args, key, value)

    if hasattr(args, "data_path") and not os.path.isabs(args.data_path):
        args.data_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.data_path))
    if hasattr(args, "log_path") and not os.path.isabs(args.log_path):
        args.log_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.log_path))
    if hasattr(args, "pretrain_save_path") and not os.path.isabs(args.pretrain_save_path):
        args.pretrain_save_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.pretrain_save_path))
    return args


def parse_cli_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a KGE model for SGKU experiments.")
    parser.add_argument("--config", default="/Volumes/DATI/GitHub/KGUNLEARNING/src/main/configs/pretrain_fb15k-237-10_transe.yaml", help="Path to YAML config with defaults.")
    return parser.parse_args()



if __name__ == "__main__":
    cli_args = parse_cli_arguments()
    args = read_parameters(cli_args.config)
    if args is None:
        sys.exit(1)
    setattr(args, "config_path", os.path.abspath(cli_args.config))
    if hasattr(args, "data_path") and not os.path.isabs(args.data_path):
        args.data_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.data_path))
    if hasattr(args, "log_path") and not os.path.isabs(args.log_path):
        args.log_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.log_path))
    if hasattr(args, "pretrain_save_path") and not os.path.isabs(args.pretrain_save_path):
        args.pretrain_save_path = os.path.normpath(os.path.join(PROJECT_ROOT, args.pretrain_save_path))
    runner = PretrainRunner(args)
    runner.pretrain()
