# SGKU: Schema-Guided Knowledge Unlearning

This repository implements **Schema-Guided Knowledge Unlearning (SGKU)** and related baselines for knowledge-graph embedding (KGE) models. It provides an end-to-end pipeline to pretrain a KGE model, construct forget sets, run unlearning, and evaluate retain/forget performance on several standard KGs.

**What this repo can do**
- Pretrain KGE models (TransE, DistMult, RotatE, ComplEx) on KG datasets.
- Build **model-specific forget sets** from ranked triples.
- Run **SGKU** and **SDKU** unlearning methods and compare with **retrain** baselines.
- Evaluate checkpoints on retain/forget splits and store metrics.
- Run hyperparameter sweeps for SGKU/SDKU.

## Repository Layout
- `src/`
- `src/main/`: CLI entrypoints for pretraining, SGKU/SDKU, evaluation, and sweeps.
- `src/model/`: SGKU/SDKU implementations and KGE models.
- `src/loading/`: dataset loading and schema utilities.
- `data/`: datasets, schema stores, timesteps, and download/prep scripts.
- `scripts/`: ready-to-run end-to-end pipelines per dataset.
- `logs/`, `results/`, `checkpoint_*`: run outputs (created during execution).

## Requirements
- Python 3.9+
- PyTorch (CUDA optional)
- PyYAML, NumPy
- Optional: `tqdm` and `prettytable` for nicer progress/tables

Example install (adjust torch to your CUDA setup):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch pyyaml numpy tqdm prettytable
```

## Quick Start (One Script)
Run a full SGKU pipeline for a dataset/model:
```bash
bash scripts/fb15k-237-10_transe_sgku.sh
```

You can override key settings:
```bash
DEVICE=cpu TIMESTEPS_NUM=3 TIMESTEPS_PCT=10 FORGET_KEEP_FRACTION=0.25 \
  bash scripts/fb15k-237-10_transe_sgku.sh
```

Other ready scripts:
- `scripts/wn18rr-10_transe_sgku.sh`
- `scripts/wn18rr-20_transe_sgku.sh`
- `scripts/fb15k-237-20_transe_sgku.sh`
- `scripts/codex-s-20_transe_sgku.sh`
- `scripts/codex-m-20_transe_sgku.sh`
- `scripts/codex-l-20_transe_sgku.sh`
- `scripts/nell-995-20_transe_sgku.sh`

## Data Preparation
Datasets live in `data/<DATASET>/` and should contain:
- `triples.txt`
- `timesteps/` (files `0.txt`, `1.txt`, ...)
- `schema_store.json` (or enough metadata to generate it)

Generate timesteps:
```bash
python3 data/generate_timesteps.py --dataset FB15k-237-10 --steps 3 --percentage 10
```

Generate schema store:
```bash
python3 data/generate_schema_store.py --dataset FB15k-237-10
```

Download scripts (if data folders are missing):
```bash
python3 data/download_codex.py
python3 data/download_nell995.py
```

## Running SGKU/SDKU Manually
All runs are driven by YAML config files in `src/main/configs/`.

### 1) Pretrain a KGE model
Create a config with `run.method: pretrain` and run:
```bash
python3 src/main/pretrain_model.py --config src/main/configs/fb15k-237-10_transe.yaml
```

### 2) Build the forget set
This uses the pretrained model to select triples to forget:
```bash
python3 src/main/build_forget_set.py \
  --config src/main/configs/fb15k-237-10_transe.yaml \
  --rank-threshold 2 \
  --keep-fraction 0.25 \
  --suffix transe
```

### 3) Run SGKU (or SDKU / Retrain)
Set `run.method` in the YAML to `SGKU`, `SDKU`, or `retrain`:
```bash
python3 src/main/main.py --config src/main/configs/fb15k-237-10_transe.yaml
```

## Evaluate a Checkpoint
```bash
python3 src/main/eval_checkpoint.py \
  --config src/main/configs/fb15k-237-10_transe.yaml \
  --checkpoint path/to/checkpoint.tar \
  --timestep 0
```

## Sweeps
Hyperparameter sweeps live in:
- `src/main/sweep_sgku.py`
- `src/main/grid_sdku.py`
- `src/main/narrow_combo_sweep.py`

Each accepts a config and writes the best configs/checkpoints to the output folders in your YAML.

## Outputs
Paths are controlled by config defaults (`defaults` section):
- `logs/`: run logs
- `checkpoint_pretrain/`: pretrained models
- `checkpoint_unlearning/`: SGKU/SDKU checkpoints
- `checkpoint_retrain_baseline/`: retrain baselines
- `results/`: saved metrics (JSON/CSV)

## Notes
- The CLI defaults in `src/main/main.py` and `src/main/pretrain_model.py` reference an external path. Always pass `--config` to point to configs in this repo.
- Device selection comes from `defaults.device` in the YAML or `--device` for evaluation.
- To use CPU, set `defaults.device: cpu` or export `DEVICE=cpu` before running scripts.

