#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="CoDEx-S-20"
KGE="transe"
METHOD="SGKU"
BASE_CFG="${ROOT_DIR}/src/main/configs/codex-s-20_transe.yaml"

DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TIMESTEPS_NUM="${TIMESTEPS_NUM:-3}"
TIMESTEPS_PCT="${TIMESTEPS_PCT:-10}"
FORGET_KEEP_FRACTION="${FORGET_KEEP_FRACTION:-0.25}"
FORGET_RANK_THRESHOLD="${FORGET_RANK_THRESHOLD:-2}"
FORGET_BATCH_SIZE="${FORGET_BATCH_SIZE:-512}"

export CUDA_VISIBLE_DEVICES
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TMP_DIR="${ROOT_DIR}/src/main/configs/pipeline_tmp"
mkdir -p "${TMP_DIR}"

echo "[info] dataset=${DATASET} kge=${KGE} method=${METHOD} device=${DEVICE} stamp=${STAMP}"

# 0) Prepare data (timesteps + schema)
DATA_DIR="${ROOT_DIR}/data/${DATASET}"
if [ ! -d "${DATA_DIR}/timesteps" ] || ! compgen -G "${DATA_DIR}/timesteps/[0-9]*.txt" > /dev/null; then
  echo "[prep] generating timesteps"
  python3 "${ROOT_DIR}/data/generate_timesteps.py" \
    --dataset "${DATASET}" --steps "${TIMESTEPS_NUM}" --percentage "${TIMESTEPS_PCT}"
fi
if [ ! -f "${DATA_DIR}/schema_store.json" ]; then
  echo "[prep] generating schema store"
  python3 "${ROOT_DIR}/data/generate_schema_store.py" --dataset "${DATASET}"
fi

# 1) Pretrain
PRE_CFG="${TMP_DIR}/pretrain_${DATASET//\//_}_${KGE}_${STAMP}.yaml"
python3 - "${BASE_CFG}" "${PRE_CFG}" "${DEVICE}" <<'PY'
import sys, yaml
src, dst, device = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src)) or {}
cfg.setdefault("defaults", {})
cfg["defaults"]["device"] = device
cfg.setdefault("run", {})
cfg["run"]["method"] = "pretrain"
name = cfg.get("name") or "pretrain"
cfg["name"] = f"pretrain_{name}"
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(dst)
PY
python3 "${ROOT_DIR}/src/main/pretrain_model.py" --config "${PRE_CFG}"

# 2) Build memorized forget set
python3 "${ROOT_DIR}/src/main/build_forget_set.py" \
  --config "${PRE_CFG}" \
  --input-dir timesteps \
  --rank-threshold "${FORGET_RANK_THRESHOLD}" \
  --keep-fraction "${FORGET_KEEP_FRACTION}" \
  --suffix "${KGE}" \
  --batch-size "${FORGET_BATCH_SIZE}"

# 3) Retrain baseline (exact unlearning)
RETRAIN_CFG="${TMP_DIR}/retrain_${DATASET//\//_}_${KGE}_${STAMP}.yaml"
python3 - "${BASE_CFG}" "${RETRAIN_CFG}" "${DEVICE}" <<'PY'
import sys, yaml
src, dst, device = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src)) or {}
cfg.setdefault("defaults", {})
cfg["defaults"]["device"] = device
cfg["defaults"]["unlearning_save_path"] = "checkpoint_retrain_baseline"
cfg.setdefault("run", {})
cfg["run"]["method"] = "retrain"
name = cfg.get("name") or "retrain"
cfg["name"] = f"retrain_{name}"
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(dst)
PY
python3 "${ROOT_DIR}/src/main/main.py" --config "${RETRAIN_CFG}"

# 4) SGKU unlearning (approximate)
SGKU_CFG="${TMP_DIR}/sgku_${DATASET//\//_}_${KGE}_${STAMP}.yaml"
python3 - "${BASE_CFG}" "${SGKU_CFG}" "${DEVICE}" <<'PY'
import sys, yaml
src, dst, device = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(src)) or {}
cfg.setdefault("defaults", {})
cfg["defaults"]["device"] = device
cfg.setdefault("run", {})
cfg["run"]["method"] = "SGKU"
name = cfg.get("name") or "sgku"
cfg["name"] = f"sgku_{name}"
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(dst)
PY
python3 "${ROOT_DIR}/src/main/main.py" --config "${SGKU_CFG}"

echo "[done] Pipeline complete for ${DATASET} (${KGE}/${METHOD})"
