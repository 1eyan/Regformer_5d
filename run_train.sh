#!/bin/bash
# E2E V9 training — queryctx mode
# Usage:
#   bash run_train.sh
#   H5_FILE=/path/to/irregular.h5 DATASET_NEIGHBORS_TRAIN=/path/to/train_pool.npz bash run_train.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- GPU ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"

# ---- Training ----
MODEL_NAME="${MODEL_NAME:-e2e_v9}"
BATCH_SIZE="${BATCH_SIZE:-20}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-2500}"
SEED="${SEED:-515}"
DATA_TYPE="${DATA_TYPE:-df_field1031_5d}"
RESULTS_DIR="${RESULTS_DIR:-./resultsE2E_selfV1_block}"

# ---- Data ----
H5_DIR="${H5_DIR:-/data/shared/测试数据/h5}"
H5_FILE="${H5_FILE:-${H5_DIR}/field1031_irregular.h5}"
H5_FILE_REGULAR="${H5_FILE_REGULAR:-${H5_DIR}/field1031_label.h5}"
H5_FILE_TGT="${H5_FILE_TGT:-}"
DATASET_NEIGHBORS_TRAIN="${DATASET_NEIGHBORS_TRAIN:-/data/shared/测试数据/h5/anchor_patch_e2ev1/train_pool_idx_2d_block.npz}"
# 验证集 npz，从训练路径自动推导
_default_val_npz="${DATASET_NEIGHBORS_TRAIN/train_pool_idx_2d_block.npz/infer_query_context.npz}"
_default_val_npz="${_default_val_npz/train_pool_idx_2d.npz/infer_query_context.npz}"
_default_val_npz="${_default_val_npz/train_pool_idx_2d_block_orig.npz/infer_query_context.npz}"
DATASET_NEIGHBORS_VAL="${DATASET_NEIGHBORS_VAL:-$_default_val_npz}"
TARGET_MODE="${TARGET_MODE:-self}"

# ---- Queryctx ----
TRAIN_NUM_QUERY="${TRAIN_NUM_QUERY:-30}"
TRAIN_CONTEXT_SIZE="${TRAIN_CONTEXT_SIZE:-}"
PATCH_BETA="${PATCH_BETA:-0.5}"
FORCE_ANCHOR_QUERY="${FORCE_ANCHOR_QUERY:-false}"
# Pool count is already very large (21,668), no need to repeat
EPOCH_REPEAT="${EPOCH_REPEAT:-3}"
#TRACE_SORT_KEYS="${TRACE_SORT_KEYS:-rx}"

TIME_PS="${TIME_PS:-1251}"
TRACE_PS="${TRACE_PS:-256}"
USE_P_SCALE="${USE_P_SCALE:-false}"
CHUNK_LENGTH="${CHUNK_LENGTH:-256}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.125}"
QUERY_LOSS_WEIGHT="${QUERY_LOSS_WEIGHT:-1.0}"
CONTEXT_LOSS_WEIGHT="${CONTEXT_LOSS_WEIGHT:-0.1}"
ENERGY_LOSS_WEIGHT="${ENERGY_LOSS_WEIGHT:-0.5}"
HF_GRAD_LOSS_WEIGHT="${HF_GRAD_LOSS_WEIGHT:-0.1}"
PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.1}"
COORD_AUG_SCALE="${COORD_AUG_SCALE:-0.01}"

# ---- V9 model ----
D_MODEL="${D_MODEL:-768}"
N_HEADS="${N_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-6}"
D_FF="${D_FF:-2048}"
ROPE_FREQ_MODE="${ROPE_FREQ_MODE:-physical}"
LAMBDA_PHYS_X="${LAMBDA_PHYS_X:-auto}"
LAMBDA_PHYS_Y="${LAMBDA_PHYS_Y:-auto}"
ROPE_NYQUIST_SAFETY="${ROPE_NYQUIST_SAFETY:-1.0}"

# ---- SEG-Y config (controls key_columns, trace_sort_keys, coord_col) ----
SEGY_CONFIG="${SEGY_CONFIG:-field1031}"
export SEGY_CONFIG

echo "======================================"
echo "E2E V9 Training — queryctx"
echo "GPU: ${CUDA_VISIBLE_DEVICES}  |  Num: ${NUM_GPUS}"
echo "Model: ${MODEL_NAME}  |  Batch: ${BATCH_SIZE}  |  LR: ${LR}  |  Epochs: ${EPOCHS}"
echo "H5_DIR:     ${H5_DIR}"
echo "H5 irregular: ${H5_FILE}"
echo "H5 regular:   ${H5_FILE_REGULAR}"
echo "H5 target:    ${H5_FILE_TGT:-<regular>}"
echo "neighbors:    ${DATASET_NEIGHBORS_TRAIN}"
echo "target_mode: ${TARGET_MODE}  |  num_query: ${TRAIN_NUM_QUERY}  |  trace_ps: ${TRACE_PS}  |  beta: ${PATCH_BETA}"
echo "epoch_repeat: ${EPOCH_REPEAT}  |  chunk_length: ${CHUNK_LENGTH}  |  overlap: ${OVERLAP_RATIO}"
echo "loss: query=${QUERY_LOSS_WEIGHT} context=${CONTEXT_LOSS_WEIGHT} energy=${ENERGY_LOSS_WEIGHT} hf_grad=${HF_GRAD_LOSS_WEIGHT} phase=${PHASE_LOSS_WEIGHT}"
echo "rope_freq_mode: ${ROPE_FREQ_MODE}  |  lambda_x/y: ${LAMBDA_PHYS_X}/${LAMBDA_PHYS_Y}"
echo "segy_config: ${SEGY_CONFIG}"
echo "======================================"

cmd_args=(
  --model_name "${MODEL_NAME}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --data_type "${DATA_TYPE}"
  --results_dir "${RESULTS_DIR}"
  --use_p_scale "${USE_P_SCALE}"
  --time_ps "${TIME_PS}"
  --trace_ps "${TRACE_PS}"
  --h5File "${H5_FILE}"
  --h5File_regular "${H5_FILE_REGULAR}"
  --dataset_neighbors_train "${DATASET_NEIGHBORS_TRAIN}"
  --dataset_neighbors_test "${DATASET_NEIGHBORS_VAL}"
  --target_mode "${TARGET_MODE}"
  --train_num_query "${TRAIN_NUM_QUERY}"
  --patch_beta "${PATCH_BETA}"
  --force_anchor_query "${FORCE_ANCHOR_QUERY}"
  --epoch_repeat "${EPOCH_REPEAT}"
  --chunk_length "${CHUNK_LENGTH}"
  --overlap_ratio "${OVERLAP_RATIO}"
  --query_loss_weight "${QUERY_LOSS_WEIGHT}"
  --context_loss_weight "${CONTEXT_LOSS_WEIGHT}"
  --energy_loss_weight "${ENERGY_LOSS_WEIGHT}"
  --hf_grad_loss_weight "${HF_GRAD_LOSS_WEIGHT}"
  --phase_loss_weight "${PHASE_LOSS_WEIGHT}"
  --d_model "${D_MODEL}"
  --n_heads "${N_HEADS}"
  --num_layers "${NUM_LAYERS}"
  --d_ff "${D_FF}"
  --rope_freq_mode "${ROPE_FREQ_MODE}"
  --lambda_phys_x "${LAMBDA_PHYS_X}"
  --lambda_phys_y "${LAMBDA_PHYS_Y}"
  --rope_nyquist_safety "${ROPE_NYQUIST_SAFETY}"
  --coord_aug_scale "${COORD_AUG_SCALE}"
  #--trace_sort_keys "${TRACE_SORT_KEYS}"
  --dataset_type queryctx
)

if [[ -n "${H5_FILE_TGT}" ]]; then
  cmd_args+=(--h5File_tgt "${H5_FILE_TGT}")
fi

if [[ -n "${TRAIN_CONTEXT_SIZE}" ]]; then
  cmd_args+=(--train_context_size "${TRAIN_CONTEXT_SIZE}")
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch --config_file "${ROOT_DIR}/accelerate_config.yaml" \
    "${ROOT_DIR}/train.py" "${cmd_args[@]}"
else
  "${PYTHON_BIN}" "${ROOT_DIR}/train.py" "${cmd_args[@]}"
fi

echo "Training done!"
