#!/usr/bin/env bash
# E2E V9 inference — queryctx mode
# Usage:
#   bash run_infer.sh
#   CHECKPOINT=/path/to/model.pth MASK_SEGY=/path/to/mask.sgy bash run_infer.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ---- GPU ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-29502}"

# ---- Model checkpoint ----
CHECKPOINT="${CHECKPOINT:-/home/chengzhitong/5d_regular/Regformer_5d/resultsE2E_V2/e2e_v9_datatype_df_field1031_5d_queryctx/checkpoints/model-1650.pth}"

# ---- Data ----
H5_DIR="${H5_DIR:-/data/shared/测试数据/h5}"
H5_IRREGULAR="${H5_IRREGULAR:-${H5_DIR}/field1031_irregular.h5}"
H5_REGULAR="${H5_REGULAR:-${H5_DIR}/field1031_label.h5}"
H5_MASK="${H5_MASK:-${H5_DIR}/field1031_mask.h5}"
H5_TGT="${H5_TGT:-}"
MASK_SEGY="${MASK_SEGY:-/data/shared/测试数据/mask_from_label.sgy}"
DATASET_NEIGHBORS_INFER="${DATASET_NEIGHBORS_INFER:-/data/shared/测试数据/h5/anchor_patch_e2ev1/infer_query_context.npz}"
LABEL_SEGY="${LABEL_SEGY:-/data/shared/测试数据/reg_pku_1031/reg_pku_1030/reg5dbin_label1031.sgy}"

# ---- Output ----
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gen_fill_results}"
OUTPUT_SEGY="${OUTPUT_SEGY:-${OUTPUT_DIR}/filled_missing.sgy}"
OUTPUT_RESIDUAL_SEGY="${OUTPUT_RESIDUAL_SEGY:-${OUTPUT_DIR}/residual.sgy}"

# ---- Inference params ----
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-156}"
TIME_PS="${TIME_PS:-1251}"
TRACE_PS="${TRACE_PS:-128}"
HEADER_MODE="${HEADER_MODE:-fixed}"
USE_P_SCALE="${USE_P_SCALE:-false}"
CHUNK_LENGTH="${CHUNK_LENGTH:-256}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.125}"
FILL_INTERVAL="${FILL_INTERVAL:-0}"
PRED_CLAMP="${PRED_CLAMP:-}"

# ---- Model params (auto-overridden from checkpoint training_config when present) ----
MODEL_TYPE="${MODEL_TYPE:-e2e_encdec_v9}"
D_MODEL="${D_MODEL:-768}"
N_HEADS="${N_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-6}"
D_FF="${D_FF:-2048}"
USE_LOCAL_QUERY_ATTENTION="${USE_LOCAL_QUERY_ATTENTION:-true}"
QUERY_LOCAL_K="${QUERY_LOCAL_K:-8}"
QUERY_LOCAL_SAME_TIME="${QUERY_LOCAL_SAME_TIME:-true}"
USE_QUERY_REFINEMENT="${USE_QUERY_REFINEMENT:-true}"
REFINE_QUERY_K="${REFINE_QUERY_K:-8}"
REFINE_CONTEXT_K="${REFINE_CONTEXT_K:-16}"
REFINE_GAMMA_INIT="${REFINE_GAMMA_INIT:-0.0}"
ROPE_FREQ_MODE="${ROPE_FREQ_MODE:-default}"
LAMBDA_PHYS_X="${LAMBDA_PHYS_X:-auto}"
LAMBDA_PHYS_Y="${LAMBDA_PHYS_Y:-auto}"
ROPE_NYQUIST_SAFETY="${ROPE_NYQUIST_SAFETY:-1.0}"

VISUALIZE="${VISUALIZE:-true}"
VIS_BATCHES="${VIS_BATCHES:-1}"
SORT_SEGY="${SORT_SEGY:-true}"

# ---- SEG-Y config preset ----
SEGY_CONFIG="${SEGY_CONFIG:-field1031}"

mkdir -p "${OUTPUT_DIR}"

shared_args=(
  --checkpoint "${CHECKPOINT}"
  --segy_config "${SEGY_CONFIG}"
  --h5_irregular "${H5_IRREGULAR}"
  --h5_regular "${H5_REGULAR}"
  --h5_mask "${H5_MASK}"
  --mask_path "${MASK_SEGY}"
  --dataset_neighbors_infer "${DATASET_NEIGHBORS_INFER}"
  --output_dir "${OUTPUT_DIR}"
  --output_segy "${OUTPUT_SEGY}"
  --output_residual_segy "${OUTPUT_RESIDUAL_SEGY}"
  --batch_size "${BATCH_SIZE}"
  --time_ps "${TIME_PS}"
  --trace_ps "${TRACE_PS}"
  --chunk_length "${CHUNK_LENGTH}"
  --overlap_ratio "${OVERLAP_RATIO}"
  --header_mode "${HEADER_MODE}"
  --model_type "${MODEL_TYPE}"
  --d_model "${D_MODEL}"
  --n_heads "${N_HEADS}"
  --num_layers "${NUM_LAYERS}"
  --d_ff "${D_FF}"
  --use_local_query_attention "${USE_LOCAL_QUERY_ATTENTION}"
  --query_local_k "${QUERY_LOCAL_K}"
  --query_local_same_time "${QUERY_LOCAL_SAME_TIME}"
  --use_query_refinement "${USE_QUERY_REFINEMENT}"
  --refine_query_k "${REFINE_QUERY_K}"
  --refine_context_k "${REFINE_CONTEXT_K}"
  --refine_gamma_init "${REFINE_GAMMA_INIT}"
  --rope_freq_mode "${ROPE_FREQ_MODE}"
  --lambda_phys_x "${LAMBDA_PHYS_X}"
  --lambda_phys_y "${LAMBDA_PHYS_Y}"
  --rope_nyquist_safety "${ROPE_NYQUIST_SAFETY}"
  --use_p_scale "${USE_P_SCALE}"
  --fill_interval "${FILL_INTERVAL}"
  --visualize "${VISUALIZE}"
  --vis_batches "${VIS_BATCHES}"
  --sort_segy "${SORT_SEGY}"
  --strict_fill
)

if [[ -n "${LABEL_SEGY}" ]]; then
  shared_args+=(--label_segy "${LABEL_SEGY}")
fi
if [[ -n "${H5_TGT}" ]]; then
  shared_args+=(--h5_tgt "${H5_TGT}")
fi
if [[ -n "${PRED_CLAMP}" ]]; then
  shared_args+=(--pred_clamp "${PRED_CLAMP}")
fi

echo "============================================================"
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "E2E V9 Inference — queryctx (${NUM_GPUS} GPUs)"
else
  echo "E2E V9 Inference — queryctx (single GPU)"
fi
echo "checkpoint:    ${CHECKPOINT}"
echo "H5_DIR:        ${H5_DIR}"
echo "h5_irregular:  ${H5_IRREGULAR}"
echo "h5_regular:    ${H5_REGULAR}"
echo "h5_mask:       ${H5_MASK}"
echo "h5_tgt:        ${H5_TGT:-<regular>}"
echo "mask_segy:     ${MASK_SEGY}"
echo "neighbors_npz: ${DATASET_NEIGHBORS_INFER}"
echo "output_segy:   ${OUTPUT_SEGY}"
echo "device:        ${DEVICE}"
echo "batch_size:    ${BATCH_SIZE}"
echo "fill_interval: ${FILL_INTERVAL}"
echo "pred_clamp:    ${PRED_CLAMP:-<none>}"
echo "model_type:    ${MODEL_TYPE}"
echo "v10 local:     attn=${USE_LOCAL_QUERY_ATTENTION} k=${QUERY_LOCAL_K} refine=${USE_QUERY_REFINEMENT} q/context=${REFINE_QUERY_K}/${REFINE_CONTEXT_K}"
echo "rope_freq:     ${ROPE_FREQ_MODE} (${LAMBDA_PHYS_X}/${LAMBDA_PHYS_Y})"
echo "chunk_length:  ${CHUNK_LENGTH}"
echo "overlap_ratio: ${OVERLAP_RATIO}"
echo "visualize:     ${VISUALIZE}"
echo "segy_config:   ${SEGY_CONFIG}"
echo "============================================================"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT_DIR}/infer_cli.py" \
    --device "${DEVICE}" \
    "${shared_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/infer.stdout.log"
else
  "${PYTHON_BIN}" "${ROOT_DIR}/infer_cli.py" \
    --device "${DEVICE}" \
    "${shared_args[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/infer.stdout.log"
fi

echo "Inference done!"
