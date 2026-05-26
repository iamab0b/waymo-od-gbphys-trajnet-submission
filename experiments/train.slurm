#!/bin/bash
#SBATCH --job-name=gbphys-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smehta22@terpmail.umd.edu

# ---------------------------------------------------------------------------
# GB-Phys TrajNet  –  SLURM training script
# ---------------------------------------------------------------------------
# Usage (single-phase example):
#   sbatch --export=PHASE=1,CHECKPOINT_DIR=results/checkpoints/phase1 experiments/train.slurm
#
# Phase 1: Stage-1 goal prediction  (5-10 epochs, lr=5e-4)
# Phase 2: Stage-2 GRU decoder      (10-15 epochs, lr=1e-3, frozen Stage-1)
# Phase 3: End-to-end fine-tuning   (5-10 epochs, lr=1e-4)
# ---------------------------------------------------------------------------

# ── Environment ──────────────────────────────────────────────────────────────
if command -v module &>/dev/null; then
    echo "Loading CUDA 12.3.0 module..."
    module load cuda-new/zen2/12.3.0 2>/dev/null || module load cuda/12.3.0 2>/dev/null || true
    echo "  CUDA module load attempted."
fi

# Activate conda environment by prepending its bin to PATH.
CONDA_ENV="/home/smehta22/scratch.cmsc472/conda/miniforge3-26.1.1/envs/waymo-trajnet"
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export CONDA_DEFAULT_ENV="waymo-trajnet"

# Confirm the right Python is active — catches misconfigured environments early
echo "Python: $(which python)"
echo "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/smehta22/scratch.cmsc472/waymo-od-gbphys-trajnet-final"
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

PHASE="${PHASE:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/checkpoints/phase${PHASE}}"
LOG_DIR="results/logs/tensorboard/phase${PHASE}"
DATA_DIR="${DATA_DIR:-}"          # leave empty to stream from GCS
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"   # 'train' or 'train_20s'
VAL_SPLIT="${VAL_SPLIT:-val}"         # 'val' or 'val_interactive'

# --- GCS paths (tf_example format — all available splits) ---
# Training: 1000 shards
GCS_TRAIN="${GCS_TRAIN:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-*-of-01000}"
# Validation: 150 shards
GCS_VAL="${GCS_VAL:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord-*-of-00150}"
# Testing: 150 shards (no GT, for submission only)
GCS_TEST="${GCS_TEST:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing/testing_tfexample.tfrecord-*-of-00150}"
# Interactive splits (for interactive prediction challenge variant)
GCS_VAL_INTERACTIVE="${GCS_VAL_INTERACTIVE:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-*-of-00150}"
GCS_TEST_INTERACTIVE="${GCS_TEST_INTERACTIVE:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing_interactive/testing_interactive_tfexample.tfrecord-*-of-00150}"

# ---- Phase-specific hyper-parameters ---------------------------------------
case "${PHASE}" in
  1)
    LR=5e-4
    NUM_EPOCHS=10
    RESUME_FROM="${RESUME_FROM:-}"
    ;;
  2)
    LR=1e-3
    NUM_EPOCHS=15
    RESUME_FROM="${RESUME_FROM:-results/checkpoints/phase1/best_val_loss.pt}"
    ;;
  3)
    LR=1e-4
    NUM_EPOCHS=10
    RESUME_FROM="${RESUME_FROM:-results/checkpoints/phase2/best_val_loss.pt}"
    ;;
  *)
    echo "Invalid PHASE=${PHASE}" >&2
    exit 1
    ;;
esac

# ---- Derived args ----------------------------------------------------------
RESUME_ARG=""
if [[ -n "${RESUME_FROM}" && -f "${RESUME_FROM}" ]]; then
  RESUME_ARG="--resume_from ${RESUME_FROM}"
fi

DATA_ARG=""
if [[ -n "${DATA_DIR}" ]]; then
  DATA_ARG="--data_dir ${DATA_DIR}"
fi

mkdir -p results/checkpoints results/logs results/submissions

echo "======================================================"
echo " GB-Phys TrajNet   Phase ${PHASE} training"
echo " SLURM Job: ${SLURM_JOB_ID}"
echo " Nodes: ${SLURM_JOB_NODELIST:-local}"
echo " Checkpoint dir: ${CHECKPOINT_DIR}"
echo " Log dir: ${LOG_DIR}"
echo "======================================================"

# ---- Launch training -------------------------------------------------------
python -u src/training/trainer.py \
  --phase              "${PHASE}" \
  --num_epochs         "${NUM_EPOCHS}" \
  --lr                 "${LR}" \
  --batch_size         32 \
  --accumulation_steps 4 \
  --use_mixed_precision \
  --augment \
  --agent_embedding_dim 128 \
  --road_embedding_dim  128 \
  --gru_hidden_dim      256 \
  --gru_layers          2 \
  --K                   6 \
  --N_candidates        64 \
  --lambda_goal         1.0 \
  --lambda_wta          1.0 \
  --lambda_physics      0.1 \
  --lambda_boundary     0.5 \
  --checkpoint_dir      "${CHECKPOINT_DIR}" \
  --log_dir             "${LOG_DIR}" \
  --log_every           50 \
  --save_every          500 \
  --eval_every          1000 \
  --gcs_train           "${GCS_TRAIN}" \
  --gcs_val             "${GCS_VAL}" \
  --gcs_test            "${GCS_TEST}" \
  --train_split         "${TRAIN_SPLIT}" \
  --val_split           "${VAL_SPLIT}" \
  --device              cuda \
  --seed                42 \
  ${DATA_ARG} \
  ${RESUME_ARG}

echo "Training complete exit code $?"
