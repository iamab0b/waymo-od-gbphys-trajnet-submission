#!/bin/bash
#SBATCH --job-name=gbphys-remaining
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j_remaining_pipeline.out
#SBATCH --error=logs/%j_remaining_pipeline.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smehta22@terpmail.umd.edu

# ---------------------------------------------------------------------------
# GB-Phys TrajNet  --  Resume pipeline from Phase 2 onwards
#
# Use this when Phase 1 completed successfully but the job died before
# Phases 2/3 or evaluation.  Loads Phase 1 best_minADE.pt automatically.
#
# Usage:
#   sbatch experiments/run_remaining_pipeline.slurm
#
# To start from Phase 3 instead:
#   sbatch --export=START_PHASE=3 experiments/run_remaining_pipeline.slurm
# ---------------------------------------------------------------------------

# ── Environment ──────────────────────────────────────────────────────────────
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.config/gcloud/application_default_credentials.json"

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

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/smehta22/scratch.cmsc472/waymo-od-gbphys-trajnet-final"
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

# ── Parameters ────────────────────────────────────────────────────────────────
START_PHASE="${START_PHASE:-2}"
NUM_SHARDS="${NUM_SHARDS:-50}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/checkpoints}"
LOG_DIR="${LOG_DIR:-results/logs}"

mkdir -p results/checkpoints results/logs results/submissions

echo "======================================================"
echo " GB-Phys TrajNet — Remaining Pipeline"
echo " SLURM Job:     ${SLURM_JOB_ID:-local}"
echo " Node:          ${SLURM_JOB_NODELIST:-$(hostname)}"
echo " GPUs:          $(nvidia-smi -L 2>/dev/null | wc -l)"
echo " Starting from: Phase ${START_PHASE}"
echo " Shards:        ${NUM_SHARDS}"
echo " Checkpoint:    ${CHECKPOINT_DIR}"
echo "======================================================"
echo ""

python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')"

# Confirm Phase 1 checkpoint exists before starting
if [[ ! -f "${CHECKPOINT_DIR}/phase1/best_minADE.pt" ]]; then
    echo "ERROR: ${CHECKPOINT_DIR}/phase1/best_minADE.pt not found."
    echo "Phase 1 must complete before running this script."
    exit 1
fi
echo "Phase 1 checkpoint found: ${CHECKPOINT_DIR}/phase1/best_minADE.pt"

# ── Run the remaining pipeline ─────────────────────────────────────────────
# To skip training entirely and only run evaluation, add --skip_training:
#   python -u experiments/run_final_pipeline.py ... --skip_training
python -u experiments/run_final_pipeline.py \
    --num_shards        "${NUM_SHARDS}" \
    --device            "${DEVICE}" \
    --checkpoint_dir    "${CHECKPOINT_DIR}" \
    --log_dir           "${LOG_DIR}" \
    --submission_dir    results/submissions \
    --start_phase       "${START_PHASE}"

echo ""
echo "======================================================"
echo " Remaining pipeline complete — exit code $?"
echo " Checkpoints: ${CHECKPOINT_DIR}/"
echo " Logs:        ${LOG_DIR}/"
echo " Submission:  results/submissions/"
echo "======================================================"
