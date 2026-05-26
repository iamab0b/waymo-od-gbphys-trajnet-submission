#!/bin/bash
#SBATCH --job-name=gbphys-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%j_final_notebook.out
#SBATCH --error=logs/%j_final_notebook.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smehta22@terpmail.umd.edu

# ---------------------------------------------------------------------------
# GB-Phys TrajNet  –  Run the entire final_model pipeline on HPC
#
# This executes experiments/run_final_pipeline.py which reproduces every section
# of final_model.ipynb as a non-interactive Python script:
#   1. Data pipeline verification
#   2. Baseline evaluation (CV + LSTM)
#   3. Three-phase GB-Phys TrajNet training (50 shards, increased epochs)
#   4. Full evaluation with official MotionMetrics
#   5. Submission generation
#
# Usage:
#   sbatch experiments/run_final_notebook.slurm
#
# Or with overrides:
#   sbatch --export=NUM_SHARDS=100,DEVICE=cuda experiments/run_final_notebook.slurm
# ---------------------------------------------------------------------------

# ── Environment ──────────────────────────────────────────────────────────────
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt

if command -v module &>/dev/null; then
    echo "Loading CUDA 12.3.0 module..."
    module load cuda-new/zen2/12.3.0 2>/dev/null || module load cuda/12.3.0 2>/dev/null || true
    echo "  CUDA module load attempted."
fi

# Activate conda environment by prepending its bin to PATH.
# Avoids 'conda activate' which calls 'return' internally and silently
# exits the entire SLURM script when run in a non-interactive shell.
CONDA_ENV="/home/smehta22/scratch.cmsc472/conda/miniforge3-26.1.1/envs/waymo-trajnet"
export PATH="${CONDA_ENV}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV}"
export CONDA_DEFAULT_ENV="waymo-trajnet"

# ── Paths ────────────────────────────────────────────────────────────────────
# $0 in SLURM points to the spool directory, not the project folder.
# Always use an explicit absolute path.
PROJECT_DIR="/home/smehta22/scratch.cmsc472/waymo-od-gbphys-trajnet-final"
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

# ── Configurable parameters ──────────────────────────────────────────────────
NUM_SHARDS="${NUM_SHARDS:-50}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/checkpoints}"
LOG_DIR="${LOG_DIR:-results/logs}"

mkdir -p results/checkpoints results/logs results/submissions

echo "======================================================"
echo " GB-Phys TrajNet — Full Pipeline"
echo " SLURM Job:     ${SLURM_JOB_ID:-local}"
echo " Node:          ${SLURM_JOB_NODELIST:-$(hostname)}"
echo " GPUs:          $(nvidia-smi -L 2>/dev/null | wc -l)"
echo " Shards:        ${NUM_SHARDS}"
echo " Device:        ${DEVICE}"
echo " Checkpoint:    ${CHECKPOINT_DIR}"
echo "======================================================"
echo ""

python -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')"

# ── Run the full pipeline ────────────────────────────────────────────────────
python -u experiments/run_final_pipeline.py \
    --num_shards     "${NUM_SHARDS}" \
    --device         "${DEVICE}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --log_dir        "${LOG_DIR}" \
    --submission_dir results/submissions

echo ""
echo "======================================================"
echo " Pipeline complete — exit code $?"
echo " Checkpoints: ${CHECKPOINT_DIR}/"
echo " Logs:        ${LOG_DIR}/"
echo " Submission:  results/submissions/"
echo "======================================================"
