#!/bin/bash
#SBATCH --job-name=gbphys-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=smehta22@terpmail.umd.edu

# ---------------------------------------------------------------------------
# GB-Phys TrajNet  –  SLURM evaluation / submission generation script
# ---------------------------------------------------------------------------
# Usage:
#   sbatch --export=CHECKPOINT=results/checkpoints/phase3/best_val_loss.pt experiments/eval.slurm
#
# To generate a submission file:
#   sbatch --export=CHECKPOINT=<path>,MODE=submit experiments/eval.slurm
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

echo "Python: $(which python)"
echo "PYTHONNOUSERSITE: ${PYTHONNOUSERSITE}"
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/smehta22/scratch.cmsc472/waymo-od-gbphys-trajnet-final"
cd "${PROJECT_DIR}"
echo "Working directory: $(pwd)"

MODE="${MODE:-eval}"          # 'eval' or 'submit'
CHECKPOINT="${CHECKPOINT:-results/checkpoints/phase3/best_val_loss.pt}"
DATA_DIR="${DATA_DIR:-}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"   # 'val', 'val_interactive', 'test', 'test_interactive'

# GCS paths for all evaluation/test splits
GCS_VAL="${GCS_VAL:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord-*-of-00150}"
GCS_VAL_INTERACTIVE="${GCS_VAL_INTERACTIVE:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-*-of-00150}"
GCS_TEST="${GCS_TEST:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing/testing_tfexample.tfrecord-*-of-00150}"
GCS_TEST_INTERACTIVE="${GCS_TEST_INTERACTIVE:-gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing_interactive/testing_interactive_tfexample.tfrecord-*-of-00150}"

# Resolve the GCS path based on EVAL_SPLIT
case "${EVAL_SPLIT}" in
  val)               EVAL_GCS="${GCS_VAL}" ;;
  val_interactive)   EVAL_GCS="${GCS_VAL_INTERACTIVE}" ;;
  test)              EVAL_GCS="${GCS_TEST}" ;;
  test_interactive)  EVAL_GCS="${GCS_TEST_INTERACTIVE}" ;;
  *)
    echo "Invalid EVAL_SPLIT=${EVAL_SPLIT}. Use: val, val_interactive, test, test_interactive" >&2
    exit 1
    ;;
esac

OUTPUT_PATH="${OUTPUT_PATH:-results/submissions/submission_${EVAL_SPLIT}.pb}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-200}"

mkdir -p results/checkpoints results/logs results/submissions

echo "======================================================"
echo " GB-Phys TrajNet   Evaluation"
echo " SLURM Job: ${SLURM_JOB_ID}"
echo " Checkpoint: ${CHECKPOINT}"
echo " Mode: ${MODE}"
echo "======================================================"

if [[ "${MODE}" == "submit" ]]; then
  # ----- Submission generation -----------------------------------------------
  TEST_DATA="${TEST_DATA:-${EVAL_GCS}}"

  python -u src/evaluation/submission.py \
    --checkpoint   "${CHECKPOINT}" \
    --test_data    "${TEST_DATA}" \
    --test_split   "${EVAL_SPLIT}" \
    --output_path  "${OUTPUT_PATH}" \
    --batch_size   "${BATCH_SIZE}" \
    --device       cuda

  echo "Submission file written to: ${OUTPUT_PATH}"
  echo "Split evaluated: ${EVAL_SPLIT}"

else
  # ----- Validation evaluation -----------------------------------------------
  DATA_ARG=""
  if [[ -n "${DATA_DIR}" ]]; then
    DATA_ARG="--data_dir ${DATA_DIR}"
  fi

  python -u - <<'PYEOF'
import sys
import torch
import argparse
import numpy as np

from src.models.gbphys_trajnet import GBPhysTrajNet
from src.data.loader           import create_local_dataset, create_gcs_dataset, tf_to_torch_batch
from src.training.checkpointing import load_checkpoint
from src.evaluation.metrics    import batch_compute_minADE, batch_compute_minFDE, compute_miss_rate

import os

checkpoint  = os.environ.get('CHECKPOINT', 'results/checkpoints/phase3/best_val_loss.pt')
eval_gcs    = os.environ.get('EVAL_GCS',
    'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/'
    'validation_tfexample.tfrecord-*-of-00150')
eval_split  = os.environ.get('EVAL_SPLIT', 'val')
data_dir    = os.environ.get('DATA_DIR', '')
batch_size  = int(os.environ.get('BATCH_SIZE', '32'))
max_batches = int(os.environ.get('MAX_VAL_BATCHES', '200'))
device_str  = 'cuda' if torch.cuda.is_available() else 'cpu'
device      = torch.device(device_str)

print(f'Evaluating checkpoint: {checkpoint}  on {device}')
print(f'Eval split: {eval_split}  GCS: {eval_gcs}')

model     = GBPhysTrajNet()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
load_checkpoint(model, optimizer, checkpoint, device=device_str)
model.to(device)
model.eval()

if data_dir:
    ds = create_local_dataset(data_dir, batch_size=batch_size,
                              shuffle_buffer=0, split=eval_split)
else:
    ds = create_gcs_dataset(eval_gcs, batch_size=batch_size, shuffle_buffer=0)

all_minADE  = []
all_minFDE  = []
all_miss    = []

with torch.no_grad():
    for idx, tf_batch in enumerate(ds.take(max_batches)):
        tb   = tf_to_torch_batch(tf_batch, device=device_str)
        out  = model(tb, phase=3)

        trajs = out['trajectories'].cpu().numpy()      # [B, 128, K, 80, 2]
        ttp   = tb['tracks_to_predict'].cpu().numpy()  # [B, 128]
        gt_f  = tb['gt_future_states'].cpu().numpy()   # [B, 128, 91, 7]
        gt_v  = tb['gt_future_is_valid'].cpu().numpy() # [B, 128, 91]

        gt_future_pos   = gt_f[:, :, 11:, 0:2]          # [B, 128, 80, 2]
        gt_future_valid = gt_v[:, :, 11:]                # [B, 128, 80]

        B, A = ttp.shape
        for b in range(B):
            for a in range(A):
                if not ttp[b, a]:
                    continue
                p = trajs[b, a]           # [K, 80, 2]
                g = gt_future_pos[b, a]   # [80, 2]
                v = gt_future_valid[b, a] # [80]
                all_minADE.append(batch_compute_minADE(p[np.newaxis], g[np.newaxis], v[np.newaxis])[0])
                all_minFDE.append(batch_compute_minFDE(p[np.newaxis], g[np.newaxis], v[np.newaxis])[0])
                all_miss.append(compute_miss_rate(p, g, v, threshold=2.0))

        if (idx + 1) % 10 == 0:
            print(f'  batch {idx+1}/{max_batches}  '
                  f'minADE={np.mean(all_minADE):.3f}  '
                  f'minFDE={np.mean(all_minFDE):.3f}  '
                  f'MR={np.mean(all_miss):.3f}')

print('\n===== Final Metrics =====')
print(f'  minADE (m) : {np.mean(all_minADE):.4f} ± {np.std(all_minADE):.4f}')
print(f'  minFDE (m) : {np.mean(all_minFDE):.4f} ± {np.std(all_minFDE):.4f}')
print(f'  Miss Rate  : {np.mean(all_miss):.4f}')
print(f'  Agents eval: {len(all_minADE)}')
PYEOF

fi

echo "Done exit code $?"
