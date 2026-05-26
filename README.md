# GB-Phys TrajNet: Goal-Based Physics-Informed Trajectory Prediction for Autonomous Driving

A two-stage multi-modal trajectory prediction model combining goal-conditioned planning with differentiable bicycle kinematics, designed for the Waymo Open Motion Dataset.

## Environment Setup

### Option A: Conda (Recommended for HPC)

Setup Anaconda by following the instructions provided by Zaratan (we defaulted to run venv on start).
Ensure you edit your conda installation path across all scripts, including this setup script (`/experiments/setup_env.sh`)

```bash
bash experiments/setup_env.sh --force --cuda cu121
```

This creates a `waymo-trajnet` conda environment with all dependencies installed in the correct order.

### Option B: pip (Not recommended unless nothing else works)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Note: jaxlib requires special installation from the JAX GCS bucket:
pip install jaxlib==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install jax==0.4.13 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Note: typing_extensions must be force-installed to bypass TF metadata conflict:
pip install typing_extensions==4.12.2 --force-reinstall --no-deps
# PyTorch: install with correct CUDA version from https://pytorch.org/get-started/
```

## Reproducing Results

### Google Cloud Authentication (Required)

```bash
gcloud auth application-default login --no-launch-browser
export GOOGLE_APPLICATION_CREDENTIALS="${HOME}/.config/gcloud/application_default_credentials.json"
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
```

### Full Pipeline (Training + Evaluation + Visualizations)

```bash
# SLURM (HPC with A100 GPU):
sbatch experiments/run_final_notebook.slurm

# Or run directly:
python -u experiments/run_final_pipeline.py \
    --num_shards 50 --device cuda \
    --checkpoint_dir results/checkpoints \
    --log_dir results/logs \
    --submission_dir results/submissions
```

### Phase-by-Phase Training

```bash
# Phase 1: Goal prediction (15 epochs, lr=5e-4)
# Phase 2: Trajectory generation (20 epochs, lr=2e-4)
# Phase 3: End-to-end fine-tuning (15 epochs, lr=1e-4)
sbatch experiments/run_remaining_pipeline.slurm  # starts from Phase 2
```

### Evaluation Only (using existing checkpoint)

```bash
python -u experiments/run_final_pipeline.py \
    --skip_training --device cuda \
    --checkpoint_dir results/checkpoints \
    --log_dir results/logs \
    --submission_dir results/submissions
```

## Expected Runtime and Hardware

| Phase | Hardware | Time |
|-------|----------|------|
| Full pipeline (50 shards) | 1x NVIDIA A100 (HPC) | ~14-20 hours |
| Evaluation only | 1x NVIDIA A100 (HPC) | ~1-2 hours |
| Local CPU (5 shards, debug) | Any modern CPU | ~30 min/epoch |

## Project Structure

```
project/
├── README.md               # This file
├── requirements.txt        # pip dependencies
├── environment.yml         # Conda environment spec
├── .gitignore
├── data/                   # Data access instructions (GCS streaming)
│   └── README.md           # How to authenticate and access WOMD
├── src/                    # Source code
│   ├── data/               # Data loading, parsing, preprocessing
│   ├── models/             # GBPhysTrajNet, encoders, decoders, baselines
│   ├── losses/             # WTA, goal CE, physics reg, boundary, combined
│   ├── training/           # Trainer, checkpointing
│   └── evaluation/         # Metrics, submission generation
├── experiments/            # Scripts to reproduce results
│   ├── run_final_pipeline.py        # Main training + eval script
│   ├── run_final_notebook.slurm     # SLURM: full pipeline
│   ├── run_remaining_pipeline.slurm # SLURM: resume from Phase 2
│   ├── train.slurm                  # SLURM: single-phase training
│   ├── eval.slurm                   # SLURM: evaluation only
│   └── setup_env.sh                 # Environment setup script
├── results/                # Generated outputs (gitignored)
│   ├── checkpoints/        # Model checkpoints (.pt files)
│   ├── logs/               # Training logs, visualizations (.png)
│   └── submissions/        # Waymo challenge submission files
└── configs/
    └── default.yaml        # All hyperparameters
```

## Reproducibility

- Random seed: `42` (set in all scripts via `--seed 42`)
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- CUDA: `torch.cuda.manual_seed_all(42)`
- Training data: 50 shards of Waymo Open Motion Dataset v1.3.1 (GCS streamed)
- Validation: Full 150-shard validation set
- Model: ~1.2M parameters (see `src/models/gbphys_trajnet.py`)

## Key Results

| Method | minADE (m) | minFDE (m) | Miss Rate |
|--------|-----------|-----------|-----------|
| Constant Velocity | 9.58 | 26.13 | 0.946 |
| GB-Phys TrajNet | 11.15 | 30.67 | 0.963 |

Trained on 5% of available data (50/1000 shards).

## References and Attribution

### Waymo Open Dataset Tutorials

This project is built directly on top of the official Waymo Open Dataset tutorial
notebooks, which are part of the
[`waymo-research/waymo-open-dataset`](https://github.com/waymo-research/waymo-open-dataset)
repository. The following tutorials were used as primary references:

| Tutorial Notebook | How it was used in this project |
|---|---|
| **`tutorial_motion.ipynb`** | Primary reference. The WOMD TFRecord feature schema, `_parse()` data loading function, `SimpleModel` architecture, `MotionMetrics` class, `_default_metrics_config()`, and `train_step()` training loop were all adapted from this notebook. |
| **`tutorial_v2.ipynb`** | Reference for v2 Parquet-based dataset format (alternative to TFRecord). |
| **`tutorial_local.ipynb`** | Reference for local TFRecord loading patterns used in `src/data/loader.py`. |

#### Specific code derived from `tutorial_motion.ipynb`

- **`src/data/feature_defs.py`** — The complete WOMD TFRecord feature schema
  (roadgraph, state, and traffic light field definitions with shapes and dtypes)
  is copied directly from `tutorial_motion.ipynb`.

- **`src/data/loader.py`** — `parse_womd_example()` extends the tutorial's
  `_parse()` function with additional features: full 7-feature state vectors,
  roadgraph context (xyz, dir, valid, type), and traffic light states.

- **`src/evaluation/metrics.py`** — `MotionMetrics` and `_default_metrics_config()`
  are adapted from `tutorial_motion.ipynb`. The proto config (track steps, prediction
  steps, speed bounds, per-horizon miss thresholds, `max_predictions=6`) is taken
  verbatim and wrapped in a `tf.keras.metrics.Metric` subclass.

- **`src/training/trainer.py`** — The three-phase training loop structure follows
  the `train_step()` pattern from `tutorial_motion.ipynb`, extended with mixed
  precision (`torch.amp`), gradient accumulation (×4 steps), and phase-based
  parameter freezing.

### Dataset

> **Scalability in Perception for Autonomous Driving: Waymo Open Dataset.**
> *CVPR 2020.* https://arxiv.org/abs/1912.04838

> **Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset.**
> *ICCV 2021.* https://arxiv.org/abs/2104.10133

### Related Architecture References

The following works informed the design of GB-Phys TrajNet:

- **Map-Adaptive Goal-Based Trajectory Prediction** — Zhang, L. et al. (2020). Inspired
  the two-stage goal-then-trajectory architecture used in Stage 1 + Stage 2.
  https://arxiv.org/abs/2009.04450

- **PointNet** — Qi, C.R. et al. (2017). The `RoadContextEncoder` in
  `src/models/encoders.py` uses a PointNet-style MLP with max-pooling over the
  30,000-point roadgraph.
  https://arxiv.org/abs/1612.00593

- **Bicycle kinematic model** — https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html. 
  The differentiable Euler-integration bicycle model in
  `src/models/bicycle_model.py` implements the standard front-wheel steering
  kinematic bicycle model with clamped controls.

---

## Authors

- Sahil Mehta (smehta22@terpmail.umd.edu) & Rishith Vemireddy (rishiv@terpmail.umd.edu)
- University of Maryland, College Park
