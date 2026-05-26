# Development Notebooks

These notebooks document the iterative development process of GB-Phys TrajNet.
They are designed for local exploration on CPU (small data subsets) and are
**not** the primary training/evaluation pipeline. For full training and
reproducible results, use `experiments/run_final_pipeline.py`.

## Prerequisites

- Activate the `waymo-trajnet` conda environment (see `experiments/setup_env.sh`)
- Ensure Google Cloud authentication is configured for GCS data access
- Select the **"Waymo TrajNet (Python 3.10)"** Jupyter kernel

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Connects to GCS, parses WOMD TFRecords, inspects tensor shapes (128 agents, 11 history steps, 80 future steps, 30k road points), and visualizes scenarios with roadgraph overlays, agent trajectories, traffic light states, and type/speed distributions. |
| `02_baseline_cv.ipynb` | Implements and evaluates the Constant Velocity baseline (linear extrapolation from last observed velocity). Computes minADE, minFDE, and Miss Rate on a validation subset. Integrates with official Waymo `MotionMetrics` for per-type/per-horizon evaluation. |
| `03_baseline_lstm.ipynb` | Trains an LSTM encoder-decoder baseline with Winner-Take-All loss. Demonstrates multi-modal (K=6) trajectory prediction, checkpoint save/load, and comparison with the CV baseline. |
| `04_stage1_goal_prediction.ipynb` | Develops Stage 1 of GB-Phys TrajNet: agent history encoding (1D CNN), road context encoding (PointNet), goal candidate sampling from the roadgraph, and goal scoring via dot-product attention. Trains goal prediction with cross-entropy loss and analyzes top-K accuracy. |
| `05_stage2_trajectory_generation.ipynb` | Develops Stage 2: the GRU decoder with differentiable bicycle kinematics. Tests the BicycleKinematics model (straight line, circular arc, braking), demonstrates the autoregressive decoding loop, and trains with WTA + physics regularization losses. |
| `06_end_to_end_training.ipynb` | Combines both stages for end-to-end fine-tuning (Phase 3). Demonstrates the full `GBPhysTrajNet` model forward pass, combined loss computation (goal + WTA + physics + boundary), gradient accumulation, and cosine LR scheduling. |
| `07_evaluation_submission.ipynb` | Runs official Waymo MotionMetrics (minADE, minFDE, MissRate, OverlapRate, SoftmAP at 3s/5s/8s), generates the challenge submission protobuf file (K=6 trajectories at 2Hz, 16 points per trajectory), and validates submission format. |

## Notes

- All notebooks use `device = torch.device('cpu')` for local development. GPU training is handled by SLURM scripts in `experiments/`.
- Output directories (`results/logs/`, `results/checkpoints/`) are created relative to the project root (one level above `notebooks/`).
- Data is streamed from GCS using `tf.data`. A small subset (5-10 shards) is used for notebook exploration; full-dataset training uses 50+ shards via the pipeline script.
- All matplotlib output is saved to files (Agg backend) rather than displayed inline, ensuring compatibility with headless environments.
