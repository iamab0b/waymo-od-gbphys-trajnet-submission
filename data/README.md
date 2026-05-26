# Data Access Guide

This document explains how to access and stage Waymo Open Motion Dataset (WOMD) v1.3.1 data for GB-Phys TrajNet.

## 1. Available Data Splits

The WOMD v1.3.1 dataset is hosted on Google Cloud Storage in **two formats**:

### tf_example format (primary — used for model training/evaluation)

| Split | GCS Path | Shards | Usage |
|-------|----------|--------|-------|
| **Training** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-*-of-01000` | 1000 (00000–00999) | Model training |
| **Validation** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord-*-of-00150` | 150 (00000–00149) | Hyperparameter tuning, metrics |
| **Testing** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing/testing_tfexample.tfrecord-*-of-00150` | 150 (00000–00149) | Challenge submission (no GT) |
| **Validation Interactive** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-*-of-00150` | 150 (00000–00149) | Interactive prediction eval |
| **Testing Interactive** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing_interactive/testing_interactive_tfexample.tfrecord-*-of-00150` | 150 (00000–00149) | Interactive challenge submission |

### scenario format (alternative proto-based format)

| Split | GCS Path | Shards | Usage |
|-------|----------|--------|-------|
| **Training** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-*-of-01000` | 1000 (00000–00999) | Alternative training format |
| **Training 20s** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training_20s/training_20s.tfrecord-*-of-01000` | 1000 (00000–00999) | Extended 20-second scenarios |
| **Validation** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation/validation.tfrecord-*-of-00150` | 150 (00000–00149) | Alternative val format |
| **Validation Interactive** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/validation_interactive/validation_interactive.tfrecord-*-of-00150` | 150 (00000–00149) | Interactive val |
| **Testing** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/testing/testing.tfrecord-*-of-00150` | 150 (00000–00149) | Challenge submission |
| **Testing Interactive** | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/testing_interactive/testing_interactive.tfrecord-*-of-00150` | 150 (00000–00149) | Interactive challenge |

> **Note:** Our model primarily uses the **tf_example** format. The scenario format is available for future extensions or comparison with alternative parsers.

---

## 2. GCS Access (Streaming)

The dataset is hosted publicly on Google Cloud Storage. You need a Google account and the `gcloud` CLI.

### Authenticate

```bash
gcloud auth login
gcloud auth application-default login
```

If running on HPC without a browser, use:

```bash
gcloud auth login --no-launch-browser
```

### Verify Access

```bash
# List all tf_example splits
gsutil ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/

# List all scenario splits
gsutil ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/

# Count training shards
gsutil ls gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/ | wc -l
# Expected: 1000
```

### Install gcsfs (Python GCS streaming)

```bash
pip install gcsfs google-cloud-storage
```

---

## 3. Staging Data to HPC Scratch

For high-throughput training on HPC systems (e.g., SLURM clusters), copy shards to local fast storage (scratch/SSD) first.

### Copy a subset (recommended for initial experiments)

```bash
SCRATCH=/scratch/$USER/waymo_motion
mkdir -p ${SCRATCH}/{training,validation,testing,validation_interactive,testing_interactive}

# Copy first 10 training shards (~3 GB)
gsutil -m cp \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-0000[0-9]-of-01000" \
  ${SCRATCH}/training/

# Copy all validation shards (~40 GB, 150 files)
gsutil -m cp \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/*.tfrecord*" \
  ${SCRATCH}/validation/
```

### Copy all splits (complete dataset — ~600+ GB total)

```bash
SCRATCH=/scratch/$USER/waymo_motion

# Training (1000 shards, ~280 GB)
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/" \
  ${SCRATCH}/training/

# Validation (150 shards, ~40 GB)
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/" \
  ${SCRATCH}/validation/

# Testing (150 shards, ~40 GB, no ground truth)
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing/" \
  ${SCRATCH}/testing/

# Validation Interactive (150 shards, ~40 GB)
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation_interactive/" \
  ${SCRATCH}/validation_interactive/

# Testing Interactive (150 shards, ~40 GB, no ground truth)
gsutil -m cp -r \
  "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/testing_interactive/" \
  ${SCRATCH}/testing_interactive/
```

> Tip: Use `gsutil -m` for parallel multi-threaded transfer.

---

## 4. Expected Directory Structure

After staging, the expected layout is:

```
/scratch/$USER/waymo_motion/
├── training/
│   ├── training_tfexample.tfrecord-00000-of-01000
│   ├── training_tfexample.tfrecord-00001-of-01000
│   └── ...  (1000 shards: 00000 to 00999)
├── validation/
│   ├── validation_tfexample.tfrecord-00000-of-00150
│   └── ...  (150 shards: 00000 to 00149)
├── testing/
│   ├── testing_tfexample.tfrecord-00000-of-00150
│   └── ...  (150 shards: 00000 to 00149)
├── validation_interactive/
│   ├── validation_interactive_tfexample.tfrecord-00000-of-00150
│   └── ...  (150 shards)
└── testing_interactive/
    ├── testing_interactive_tfexample.tfrecord-00000-of-00150
    └── ...  (150 shards)
```

Pass the root directory to the training script:

```bash
# Training (uses training/ subdir automatically)
python src/training/trainer.py --data_dir /scratch/$USER/waymo_motion --train_split train ...

# Evaluation on validation_interactive
python src/training/trainer.py --data_dir /scratch/$USER/waymo_motion --val_split val_interactive ...

# Submission generation on test split
python src/evaluation/submission.py --test_data /scratch/$USER/waymo_motion --test_split test ...
```

---

## 5. Verifying Data Access

### Via command line (tf.data inspection)

```bash
# Inspect training data (local)
python src/data/loader.py \
  --data_dir /scratch/$USER/waymo_motion \
  --batch_size 2 \
  --num_batches 3 \
  --split train

# Inspect validation_interactive (local)
python src/data/loader.py \
  --data_dir /scratch/$USER/waymo_motion \
  --batch_size 2 \
  --num_batches 1 \
  --split val_interactive

# Inspect test data via GCS streaming
python src/data/loader.py \
  --split test \
  --batch_size 2 \
  --num_batches 1
```

Expected output (per batch):

```
--- Batch 1 ---
  input_states: shape=(2, 128, 11, 7), dtype=float32
  gt_future_states: shape=(2, 128, 91, 7), dtype=float32
  gt_future_is_valid: shape=(2, 128, 91), dtype=bool
  object_type: shape=(2, 128), dtype=float32
  tracks_to_predict: shape=(2, 128), dtype=bool
  roadgraph_xyz: shape=(2, 30000, 3), dtype=float32
  ...
```

### Via GCS (requires gcloud auth)

```bash
# Any split can be loaded by name
python src/data/loader.py --split train --batch_size 2 --num_batches 1
python src/data/loader.py --split val --batch_size 2 --num_batches 1
python src/data/loader.py --split test --batch_size 2 --num_batches 1
python src/data/loader.py --split val_interactive --batch_size 2 --num_batches 1
python src/data/loader.py --split test_interactive --batch_size 2 --num_batches 1

# Or use an explicit GCS pattern
python src/data/loader.py \
  --gcs_pattern "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord-*-of-01000" \
  --batch_size 2 \
  --num_batches 1
```

---

## 6. Dataset Statistics

| Split | Shards | Scenarios (approx.) | Approx. Size | Has Ground Truth |
|-------|--------|---------------------|-------------|-----------------|
| Training | 1000 | ~486,995 | ~280 GB | Yes |
| Validation | 150 | ~44,097 | ~40 GB | Yes |
| Testing | 150 | ~44,920 | ~40 GB | **No** (server-side eval) |
| Validation Interactive | 150 | ~44,097 | ~40 GB | Yes |
| Testing Interactive | 150 | ~44,920 | ~40 GB | **No** (server-side eval) |

Each shard is a `.tfrecord` file with serialized `tf.Example` protos.
Each scenario contains 128 agent slots, 11 history steps (10 past + 1 current), 80 future steps, and 30,000 road-graph samples.

---

## 7. Split Usage in the Training Pipeline

| Phase | Training Data | Validation Data |
|-------|---------------|-----------------|
| Phase 1 (Goal Prediction) | `train` (1000 shards) | `val` (150 shards) |
| Phase 2 (Trajectory Gen.) | `train` (1000 shards) | `val` (150 shards) |
| Phase 3 (End-to-End) | `train` (1000 shards) | `val` (150 shards) |
| Final Evaluation | — | `val` + `val_interactive` |
| Challenge Submission | — | `test` or `test_interactive` |

To use the interactive splits or the scenario-format training_20s:

```bash
# Train with interactive validation
sbatch --export=PHASE=3,VAL_SPLIT=val_interactive scripts/train.slurm

# Generate submission for interactive challenge
sbatch --export=MODE=submit,EVAL_SPLIT=test_interactive scripts/eval.slurm

# Generate submission for standard motion prediction challenge
sbatch --export=MODE=submit,EVAL_SPLIT=test scripts/eval.slurm
```
