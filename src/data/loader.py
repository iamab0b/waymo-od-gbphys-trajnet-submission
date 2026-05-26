"""
Data loading utilities for Waymo Open Motion Dataset (WOMD).

Provides tf.data pipelines for both GCS streaming and local HPC data,
plus conversion utilities to PyTorch tensors.
"""

import argparse
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from src.data.feature_defs import (
    features_description,
    get_features_description,
    num_map_samples,
    NUM_AGENTS,
    NUM_PAST_STEPS,
    NUM_FUTURE_STEPS,
    NUM_HISTORY_STEPS,
)

# ---------------------------------------------------------------------------
# GCS paths — tf_example format (used for training / evaluation)
# ---------------------------------------------------------------------------
GCS_BUCKET_TF = 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example'
GCS_TRAIN = f'{GCS_BUCKET_TF}/training/training_tfexample.tfrecord-*-of-01000'
GCS_VAL = f'{GCS_BUCKET_TF}/validation/validation_tfexample.tfrecord-*-of-00150'
GCS_TEST = f'{GCS_BUCKET_TF}/testing/testing_tfexample.tfrecord-*-of-00150'
GCS_VAL_INTERACTIVE = f'{GCS_BUCKET_TF}/validation_interactive/validation_interactive_tfexample.tfrecord-*-of-00150'
GCS_TEST_INTERACTIVE = f'{GCS_BUCKET_TF}/testing_interactive/testing_interactive_tfexample.tfrecord-*-of-00150'

# ---------------------------------------------------------------------------
# GCS paths — scenario format (alternative proto-based format)
# ---------------------------------------------------------------------------
GCS_BUCKET_SCENARIO = 'gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario'
GCS_SCENARIO_TRAIN = f'{GCS_BUCKET_SCENARIO}/training/training.tfrecord-*-of-01000'
GCS_SCENARIO_TRAIN_20S = f'{GCS_BUCKET_SCENARIO}/training_20s/training_20s.tfrecord-*-of-01000'
GCS_SCENARIO_VAL = f'{GCS_BUCKET_SCENARIO}/validation/validation.tfrecord-*-of-00150'
GCS_SCENARIO_VAL_INTERACTIVE = f'{GCS_BUCKET_SCENARIO}/validation_interactive/validation_interactive.tfrecord-*-of-00150'
GCS_SCENARIO_TEST = f'{GCS_BUCKET_SCENARIO}/testing/testing.tfrecord-*-of-00150'
GCS_SCENARIO_TEST_INTERACTIVE = f'{GCS_BUCKET_SCENARIO}/testing_interactive/testing_interactive.tfrecord-*-of-00150'

# Convenience lookup mapping split name → GCS pattern (tf_example format)
GCS_SPLIT_PATTERNS = {
    'train': GCS_TRAIN,
    'val': GCS_VAL,
    'test': GCS_TEST,
    'val_interactive': GCS_VAL_INTERACTIVE,
    'test_interactive': GCS_TEST_INTERACTIVE,
}

# Convenience lookup mapping split name → GCS pattern (scenario format)
GCS_SCENARIO_SPLIT_PATTERNS = {
    'train': GCS_SCENARIO_TRAIN,
    'train_20s': GCS_SCENARIO_TRAIN_20S,
    'val': GCS_SCENARIO_VAL,
    'val_interactive': GCS_SCENARIO_VAL_INTERACTIVE,
    'test': GCS_SCENARIO_TEST,
    'test_interactive': GCS_SCENARIO_TEST_INTERACTIVE,
}

# Backward compatibility alias
GCS_BUCKET = GCS_BUCKET_TF


def parse_womd_example(serialized_example):
    """
    Parse a serialized WOMD tf.Example proto into a structured feature dict.

    Extends tutorial _parse() to include full 7-feature states, roadgraph,
    and traffic-light context required by GB-Phys TrajNet.

    Args:
        serialized_example: Scalar string tensor (serialized tf.Example).

    Returns:
        dict with keys:
            input_states        : [128, 11, 7]  past+current (x,y,l,w,yaw,vx,vy)
            gt_future_states    : [128, 91, 7]  past+cur+future all 7 features
            gt_future_is_valid  : [128, 91]     boolean validity mask
            object_type         : [128]
            tracks_to_predict   : [128]         boolean
            sample_is_valid     : [128]         boolean
            roadgraph_xyz       : [30000, 3]
            roadgraph_dir       : [30000, 3]
            roadgraph_valid     : [30000, 1]    int64
            roadgraph_type      : [30000, 1]    int64
            tl_current_state    : [1, 16]
            tl_current_valid    : [1, 16]
            tl_current_xyz      : [1, 16, 3]
            tl_past_state       : [10, 16]
            tl_past_valid       : [10, 16]
    """
    fd = get_features_description()
    decoded = tf.io.parse_single_example(serialized_example, fd)

    # ------------------------------------------------------------------
    # Build state tensors [128, T, 7] for past / current / future
    # 7 features: x, y, length, width, bbox_yaw, velocity_x, velocity_y
    # ------------------------------------------------------------------
    def _stack7(prefix, T):
        """Stack 7 features for a given temporal window."""
        return tf.stack([
            decoded[f'state/{prefix}/x'],         # [128, T]
            decoded[f'state/{prefix}/y'],
            decoded[f'state/{prefix}/length'],
            decoded[f'state/{prefix}/width'],
            decoded[f'state/{prefix}/bbox_yaw'],
            decoded[f'state/{prefix}/velocity_x'],
            decoded[f'state/{prefix}/velocity_y'],
        ], axis=-1)  # -> [128, T, 7]

    past_states    = _stack7('past', NUM_PAST_STEPS)       # [128, 10, 7]
    current_states = _stack7('current', 1)                 # [128,  1, 7]
    future_states  = _stack7('future', NUM_FUTURE_STEPS)   # [128, 80, 7]

    # Input to model: past + current, x/y only for encoders
    input_states = tf.concat([past_states, current_states], axis=1)  # [128, 11, 7]

    # Ground-truth: past + current + future all 7 features
    gt_future_states = tf.concat([past_states, current_states, future_states], axis=1)  # [128, 91, 7]

    # ------------------------------------------------------------------
    # Validity masks
    # ------------------------------------------------------------------
    past_is_valid    = decoded['state/past/valid'] > 0       # [128, 10]
    current_is_valid = decoded['state/current/valid'] > 0    # [128,  1]
    future_is_valid  = decoded['state/future/valid'] > 0     # [128, 80]
    gt_future_is_valid = tf.concat([past_is_valid, current_is_valid, future_is_valid], axis=1)  # [128, 91]

    # An agent is "valid" if it appears in at least one observed step
    sample_is_valid = tf.reduce_any(tf.concat([past_is_valid, current_is_valid], axis=1), axis=1)  # [128]

    # ------------------------------------------------------------------
    # Roadgraph
    # ------------------------------------------------------------------
    roadgraph_xyz   = decoded['roadgraph_samples/xyz']    # [30000, 3]
    roadgraph_dir   = decoded['roadgraph_samples/dir']    # [30000, 3]
    roadgraph_valid = decoded['roadgraph_samples/valid']  # [30000, 1]
    roadgraph_type  = decoded['roadgraph_samples/type']   # [30000, 1]

    # ------------------------------------------------------------------
    # Traffic lights
    # ------------------------------------------------------------------
    tl_cur_state = decoded['traffic_light_state/current/state']  # [1, 16]
    tl_cur_valid = decoded['traffic_light_state/current/valid']  # [1, 16]
    tl_cur_x     = decoded['traffic_light_state/current/x']      # [1, 16]
    tl_cur_y     = decoded['traffic_light_state/current/y']      # [1, 16]
    tl_cur_z     = decoded['traffic_light_state/current/z']      # [1, 16]
    tl_cur_xyz   = tf.stack([tl_cur_x, tl_cur_y, tl_cur_z], axis=-1)  # [1, 16, 3]

    tl_past_state = decoded['traffic_light_state/past/state']    # [10, 16]
    tl_past_valid = decoded['traffic_light_state/past/valid']    # [10, 16]

    return {
        'input_states':       input_states,          # [128, 11, 7]
        'gt_future_states':   gt_future_states,      # [128, 91, 7]
        'gt_future_is_valid': gt_future_is_valid,    # [128, 91]
        'object_type':        decoded['state/type'],           # [128]
        'tracks_to_predict':  decoded['state/tracks_to_predict'] > 0,  # [128]
        'sample_is_valid':    sample_is_valid,                 # [128]
        'roadgraph_xyz':      roadgraph_xyz,         # [30000, 3]
        'roadgraph_dir':      roadgraph_dir,         # [30000, 3]
        'roadgraph_valid':    roadgraph_valid,        # [30000, 1]
        'roadgraph_type':     roadgraph_type,         # [30000, 1]
        'tl_current_state':   tl_cur_state,          # [1, 16]
        'tl_current_valid':   tl_cur_valid,          # [1, 16]
        'tl_current_xyz':     tl_cur_xyz,            # [1, 16, 3]
        'tl_past_state':      tl_past_state,         # [10, 16]
        'tl_past_valid':      tl_past_valid,         # [10, 16]
    }


def _build_dataset(file_pattern, batch_size=32, shuffle_buffer=1000,
                   num_parallel_reads=tf.data.AUTOTUNE, repeat=True):
    """Shared dataset construction logic."""
    files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    dataset = files.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type=''),
        cycle_length=10,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.map(parse_womd_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_gcs_dataset(file_pattern=None, batch_size=32, shuffle_buffer=1000):
    """
    Create a tf.data pipeline streaming from GCS.

    Args:
        file_pattern: GCS glob pattern, defaults to GCS_TRAIN.
        batch_size:   Number of scenarios per batch.
        shuffle_buffer: Size of shuffle buffer (set 0 to disable).

    Returns:
        tf.data.Dataset yielding batched feature dicts.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for create_gcs_dataset.")
    if file_pattern is None:
        file_pattern = GCS_TRAIN
    return _build_dataset(file_pattern, batch_size=batch_size,
                          shuffle_buffer=shuffle_buffer)


def create_local_dataset(data_dir, batch_size=32, shuffle_buffer=1000, split='train'):
    """
    Create a tf.data pipeline reading TFRecord shards from local disk.

    Useful on HPC systems where data has been staged to scratch storage.

    Args:
        data_dir:       Directory containing .tfrecord files.
        batch_size:     Number of scenarios per batch.
        shuffle_buffer: Size of shuffle buffer.
        split:          One of 'train', 'val', 'test', 'val_interactive',
                        'test_interactive', 'train_20s'. Maps to sub-directory.

    Returns:
        tf.data.Dataset yielding batched feature dicts.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required for create_local_dataset.")

    # Map split name to expected sub-directory name on disk
    SPLIT_TO_DIR = {
        'train': 'training',
        'val': 'validation',
        'test': 'testing',
        'val_interactive': 'validation_interactive',
        'test_interactive': 'testing_interactive',
        'train_20s': 'training_20s',
    }
    sub = SPLIT_TO_DIR.get(split, split)
    file_pattern = os.path.join(data_dir, sub, '*.tfrecord*')
    repeat = (split in ('train', 'train_20s'))
    return _build_dataset(file_pattern, batch_size=batch_size,
                          shuffle_buffer=shuffle_buffer, repeat=repeat)


def tf_to_torch_batch(tf_batch, device='cpu'):
    """
    Convert a dict of TF tensors (from tf.data) to PyTorch tensors.

    Args:
        tf_batch: Dict of tf.Tensor values (potentially batched).
        device:   PyTorch device string, e.g. 'cuda' or 'cpu'.

    Returns:
        Dict mapping the same keys to torch.Tensor on the specified device.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for tf_to_torch_batch.")
    torch_batch = {}
    for key, tf_tensor in tf_batch.items():
        numpy_array = tf_tensor.numpy()
        # Determine dtype mapping
        if numpy_array.dtype.kind == 'f':
            dtype = torch.float32
        elif numpy_array.dtype.kind in ('i', 'u'):
            dtype = torch.long
        elif numpy_array.dtype.kind == 'b':
            dtype = torch.bool
        else:
            dtype = None  # let torch infer
        t = torch.tensor(numpy_array, dtype=dtype, device=device)
        torch_batch[key] = t
    return torch_batch


# ---------------------------------------------------------------------------
# CLI for quick dataset inspection / profiling
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description='WOMD data loader utility (inspect / profile tf.data pipelines)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Local directory with TFRecord shards')
    parser.add_argument('--gcs_pattern', type=str, default=None,
                        help='Explicit GCS glob pattern (overrides --split lookup)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inspection')
    parser.add_argument('--num_batches', type=int, default=2,
                        help='Number of batches to read and report')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test',
                                 'val_interactive', 'test_interactive',
                                 'train_20s'],
                        help='Data split to load (maps to GCS path automatically)')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    if args.data_dir is not None:
        print(f'Loading from local directory: {args.data_dir} (split={args.split})')
        ds = create_local_dataset(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            split=args.split,
        )
    else:
        # Use explicit pattern if given, otherwise look up by split name
        if args.gcs_pattern:
            pattern = args.gcs_pattern
        else:
            pattern = GCS_SPLIT_PATTERNS.get(args.split)
            if pattern is None:
                raise ValueError(
                    f"Unknown split '{args.split}'. "
                    f"Available: {list(GCS_SPLIT_PATTERNS.keys())}")
        print(f'Loading from GCS (split={args.split}): {pattern}')
        ds = create_gcs_dataset(
            file_pattern=pattern,
            batch_size=args.batch_size,
        )

    for i, batch in enumerate(ds.take(args.num_batches)):
        print(f'\n--- Batch {i+1} ---')
        for k, v in batch.items():
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
