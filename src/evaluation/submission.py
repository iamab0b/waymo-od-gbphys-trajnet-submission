"""
Submission generation for Waymo Open Dataset Motion Prediction Challenge.

Handles:
  - Downsampling internal 10 Hz trajectories to 2 Hz for submission
  - Creating the full submission proto from model predictions
  - Validating submission format
"""

import os
import argparse
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from waymo_open_dataset.protos import motion_submission_pb2
    WAYMO_AVAILABLE = True
except ImportError:
    motion_submission_pb2 = None
    WAYMO_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Indices into the 80-step (10 Hz) horizon that correspond to 2 Hz output
# Every 5th step starting at index 4: [4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#                                       54, 59, 64, 69, 74, 79]  => 16 steps
SUBMISSION_INDICES = list(range(4, 80, 5))

# Submission metadata
ACCOUNT_NAME   = 'smehta22@terpmail.umd.edu'
METHOD_NAME    = 'GB-Phys TrajNet'
DESCRIPTION    = ('Physics-informed multi-modal trajectory prediction using '
                  'goal-conditioned GRU decoder with differentiable bicycle kinematics.')
AUTHORS        = ['Sahil Mehta']
AFFILIATION    = 'University of Maryland, College Park'
UNIQUE_METHOD_NAME = 'gbphys_trajnet'


def downsample_for_submission(internal_trajectory: np.ndarray) -> np.ndarray:
    """
    Downsample a 10 Hz (80-step) trajectory to 2 Hz (16-step) for submission.

    Args:
        internal_trajectory: [..., 80, 2]  positions at 10 Hz.

    Returns:
        downsampled: [..., 16, 2]  positions at 2 Hz.
    """
    return internal_trajectory[..., SUBMISSION_INDICES, :]


def downsample_for_submission_torch(internal_trajectory):
    """
    Torch version of downsample_for_submission.

    Args:
        internal_trajectory: [..., 80, 2]  torch.Tensor.

    Returns:
        downsampled: [..., 16, 2]
    """
    if not TORCH_AVAILABLE:
        raise ImportError('PyTorch required.')
    idx = torch.tensor(SUBMISSION_INDICES, device=internal_trajectory.device)
    return internal_trajectory[..., idx, :]


def _validate_submission(submission) -> bool:
    """
    Perform basic sanity checks on a submission proto.

    Args:
        submission: motion_submission_pb2.MotionChallengeSubmission instance.

    Returns:
        True if valid.

    Raises:
        ValueError if any check fails.
    """
    if not WAYMO_AVAILABLE:
        raise ImportError('waymo-open-dataset is required for submission validation.')

    if not submission.account_name:
        raise ValueError('Submission missing account_name.')
    if not submission.unique_method_name:
        raise ValueError('Submission missing unique_method_name.')
    if len(submission.scenario_predictions) == 0:
        raise ValueError('Submission contains no scenario predictions.')

    for sp in submission.scenario_predictions:
        for pp in sp.single_predictions.predictions:
            for scored_traj in pp.trajectories:
                # trajectories is a repeated ScoredTrajectory;
                # center_x/center_y live on scored_traj.trajectory (sub-message)
                if len(scored_traj.trajectory.center_x) != 16:
                    raise ValueError(
                        f'Expected 16 trajectory points, '
                        f'got {len(scored_traj.trajectory.center_x)}.')
    return True


def create_submission(model, test_dataloader, output_path: str,
                      device: str = 'cpu', batch_size: int = 32,
                      max_batches: int = None) -> str:
    """
    Generate a Waymo submission file from model predictions.

    Args:
        model:           GBPhysTrajNet instance (will be set to eval mode).
        test_dataloader: tf.data.Dataset over test TFRecord files.
        output_path:     Path to write the .pb (protobuf) submission file.
        device:          PyTorch device.
        batch_size:      Batch size (informational; dataset already batched).
        max_batches:     Limit number of batches (None = all).

    Returns:
        output_path on success.
    """
    if not WAYMO_AVAILABLE:
        raise ImportError(
            'waymo-open-dataset is required for submission generation.')

    from src.data.loader import tf_to_torch_batch

    submission = motion_submission_pb2.MotionChallengeSubmission()
    submission.account_name       = ACCOUNT_NAME
    submission.unique_method_name = UNIQUE_METHOD_NAME
    submission.description        = DESCRIPTION

    # `method_name` was removed in waymo-open-dataset SDK v1.6.x.
    # Wrap in try/except so the submission still generates on all SDK versions.
    try:
        submission.method_name = METHOD_NAME
    except AttributeError:
        pass

    # `authors` is a repeated string field in some SDK versions, a single
    # string in others. Handle both gracefully.
    try:
        for a in AUTHORS:
            submission.authors.append(a)
    except AttributeError:
        try:
            submission.authors = AUTHORS[0]
        except AttributeError:
            pass

    # `affiliation` may also differ by SDK version
    try:
        submission.affiliation = AFFILIATION
    except AttributeError:
        pass

    model.eval()
    torch_device = torch.device(device) if TORCH_AVAILABLE else None

    with torch.no_grad():
        for batch_idx, tf_batch in enumerate(test_dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Convert to torch
            torch_batch = tf_to_torch_batch(tf_batch, device=device)

            # Model forward
            output = model(torch_batch, phase=3)

            # Extract predictions
            trajectories  = output['trajectories'].cpu().numpy()  # [B, 128, K, 80, 2]
            confidences   = output['confidences'].cpu().numpy()   # [B, 128, K]

            # Downsample to 2 Hz
            trajectories_sub = downsample_for_submission(trajectories)  # [B, 128, K, 16, 2]

            # Extract scenario IDs (if available in batch)
            scenario_ids = _extract_scenario_ids(tf_batch, batch_idx)
            object_ids   = _extract_object_ids(tf_batch)
            ttp_mask     = torch_batch['tracks_to_predict'].cpu().numpy()  # [B, 128]

            B = trajectories_sub.shape[0]
            for b in range(B):
                scenario_pred = submission.scenario_predictions.add()
                scenario_pred.scenario_id = scenario_ids[b]

                for a in range(128):
                    if not ttp_mask[b, a]:
                        continue

                    pred_proto = scenario_pred.single_predictions.predictions.add()
                    pred_proto.object_id = int(object_ids[b, a])

                    for k in range(confidences.shape[2]):
                        # pred_proto.trajectories is a repeated ScoredTrajectory.
                        # ScoredTrajectory has:
                        #   .confidence  (float)
                        #   .trajectory  (Trajectory sub-message)
                        #       .center_x  (repeated float)
                        #       .center_y  (repeated float)
                        scored_traj = pred_proto.trajectories.add()
                        scored_traj.confidence = float(confidences[b, a, k])
                        # Use .extend() on the nested Trajectory sub-message
                        scored_traj.trajectory.center_x.extend(
                            trajectories_sub[b, a, k, :, 0].tolist())
                        scored_traj.trajectory.center_y.extend(
                            trajectories_sub[b, a, k, :, 1].tolist())

    # Validate
    _validate_submission(submission)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(submission.SerializeToString())

    print(f'[Submission] Written to: {output_path}  '
          f'({len(submission.scenario_predictions)} scenarios)')
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_scenario_ids(tf_batch, batch_idx: int) -> list:
    """Extract scenario ID strings from a tf batch, or generate placeholders."""
    # WOMD test data doesn't expose scenario_id directly in this feature set.
    # If available, use it; otherwise generate indexed placeholders.
    try:
        import tensorflow as tf
        ids = tf_batch.get('scenario/id', None)
        if ids is not None:
            return [ids[b].numpy().decode('utf-8') for b in range(ids.shape[0])]
    except Exception:
        pass

    B = list(tf_batch.values())[0].shape[0]
    return [f'scenario_{batch_idx:06d}_{b:04d}' for b in range(B)]


def _extract_object_ids(tf_batch) -> np.ndarray:
    """Extract per-agent object IDs from tf batch.  [B, 128]"""
    try:
        import tensorflow as tf
        ids = tf_batch.get('state/id', None)
        if ids is not None:
            return ids.numpy().astype(np.int64)
    except Exception:
        pass
    B = list(tf_batch.values())[0].shape[0]
    return np.arange(128, dtype=np.int64)[np.newaxis, :].repeat(B, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Waymo submission file from trained GB-Phys TrajNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path or GCS pattern for test TFRecord data. '
                             'If not provided, uses --test_split to select GCS pattern.')
    parser.add_argument('--test_split', type=str, default='test',
                        choices=['test', 'test_interactive', 'val', 'val_interactive'],
                        help='Which split to generate submission for')
    parser.add_argument('--output_path', type=str, default='submissions/submission.pb',
                        help='Output path for the submission protobuf file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limit number of batches (for debugging)')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    from src.models.gbphys_trajnet import GBPhysTrajNet
    from src.data.loader import (
        create_local_dataset, create_gcs_dataset,
        GCS_TEST, GCS_TEST_INTERACTIVE, GCS_VAL, GCS_VAL_INTERACTIVE,
        GCS_SPLIT_PATTERNS,
    )
    from src.training.checkpointing import load_checkpoint

    # Build model and load checkpoint
    model     = GBPhysTrajNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    load_checkpoint(model, optimizer, args.checkpoint, device=args.device)
    model.to(args.device)

    # Resolve test data source
    if args.test_data is not None:
        test_data = args.test_data
    else:
        test_data = GCS_SPLIT_PATTERNS.get(args.test_split)
        if test_data is None:
            raise ValueError(
                f"Unknown test_split '{args.test_split}'. "
                f"Available: {list(GCS_SPLIT_PATTERNS.keys())}")

    # Build dataset
    if test_data.startswith('gs://'):
        ds = create_gcs_dataset(test_data, batch_size=args.batch_size,
                                shuffle_buffer=0)
    else:
        ds = create_local_dataset(test_data, batch_size=args.batch_size,
                                  shuffle_buffer=0, split=args.test_split)

    create_submission(model, ds, args.output_path,
                      device=args.device, max_batches=args.max_batches)
