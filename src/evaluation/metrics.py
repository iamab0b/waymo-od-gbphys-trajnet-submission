"""
Evaluation metrics for Waymo Open Motion Dataset trajectory prediction.

Implements:
  - MotionMetrics  : TF-based metrics class (exact from tutorial, bug-fixed)
  - compute_minADE : numpy minimum Average Displacement Error
  - compute_minFDE : numpy minimum Final Displacement Error
  - compute_miss_rate : numpy miss rate at a given threshold
"""

import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

try:
    from google.protobuf import text_format
    from waymo_open_dataset.metrics.ops import py_metrics_ops
    from waymo_open_dataset.metrics.python import config_util_py as config_util
    from waymo_open_dataset.protos import motion_metrics_pb2
    WAYMO_AVAILABLE = True
except ImportError:
    WAYMO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tutorial MotionMetrics (reproduced with bug fix: self.vars prefix)
# ---------------------------------------------------------------------------

def _default_metrics_config():
    """
    Return the default MotionMetrics config proto.

    Copied verbatim from tutorial_motion.ipynb.  The proto fields are:
      track_steps_per_second          – Hz of the raw tracks (10)
      prediction_steps_per_second     – Hz of the prediction output (2)
      track_history_samples           – observed history steps (10)
      track_future_samples            – future steps in gt track (80)
      speed_lower_bound / upper_bound – m/s thresholds for bucketising agents
      speed_scale_lower / upper       – miss-threshold scaling for slow agents
      step_configurations             – per-horizon lateral/longitudinal miss
                                        thresholds (3 s → step 5, 5 s → step 9,
                                        8 s → step 15 at 2 Hz prediction)
      max_predictions                 – K = 6 for the motion challenge
    """
    if not WAYMO_AVAILABLE:
        raise ImportError('waymo-open-dataset is required for _default_metrics_config.')

    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
track_steps_per_second: 10
prediction_steps_per_second: 2
track_history_samples: 10
track_future_samples: 80
speed_lower_bound: 1.4
speed_upper_bound: 11.0
speed_scale_lower: 0.5
speed_scale_upper: 1.0
step_configurations {
  measurement_step: 5
  lateral_miss_threshold: 1.0
  longitudinal_miss_threshold: 2.0
}
step_configurations {
  measurement_step: 9
  lateral_miss_threshold: 1.8
  longitudinal_miss_threshold: 3.6
}
step_configurations {
  measurement_step: 15
  lateral_miss_threshold: 3.0
  longitudinal_miss_threshold: 6.0
}
max_predictions: 6
"""
    text_format.Parse(config_text, config)
    return config


if TF_AVAILABLE:

    class MotionMetrics(tf.keras.metrics.Metric):
        """
        Official Waymo Open Motion Dataset metrics.

        Exact reproduction from tutorial_motion.ipynb with the `self.` bug
        on `reset_state()` corrected.

        Usage::

            metrics = MotionMetrics(_default_metrics_config())
            metrics.update_state(pred_trajectory, pred_score,
                                 gt_trajectory, gt_is_valid, object_type)
            result = metrics.result()
        """

        def __init__(self, config, **kwargs):
            super().__init__(**kwargs)
            self._config = config
            self._num_pred_steps = config.prediction_steps_per_second * \
                (config.track_steps_per_second // config.prediction_steps_per_second)

            self.reset_state()

        def reset_state(self):
            # Bug fix: use self. prefix consistently
            self._prediction_trajectory = []
            self._prediction_score      = []
            self._ground_truth_trajectory = []
            self._ground_truth_is_valid   = []
            self._prediction_ground_truth_indices = []
            self._prediction_ground_truth_indices_mask = []
            self._object_type = []

        def update_state(self, prediction_trajectory, prediction_score,
                         ground_truth_trajectory, ground_truth_is_valid,
                         object_type, prediction_ground_truth_indices=None,
                         prediction_ground_truth_indices_mask=None):
            """
            Accumulate one batch of predictions and ground-truth for later scoring.

            Required tensor shapes (all TF tensors or numpy arrays):

            prediction_trajectory : [B, A, K, 1, T, 2]  float32
                B = batch size, A = num agents (128), K = num modes (6),
                1 = num_agents_per_joint_prediction (always 1 for independent
                    agent predictions), T = prediction steps at 2 Hz (16),
                2 = (x, y).
                A 5-D tensor [B, A, K, T, 2] is also accepted and will be
                automatically expanded to 6-D by inserting the missing axis.

            prediction_score      : [B, A, K]            float32
                Confidence scores, one per mode per agent.

            ground_truth_trajectory : [B, A, GT, 7]      float32
                GT = total track steps = track_history_samples +
                     track_future_samples + 1  (e.g. 10 + 80 + 1 = 91).
                The 7 features are (x, y, length, width, bbox_yaw, vx, vy).
                Pass the FULL concatenated track (past + current + future),
                NOT just the future portion and NOT just (x, y).

            ground_truth_is_valid   : [B, A, GT]          bool/int
                Validity mask aligned with ground_truth_trajectory.

            object_type             : [B, A]               int64
                WOMD agent type codes (1=vehicle, 2=pedestrian, 3=cyclist).
            """
            self._prediction_trajectory.append(prediction_trajectory)
            self._prediction_score.append(prediction_score)
            self._ground_truth_trajectory.append(ground_truth_trajectory)
            self._ground_truth_is_valid.append(ground_truth_is_valid)
            self._object_type.append(object_type)

            if prediction_ground_truth_indices is not None:
                self._prediction_ground_truth_indices.append(
                    prediction_ground_truth_indices)
                self._prediction_ground_truth_indices_mask.append(
                    prediction_ground_truth_indices_mask)

        def result(self):
            if not WAYMO_AVAILABLE:
                raise ImportError('waymo-open-dataset is required.')

            # ------------------------------------------------------------------
            # Concatenate accumulated batches
            # ------------------------------------------------------------------
            pred_traj  = tf.concat(self._prediction_trajectory, 0)
            pred_score = tf.concat(self._prediction_score, 0)
            gt_traj    = tf.concat(self._ground_truth_trajectory, 0)
            gt_valid   = tf.concat(self._ground_truth_is_valid, 0)
            # op requires int64 for object_type (input #6)
            obj_type   = tf.cast(tf.concat(self._object_type, 0), tf.int64)

            # ------------------------------------------------------------------
            # Shape correction: auto-expand 5-D → 6-D prediction tensor.
            #
            # The C++ op requires:
            #   [batch, num_preds, top_k, num_agents_per_joint, steps, 2]
            #
            # A common mistake is passing [batch, agents, K, steps, 2] (5-D),
            # which is missing the num_agents_per_joint_prediction=1 axis.
            # Inserting it here avoids a kernel-killing C++ CHECK failure.
            # ------------------------------------------------------------------
            if pred_traj.shape.rank == 5:
                # [B, A, K, T, 2] → [B, A, K, 1, T, 2]
                pred_traj = tf.expand_dims(pred_traj, axis=3)

            # ------------------------------------------------------------------
            # Pre-flight shape assertions (Python-level, before the C++ op).
            # These raise ValueError instead of crashing the kernel.
            # ------------------------------------------------------------------
            pred_rank = pred_traj.shape.rank
            if pred_rank != 6:
                raise ValueError(
                    f'prediction_trajectory must be 6-D '
                    f'[batch, num_preds, top_k, num_agents_per_joint, steps, 2], '
                    f'got shape {pred_traj.shape} ({pred_rank}-D). '
                    f'If your tensor is [B, A, K, T, 2] (5-D), '
                    f'insert the missing axis with '
                    f'tf.expand_dims(pred, axis=3) before calling update_state.')

            if pred_traj.shape[-1] != 2:
                raise ValueError(
                    f'prediction_trajectory last dim must be 2 (x, y), '
                    f'got {pred_traj.shape[-1]}.')

            gt_rank = gt_traj.shape.rank
            if gt_rank != 4:
                raise ValueError(
                    f'ground_truth_trajectory must be 4-D '
                    f'[batch, num_agents, track_steps, 7], '
                    f'got shape {gt_traj.shape} ({gt_rank}-D). '
                    f'Pass the FULL concatenated track (past + current + future) '
                    f'with all 7 features, NOT just future (x, y).')

            if gt_traj.shape[-1] != 7:
                raise ValueError(
                    f'ground_truth_trajectory last dim must be 7 '
                    f'(x, y, length, width, bbox_yaw, vx, vy), '
                    f'got {gt_traj.shape[-1]}. '
                    f'Pass gt_future_states[:, :, :, :] — do NOT slice to [:, :, :, :2].')

            expected_gt_steps = (self._config.track_history_samples
                                 + self._config.track_future_samples + 1)
            if (gt_traj.shape[2] is not None
                    and gt_traj.shape[2] != expected_gt_steps):
                raise ValueError(
                    f'ground_truth_trajectory step dim must be {expected_gt_steps} '
                    f'(track_history_samples={self._config.track_history_samples} + '
                    f'track_future_samples={self._config.track_future_samples} + 1), '
                    f'got {gt_traj.shape[2]}. '
                    f'Pass the full gt_future_states [B, A, 91, 7], not a slice.')

            # ------------------------------------------------------------------
            # Build prediction_ground_truth_indices
            # ------------------------------------------------------------------
            if self._prediction_ground_truth_indices:
                # op requires int64 for prediction_ground_truth_indices (input #4)
                pg_idx  = tf.cast(
                    tf.concat(self._prediction_ground_truth_indices, 0), tf.int64)
                pg_mask = tf.concat(self._prediction_ground_truth_indices_mask, 0)
            else:
                # Default: each prediction corresponds to the agent at the same index.
                # tf.range defaults to int32 — must cast to int64.
                batch_size = tf.shape(pred_traj)[0]
                num_agents = tf.shape(pred_traj)[1]
                pg_idx  = tf.cast(
                    tf.tile(
                        tf.expand_dims(tf.range(num_agents), 0),
                        [batch_size, 1]),
                    tf.int64)
                pg_idx  = tf.expand_dims(pg_idx, -1)
                pg_mask = tf.ones(
                    tf.stack([batch_size, num_agents, 1]), dtype=tf.bool)

            return py_metrics_ops.motion_metrics(
                config=self._config.SerializeToString(),
                prediction_trajectory=pred_traj,
                prediction_score=pred_score,
                ground_truth_trajectory=gt_traj,
                ground_truth_is_valid=gt_valid,
                prediction_ground_truth_indices=pg_idx,
                prediction_ground_truth_indices_mask=pg_mask,
                object_type=obj_type,
            )

else:
    # Provide a stub so imports don't break in PyTorch-only environments
    class MotionMetrics:
        """Stub when TensorFlow is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                'TensorFlow is required to use MotionMetrics.  '
                'Install TensorFlow or use the numpy metric functions instead.')


# ---------------------------------------------------------------------------
# Pure numpy metric implementations (for quick local evaluation)
# ---------------------------------------------------------------------------

def compute_minADE(predicted_trajs: np.ndarray,
                   gt_traj:         np.ndarray,
                   valid_mask:      np.ndarray) -> float:
    """
    Compute minimum Average Displacement Error (minADE) over K modes.

    Args:
        predicted_trajs: [K, T, 2]  K predicted trajectories.
        gt_traj:         [T, 2]     ground-truth trajectory.
        valid_mask:      [T]        bool / float, 1 = valid timestep.

    Returns:
        minADE: Scalar float.
    """
    K, T, _ = predicted_trajs.shape
    mask = valid_mask.astype(float)

    # Per-step L2 distance for each mode: [K, T]
    diff = predicted_trajs - gt_traj[np.newaxis, :, :]   # [K, T, 2]
    dist = np.linalg.norm(diff, axis=-1)                  # [K, T]

    # Average over valid steps
    n_valid = mask.sum().clip(min=1.0)
    ade_k   = (dist * mask[np.newaxis, :]).sum(axis=-1) / n_valid  # [K]

    return float(ade_k.min())


def compute_minFDE(predicted_trajs: np.ndarray,
                   gt_traj:         np.ndarray,
                   valid_mask:      np.ndarray) -> float:
    """
    Compute minimum Final Displacement Error (minFDE) over K modes.

    The "final" step is the last valid timestep.

    Args:
        predicted_trajs: [K, T, 2]
        gt_traj:         [T, 2]
        valid_mask:      [T]  bool / float

    Returns:
        minFDE: Scalar float.
    """
    K, T, _ = predicted_trajs.shape
    mask = valid_mask.astype(float)

    # Last valid index
    indices   = np.arange(T, dtype=float)
    last_idx  = int((mask * indices).argmax())

    # Distances at final step
    diff = predicted_trajs[:, last_idx, :] - gt_traj[last_idx, :]  # [K, 2]
    fde_k = np.linalg.norm(diff, axis=-1)  # [K]

    return float(fde_k.min())


def compute_miss_rate(predicted_trajs: np.ndarray,
                      gt_traj:         np.ndarray,
                      valid_mask:      np.ndarray,
                      threshold:       float = 2.0) -> float:
    """
    Compute miss rate: fraction of cases where minFDE > threshold.

    Args:
        predicted_trajs: [K, T, 2]
        gt_traj:         [T, 2]
        valid_mask:      [T]
        threshold:       FDE threshold in metres (default 2.0).

    Returns:
        miss_rate: 1.0 if missed, 0.0 if at least one mode is within threshold.
    """
    fde = compute_minFDE(predicted_trajs, gt_traj, valid_mask)
    return float(fde > threshold)


def batch_compute_minADE(pred_batch: np.ndarray,
                         gt_batch:   np.ndarray,
                         valid_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised minADE for a batch of agents.

    Args:
        pred_batch:  [B, K, T, 2]
        gt_batch:    [B, T, 2]
        valid_batch: [B, T]

    Returns:
        minADE per sample: [B]
    """
    B, K, T, _ = pred_batch.shape
    diff = pred_batch - gt_batch[:, np.newaxis, :, :]  # [B, K, T, 2]
    dist = np.linalg.norm(diff, axis=-1)               # [B, K, T]
    mask = valid_batch.astype(float)                   # [B, T]
    n_valid = mask.sum(axis=-1, keepdims=True).clip(min=1.0)  # [B, 1]
    ade_k = (dist * mask[:, np.newaxis, :]).sum(axis=-1) / n_valid  # [B, K]
    return ade_k.min(axis=-1)   # [B]


def batch_compute_minFDE(pred_batch: np.ndarray,
                         gt_batch:   np.ndarray,
                         valid_batch: np.ndarray) -> np.ndarray:
    """
    Vectorised minFDE for a batch of agents.

    Args:
        pred_batch:  [B, K, T, 2]
        gt_batch:    [B, T, 2]
        valid_batch: [B, T]

    Returns:
        minFDE per sample: [B]
    """
    B, K, T, _ = pred_batch.shape
    mask = valid_batch.astype(float)
    indices  = np.arange(T, dtype=float)
    # Last valid index per sample
    last_idx = (mask * indices[np.newaxis, :]).argmax(axis=-1).astype(int)  # [B]

    # FDE at last valid step
    gt_final   = gt_batch[np.arange(B), last_idx, :]          # [B, 2]
    pred_final = pred_batch[np.arange(B), :, last_idx, :]     # [B, K, 2]
    fde_k = np.linalg.norm(pred_final - gt_final[:, np.newaxis, :], axis=-1)  # [B, K]
    return fde_k.min(axis=-1)   # [B]
