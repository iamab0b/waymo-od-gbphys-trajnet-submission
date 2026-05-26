"""
Feature definitions for Waymo Open Motion Dataset (WOMD).

Exact feature schema matching tutorial_motion.ipynb with all roadgraph,
state, and traffic_light features.
"""

try:
    import tensorflow as tf
except ImportError:
    tf = None

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
num_map_samples = 30000
NUM_AGENTS = 128
NUM_PAST_STEPS = 10
NUM_FUTURE_STEPS = 80
NUM_HISTORY_STEPS = 11  # 10 past + 1 current

# ---------------------------------------------------------------------------
# Feature dictionaries (exact schema from tutorial_motion.ipynb)
# ---------------------------------------------------------------------------

def _make_features():
    """Build feature description dicts; requires TensorFlow to be installed."""
    if tf is None:
        raise ImportError("TensorFlow is required to build feature descriptions.")

    _roadgraph_features = {
        'roadgraph_samples/dir': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None),
        'roadgraph_samples/id': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None),
        'roadgraph_samples/type': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None),
        'roadgraph_samples/valid': tf.io.FixedLenFeature(
            [num_map_samples, 1], tf.int64, default_value=None),
        'roadgraph_samples/xyz': tf.io.FixedLenFeature(
            [num_map_samples, 3], tf.float32, default_value=None),
    }

    _state_features = {
        # Agent identity / meta
        'state/id': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/type': tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/is_sdc': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        'state/tracks_to_predict': tf.io.FixedLenFeature([128], tf.int64, default_value=None),
        # Current state (1 timestep)
        'state/current/bbox_yaw': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/height': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/length': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/timestamp_micros': tf.io.FixedLenFeature([128, 1], tf.int64),
        'state/current/valid': tf.io.FixedLenFeature([128, 1], tf.int64),
        'state/current/vel_yaw': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/velocity_x': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/velocity_y': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/width': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/x': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/y': tf.io.FixedLenFeature([128, 1], tf.float32),
        'state/current/z': tf.io.FixedLenFeature([128, 1], tf.float32),
        # Future state (80 timesteps)
        'state/future/bbox_yaw': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/height': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/length': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/timestamp_micros': tf.io.FixedLenFeature([128, 80], tf.int64),
        'state/future/valid': tf.io.FixedLenFeature([128, 80], tf.int64),
        'state/future/vel_yaw': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/velocity_x': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/velocity_y': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/width': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/x': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/y': tf.io.FixedLenFeature([128, 80], tf.float32),
        'state/future/z': tf.io.FixedLenFeature([128, 80], tf.float32),
        # Past state (10 timesteps)
        'state/past/bbox_yaw': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/height': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/length': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/timestamp_micros': tf.io.FixedLenFeature([128, 10], tf.int64),
        'state/past/valid': tf.io.FixedLenFeature([128, 10], tf.int64),
        'state/past/vel_yaw': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/velocity_x': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/velocity_y': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/width': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/x': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/y': tf.io.FixedLenFeature([128, 10], tf.float32),
        'state/past/z': tf.io.FixedLenFeature([128, 10], tf.float32),
    }

    _traffic_light_features = {
        'traffic_light_state/current/state': tf.io.FixedLenFeature([1, 16], tf.int64),
        'traffic_light_state/current/valid': tf.io.FixedLenFeature([1, 16], tf.int64),
        'traffic_light_state/current/x': tf.io.FixedLenFeature([1, 16], tf.float32),
        'traffic_light_state/current/y': tf.io.FixedLenFeature([1, 16], tf.float32),
        'traffic_light_state/current/z': tf.io.FixedLenFeature([1, 16], tf.float32),
        'traffic_light_state/past/state': tf.io.FixedLenFeature([10, 16], tf.int64),
        'traffic_light_state/past/valid': tf.io.FixedLenFeature([10, 16], tf.int64),
        'traffic_light_state/past/x': tf.io.FixedLenFeature([10, 16], tf.float32),
        'traffic_light_state/past/y': tf.io.FixedLenFeature([10, 16], tf.float32),
        'traffic_light_state/past/z': tf.io.FixedLenFeature([10, 16], tf.float32),
    }

    _features_description = {}
    _features_description.update(_roadgraph_features)
    _features_description.update(_state_features)
    _features_description.update(_traffic_light_features)

    return _roadgraph_features, _state_features, _traffic_light_features, _features_description


# Build at import time if TF available; else provide lazy accessors
if tf is not None:
    roadgraph_features, state_features, traffic_light_features, features_description = _make_features()
else:
    roadgraph_features = None
    state_features = None
    traffic_light_features = None
    features_description = None


def get_features_description():
    """Return features_description, building it on demand if necessary."""
    if features_description is not None:
        return features_description
    _, _, _, fd = _make_features()
    return fd
