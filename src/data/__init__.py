"""
Data package for GB-Phys TrajNet.

Exports key symbols for feature definitions, data loading, and preprocessing.
"""

from src.data.feature_defs import (
    num_map_samples,
    NUM_AGENTS,
    NUM_PAST_STEPS,
    NUM_FUTURE_STEPS,
    NUM_HISTORY_STEPS,
    roadgraph_features,
    state_features,
    traffic_light_features,
    features_description,
    get_features_description,
)

from src.data.loader import (
    parse_womd_example,
    create_gcs_dataset,
    create_local_dataset,
    tf_to_torch_batch,
    GCS_BUCKET,
    GCS_BUCKET_TF,
    GCS_BUCKET_SCENARIO,
    GCS_TRAIN,
    GCS_VAL,
    GCS_TEST,
    GCS_VAL_INTERACTIVE,
    GCS_TEST_INTERACTIVE,
    GCS_SCENARIO_TRAIN,
    GCS_SCENARIO_TRAIN_20S,
    GCS_SCENARIO_VAL,
    GCS_SCENARIO_VAL_INTERACTIVE,
    GCS_SCENARIO_TEST,
    GCS_SCENARIO_TEST_INTERACTIVE,
    GCS_SPLIT_PATTERNS,
    GCS_SCENARIO_SPLIT_PATTERNS,
)

from src.data.preprocessing import (
    normalize_positions,
    compute_accelerations,
    agent_type_to_onehot,
    prepare_agent_history,
    prepare_road_features,
    random_rotation_augmentation,
    random_translation_augmentation,
)

__all__ = [
    # feature_defs
    'num_map_samples',
    'NUM_AGENTS',
    'NUM_PAST_STEPS',
    'NUM_FUTURE_STEPS',
    'NUM_HISTORY_STEPS',
    'roadgraph_features',
    'state_features',
    'traffic_light_features',
    'features_description',
    'get_features_description',
    # loader
    'parse_womd_example',
    'create_gcs_dataset',
    'create_local_dataset',
    'tf_to_torch_batch',
    'GCS_BUCKET',
    'GCS_BUCKET_TF',
    'GCS_BUCKET_SCENARIO',
    'GCS_TRAIN',
    'GCS_VAL',
    'GCS_TEST',
    'GCS_VAL_INTERACTIVE',
    'GCS_TEST_INTERACTIVE',
    'GCS_SCENARIO_TRAIN',
    'GCS_SCENARIO_TRAIN_20S',
    'GCS_SCENARIO_VAL',
    'GCS_SCENARIO_VAL_INTERACTIVE',
    'GCS_SCENARIO_TEST',
    'GCS_SCENARIO_TEST_INTERACTIVE',
    'GCS_SPLIT_PATTERNS',
    'GCS_SCENARIO_SPLIT_PATTERNS',
    # preprocessing
    'normalize_positions',
    'compute_accelerations',
    'agent_type_to_onehot',
    'prepare_agent_history',
    'prepare_road_features',
    'random_rotation_augmentation',
    'random_translation_augmentation',
]
