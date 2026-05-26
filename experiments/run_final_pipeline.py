#!/usr/bin/env python3
"""
GB-Phys TrajNet — Full Final Pipeline with Comprehensive Visualizations
(HPC version of final_model.ipynb)

Reproduces every section of final_model.ipynb as a non-interactive script
suitable for SLURM execution with GPU acceleration.  All figures are saved
as high-resolution PNGs to --log_dir.

Sections:
  1.  Data pipeline verification + data visualizations
  2.  Baselines (Constant Velocity)
  3.  Model setup and architecture summary
  4.  Three-phase training with per-phase loss/metric plots
  5.  Prediction visualizations (trajectories, goals, confidence)
  6.  Error analysis (per-type, per-speed, FDE distributions)
  7.  Official MotionMetrics evaluation + breakdown heatmap
  8.  Comparison summary (CV vs GB-Phys TrajNet)
  9.  Submission generation

Generated figures (all saved to --log_dir):
  data_scenario_overview.png      — roadgraph + agents + GT future
  data_agent_type_dist.png        — vehicle/ped/cyclist counts
  data_velocity_distribution.png  — speed histogram per agent type
  data_tracks_to_predict.png      — TTP count distribution
  data_roadgraph_types.png        — roadgraph points colored by type
  phase{1,2,3}_loss_curve.png     — per-phase training loss (raw + smoothed)
  phase{1,2,3}_loss_components.png— goal/wta/physics/boundary breakdown
  training_val_metrics.png        — minADE/minFDE/MR over training
  training_all_phases.png         — combined loss across all phases
  pred_trajectories.png           — K=6 predictions vs GT (3×3 grid)
  pred_goal_candidates.png        — goal candidates + top-6 + GT endpoint
  pred_confidence_dist.png        — confidence score distribution
  pred_winner_modes.png           — which mode wins (min FDE to GT)
  pred_cv_vs_gbphys.png           — side-by-side CV and GB-Phys predictions
  error_ade_histogram.png         — ADE distribution by agent type
  error_fde_histogram.png         — FDE distribution by agent type
  error_ade_vs_speed.png          — minADE vs agent speed scatter
  error_miss_rate_by_type.png     — miss rate per type at 3s/5s/8s
  metrics_official_heatmap.png    — official metric breakdown heatmap
  metrics_comparison_bar.png      — CV vs GB-Phys TrajNet bar chart

Usage:
  python experiments/run_final_pipeline.py --num_shards 50 --device cuda
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

import tensorflow as tf

from src.data.loader import (
    create_gcs_dataset, create_local_dataset, tf_to_torch_batch,
    GCS_TRAIN, GCS_VAL, GCS_TEST, GCS_SPLIT_PATTERNS,
)
from src.data.feature_defs import NUM_AGENTS, NUM_HISTORY_STEPS, NUM_FUTURE_STEPS
from src.models.gbphys_trajnet import GBPhysTrajNet, DEFAULT_CONFIG
from src.models.baselines import ConstantVelocityBaseline
from src.losses import CombinedLoss
from src.training.checkpointing import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint, save_best_checkpoint,
)
from src.evaluation.metrics import compute_minADE, compute_minFDE, compute_miss_rate

try:
    from src.evaluation.metrics import MotionMetrics, _default_metrics_config
    from src.evaluation.submission import (
        create_submission, downsample_for_submission, SUBMISSION_INDICES,
        _validate_submission,
    )
    WAYMO_METRICS_AVAILABLE = True
except ImportError:
    WAYMO_METRICS_AVAILABLE = False

# Agent type constants
TYPE_VEHICLE    = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST    = 3
TYPE_NAMES      = {TYPE_VEHICLE: 'Vehicle', TYPE_PEDESTRIAN: 'Pedestrian',
                   TYPE_CYCLIST: 'Cyclist', 0: 'Unknown'}
TYPE_COLORS     = {TYPE_VEHICLE: '#2196F3', TYPE_PEDESTRIAN: '#4CAF50',
                   TYPE_CYCLIST: '#FF9800', 0: '#9E9E9E'}

ROADGRAPH_TYPE_COLORS = {
    0: '#BDBDBD',   # unknown
    1: '#F44336',   # freeway
    2: '#FF9800',   # surface street
    3: '#FFEB3B',   # bike lane
    6: '#FFFFFF',   # broken white
    7: '#E0E0E0',   # solid white
    8: '#FFC107',   # solid yellow
    9: '#FF5722',   # double yellow
    15: '#9C27B0',  # crosswalk
    16: '#3F51B5',  # speed bump
}


# =============================================================================
# Utilities
# =============================================================================

def section_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def save_fig(fig, log_dir, name, dpi=150):
    path = os.path.join(log_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def get_train_files(num_shards):
    all_files = sorted(tf.io.gfile.glob(GCS_TRAIN))
    selected   = all_files[:num_shards]
    print(f"Training shards: {len(selected)} / {len(all_files)}")
    return selected


def build_dataset(file_list, batch_size, shuffle=True, num_parallel_reads=8):
    from src.data.loader import parse_womd_example
    ds = tf.data.TFRecordDataset(file_list, num_parallel_reads=num_parallel_reads)
    ds = ds.map(parse_womd_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# DATA VISUALIZATIONS
# =============================================================================

def viz_scenario_overview(batches_np, log_dir, n_scenarios=3):
    """
    Figure: roadgraph + all agents (past trajectories + GT future).
    Shows 3 scenarios side by side.
    """
    fig, axes = plt.subplots(1, n_scenarios, figsize=(7 * n_scenarios, 8))
    if n_scenarios == 1:
        axes = [axes]
    fig.suptitle('WOMD Scenario Overview\n'
                 '(blue=vehicle, green=pedestrian, orange=cyclist; '
                 '*=tracks_to_predict)', fontsize=12, fontweight='bold')

    rg_color_default = '#BDBDBD'
    for sc_idx, ax in enumerate(axes):
        if sc_idx >= len(batches_np['input_states']):
            ax.axis('off')
            continue

        # Roadgraph
        rg_xyz   = batches_np['roadgraph_xyz'][sc_idx]          # [N, 3]
        rg_valid = batches_np['roadgraph_valid'][sc_idx, :, 0]  # [N]
        rg_type  = batches_np['roadgraph_type'][sc_idx, :, 0].astype(int)  # [N]

        valid_pts = rg_xyz[rg_valid > 0]
        valid_types = rg_type[rg_valid > 0]
        if len(valid_pts) > 0:
            for t_code, t_color in ROADGRAPH_TYPE_COLORS.items():
                mask = (valid_types == t_code)
                if mask.sum() > 0:
                    ax.scatter(valid_pts[mask, 0], valid_pts[mask, 1],
                               s=0.5, c=t_color, alpha=0.4, linewidths=0, zorder=1)

        # Agents
        states   = batches_np['input_states'][sc_idx]            # [128, 11, 7]
        gt_fut   = batches_np['gt_future_states'][sc_idx]        # [128, 91, 7]
        gt_valid = batches_np['gt_future_is_valid'][sc_idx]      # [128, 91]
        types    = batches_np['object_type'][sc_idx]             # [128]
        is_valid = batches_np['sample_is_valid'][sc_idx]         # [128]
        ttp      = batches_np['tracks_to_predict'][sc_idx]       # [128]

        for a in range(min(NUM_AGENTS, 128)):
            if not is_valid[a]:
                continue
            atype  = int(types[a])
            color  = TYPE_COLORS.get(atype, '#9E9E9E')
            past   = states[a, :, :2]             # [11, 2]
            future = gt_fut[a, NUM_HISTORY_STEPS:, :2]   # [80, 2]
            fv     = gt_valid[a, NUM_HISTORY_STEPS:].astype(bool)

            ax.plot(past[:, 0], past[:, 1],
                    '-', color=color, lw=1.2, alpha=0.7, zorder=2)
            cur_pos = past[-1]
            marker = '*' if ttp[a] else 'o'
            ax.plot(cur_pos[0], cur_pos[1],
                    marker, color=color,
                    markersize=10 if ttp[a] else 4,
                    zorder=3)
            if fv.sum() > 3:
                ax.plot(future[fv, 0], future[fv, 1],
                        '--', color=color, lw=0.7, alpha=0.4, zorder=2)

        # Traffic lights
        if 'tl_current_xyz' in batches_np:
            tl_xyz   = batches_np['tl_current_xyz'][sc_idx, 0]   # [16, 3]
            tl_valid = batches_np['tl_current_valid'][sc_idx, 0] # [16]
            tl_state = batches_np['tl_current_state'][sc_idx, 0] # [16]
            tl_c_map = {1: 'lime', 2: 'yellow', 3: 'red', 4: 'red'}
            for tl in range(16):
                if tl_valid[tl] > 0:
                    tc = tl_c_map.get(int(tl_state[tl]), 'gray')
                    ax.plot(tl_xyz[tl, 0], tl_xyz[tl, 1],
                            's', color=tc, markersize=6, zorder=4)

        handles = [
            mpatches.Patch(color=TYPE_COLORS[TYPE_VEHICLE], label='Vehicle'),
            mpatches.Patch(color=TYPE_COLORS[TYPE_PEDESTRIAN], label='Pedestrian'),
            mpatches.Patch(color=TYPE_COLORS[TYPE_CYCLIST], label='Cyclist'),
            mpatches.Patch(color='white', label='-- GT future'),
        ]
        ax.legend(handles=handles, fontsize=7, loc='upper right')
        ax.set_title(f'Scenario {sc_idx + 1}', fontsize=10)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    return save_fig(fig, log_dir, 'data_scenario_overview.png', dpi=150)


def viz_agent_type_distribution(batches_np, log_dir):
    """
    Figure: vehicle/pedestrian/cyclist count distributions.
    Left: pie chart of all agents. Right: bar chart of tracks_to_predict.
    """
    all_types, ttp_types = [], []
    for b in range(len(batches_np['object_type'])):
        is_v = batches_np['sample_is_valid'][b]
        ttp  = batches_np['tracks_to_predict'][b]
        for a in range(NUM_AGENTS):
            if is_v[a]:
                all_types.append(int(batches_np['object_type'][b, a]))
            if ttp[a]:
                ttp_types.append(int(batches_np['object_type'][b, a]))

    all_types  = np.array(all_types)
    ttp_types  = np.array(ttp_types)
    type_codes = [TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST]
    labels     = ['Vehicle', 'Pedestrian', 'Cyclist']
    colors_bar = [TYPE_COLORS[t] for t in type_codes]

    counts_all = [np.sum(all_types == t) for t in type_codes]
    counts_ttp = [np.sum(ttp_types == t) for t in type_codes]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Agent Type Distribution', fontsize=13, fontweight='bold')

    # Pie — all valid agents
    wedges, texts, autotexts = axes[0].pie(
        counts_all, labels=labels, colors=colors_bar,
        autopct='%1.1f%%', startangle=140, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    for t in autotexts:
        t.set_fontsize(9)
    axes[0].set_title(f'All Valid Agents\n(total = {sum(counts_all):,})', fontsize=10)

    # Pie — tracks_to_predict only
    axes[1].pie(
        counts_ttp, labels=labels, colors=colors_bar,
        autopct='%1.1f%%', startangle=140, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=1.5))
    axes[1].set_title(f'Tracks to Predict\n(total = {sum(counts_ttp):,})', fontsize=10)

    # Bar — absolute counts comparison
    x = np.arange(3)
    w = 0.35
    bars1 = axes[2].bar(x - w/2, counts_all,  w, label='All valid',       color=colors_bar, alpha=0.7)
    bars2 = axes[2].bar(x + w/2, counts_ttp,  w, label='Tracks to predict', color=colors_bar, alpha=1.0,
                        edgecolor='black', linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('Count')
    axes[2].set_title('Counts: All vs Tracks-to-Predict', fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    return save_fig(fig, log_dir, 'data_agent_type_dist.png')


def viz_velocity_distribution(batches_np, log_dir):
    """
    Figure: speed histogram per agent type.
    """
    speed_by_type = defaultdict(list)
    for b in range(len(batches_np['input_states'])):
        is_v  = batches_np['sample_is_valid'][b]
        for a in range(NUM_AGENTS):
            if not is_v[a]:
                continue
            atype = int(batches_np['object_type'][b, a])
            vx = batches_np['input_states'][b, a, -1, 5]
            vy = batches_np['input_states'][b, a, -1, 6]
            speed = float(np.sqrt(vx**2 + vy**2))
            speed_by_type[atype].append(speed)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Agent Speed Distribution at Current Timestep', fontsize=13, fontweight='bold')

    for ax, atype in zip(axes, [TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST]):
        speeds = speed_by_type.get(atype, [])
        color  = TYPE_COLORS[atype]
        if speeds:
            ax.hist(speeds, bins=40, color=color, alpha=0.8, edgecolor='none')
            med = np.median(speeds)
            ax.axvline(med, color='black', linestyle='--', linewidth=1.5,
                       label=f'Median: {med:.1f} m/s')
            ax.axvline(np.mean(speeds), color='red', linestyle='-', linewidth=1.5,
                       label=f'Mean:   {np.mean(speeds):.1f} m/s')
            ax.legend(fontsize=8)
        ax.set_xlabel('Speed (m/s)')
        ax.set_ylabel('Count')
        ax.set_title(f'{TYPE_NAMES[atype]} (n={len(speeds):,})', fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, log_dir, 'data_velocity_distribution.png')


def viz_tracks_to_predict(batches_np, log_dir):
    """
    Figure: histogram of tracks_to_predict count per scenario,
    plus type breakdown of tracked agents.
    """
    ttp_counts    = []
    horizon_steps = []  # how many future valid steps per TTP agent

    for b in range(len(batches_np['tracks_to_predict'])):
        ttp = batches_np['tracks_to_predict'][b]
        n_ttp = int(ttp.sum())
        ttp_counts.append(n_ttp)
        for a in range(NUM_AGENTS):
            if ttp[a]:
                valid_fut = batches_np['gt_future_is_valid'][b, a, NUM_HISTORY_STEPS:].sum()
                horizon_steps.append(int(valid_fut))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Tracks-to-Predict Statistics', fontsize=13, fontweight='bold')

    axes[0].hist(ttp_counts, bins=range(0, max(ttp_counts)+2),
                 color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_xlabel('# Tracks-to-Predict per Scenario')
    axes[0].set_ylabel('# Scenarios')
    axes[0].set_title('Tracks-to-Predict Count per Scenario', fontsize=10)
    axes[0].axvline(np.mean(ttp_counts), color='red', linestyle='--',
                    label=f'Mean: {np.mean(ttp_counts):.1f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].hist(np.array(horizon_steps) * 0.1, bins=20,
                 color='darkorange', edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('Valid Future Horizon (seconds)')
    axes[1].set_ylabel('# Agents')
    axes[1].set_title('GT Future Horizon Available per TTP Agent', fontsize=10)
    axes[1].axvline(8.0, color='green', linestyle='--', label='Full 8s horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return save_fig(fig, log_dir, 'data_tracks_to_predict.png')


def viz_roadgraph_types(batches_np, log_dir):
    """
    Figure: roadgraph points for one scenario, colored by road type.
    """
    sc = 0
    rg_xyz   = batches_np['roadgraph_xyz'][sc]
    rg_valid = batches_np['roadgraph_valid'][sc, :, 0]
    rg_type  = batches_np['roadgraph_type'][sc, :, 0].astype(int)
    valid    = rg_valid > 0
    pts_v    = rg_xyz[valid]
    types_v  = rg_type[valid]

    type_labels = {
        0: 'Unknown', 1: 'Freeway', 2: 'Surface street',
        3: 'Bike lane', 6: 'Broken white', 7: 'Solid white',
        8: 'Solid yellow', 9: 'Double yellow',
        15: 'Crosswalk', 16: 'Speed bump',
    }

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    present_types = np.unique(types_v)
    for t_code in present_types:
        mask  = (types_v == t_code)
        color = ROADGRAPH_TYPE_COLORS.get(t_code, '#9E9E9E')
        label = type_labels.get(t_code, f'Type {t_code}')
        ax.scatter(pts_v[mask, 0], pts_v[mask, 1],
                   s=1.0, c=color, alpha=0.6, label=label, linewidths=0)

    # Overlay agents
    types    = batches_np['object_type'][sc]
    states   = batches_np['input_states'][sc]
    is_valid = batches_np['sample_is_valid'][sc]
    for a in range(min(NUM_AGENTS, 128)):
        if not is_valid[a]:
            continue
        color = TYPE_COLORS.get(int(types[a]), '#9E9E9E')
        cur = states[a, -1, :2]
        ax.plot(cur[0], cur[1], 'o', color=color, markersize=4, zorder=5, alpha=0.8)

    ax.legend(fontsize=7, markerscale=4, loc='upper right')
    ax.set_title('Roadgraph Segmentation (Scenario 1)', fontsize=12, fontweight='bold')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    return save_fig(fig, log_dir, 'data_roadgraph_types.png')


# =============================================================================
# TRAINING VISUALIZATIONS
# =============================================================================

def viz_loss_curve(phase, phase_name, all_losses, loss_components_log, val_log, log_dir):
    """
    Figure: training loss for one phase.
    Left: raw + smoothed total loss.
    Right: stacked area for each loss component.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Phase {phase}: {phase_name} — Training', fontsize=13, fontweight='bold')

    # Total loss (raw + smoothed)
    steps = np.arange(len(all_losses))
    axes[0].plot(steps, all_losses, alpha=0.25, color='steelblue', linewidth=0.5)
    window = max(len(all_losses) // 60, 5)
    if len(all_losses) > window:
        smoothed = np.convolve(all_losses, np.ones(window) / window, mode='valid')
        axes[0].plot(np.arange(window - 1, len(all_losses)), smoothed,
                     color='steelblue', linewidth=2, label='Smoothed loss')
    if val_log:
        val_steps = [v[0] for v in val_log]
        val_ades  = [v[1] for v in val_log]
        ax2 = axes[0].twinx()
        ax2.plot(val_steps, val_ades, 'r--o', markersize=4,
                 linewidth=1.5, label='Val minADE')
        ax2.set_ylabel('minADE (m)', color='red')
        ax2.tick_params(axis='y', colors='red')
        ax2.legend(loc='upper right', fontsize=8)
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss + Validation minADE', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, loc='upper left')

    # Loss component breakdown
    if loss_components_log:
        comps = loss_components_log
        comp_steps = np.arange(len(comps))
        goal_vals    = np.array([c.get('goal', 0)     for c in comps])
        wta_vals     = np.array([c.get('wta', 0)      for c in comps])
        physics_vals = np.array([c.get('physics', 0)  for c in comps])
        boundary_vals= np.array([c.get('boundary', 0) for c in comps])

        # Smooth each component
        w = max(len(comps) // 60, 5)
        def smooth(x):
            if len(x) > w:
                return np.convolve(x, np.ones(w)/w, mode='valid')
            return x
        xs = comp_steps[w-1:] if len(comps) > w else comp_steps

        axes[1].stackplot(xs,
            smooth(goal_vals), smooth(wta_vals),
            smooth(physics_vals), smooth(boundary_vals),
            labels=['Goal CE', 'WTA L2', 'Physics Reg', 'Boundary'],
            colors=['#E53935', '#1E88E5', '#43A047', '#FB8C00'],
            alpha=0.75)
        axes[1].set_xlabel('Training step')
        axes[1].set_ylabel('Loss contribution')
        axes[1].set_title('Loss Component Breakdown', fontsize=10)
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Component logging\nnot available',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].axis('off')

    plt.tight_layout()
    save_fig(fig, log_dir, f'phase{phase}_loss_components.png')


def viz_all_phases_combined(phase_losses, log_dir):
    """
    Figure: all three phases concatenated on one timeline, with phase regions shaded.
    """
    if not any(phase_losses.values()):
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    phase_colors = {1: '#EF9A9A', 2: '#A5D6A7', 3: '#90CAF9'}
    phase_names  = {1: 'Phase 1: Goal', 2: 'Phase 2: Trajectory', 3: 'Phase 3: E2E'}
    offset = 0
    for phase in [1, 2, 3]:
        losses = phase_losses.get(phase, [])
        if not losses:
            continue
        steps = np.arange(offset, offset + len(losses))
        ax.axvspan(offset, offset + len(losses), alpha=0.15,
                   color=phase_colors[phase], label=phase_names[phase])
        ax.plot(steps, losses, alpha=0.3, linewidth=0.4,
                color=phase_colors[phase].strip('#'), rasterized=True)
        w = max(len(losses) // 60, 5)
        if len(losses) > w:
            sm = np.convolve(losses, np.ones(w)/w, mode='valid')
            ax.plot(np.arange(offset + w - 1, offset + len(losses)),
                    sm, linewidth=2, color='#37474F')
        offset += len(losses)

    ax.set_xlabel('Global training step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Combined Training Loss — All Three Phases', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, log_dir, 'training_all_phases.png')


def viz_val_metrics_over_training(val_history, log_dir):
    """
    Figure: minADE, minFDE, Miss Rate tracked over training phases.
    """
    if not val_history:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Validation Metrics over Training', fontsize=13, fontweight='bold')

    phases   = [v[0] for v in val_history]
    epochs   = [v[1] for v in val_history]
    ades     = [v[2] for v in val_history]
    fdes     = [v[3] for v in val_history]
    mrs      = [v[4] for v in val_history]
    x_labels = [f'P{p}E{e}' for p, e in zip(phases, epochs)]
    xs       = range(len(val_history))

    colors_by_phase = {1: '#E53935', 2: '#1E88E5', 3: '#43A047'}
    pt_colors = [colors_by_phase.get(p, 'gray') for p in phases]

    for ax, vals, title, ylabel in [
        (axes[0], ades, 'minADE', 'minADE (m)'),
        (axes[1], fdes, 'minFDE', 'minFDE (m)'),
        (axes[2], mrs,  'Miss Rate (2m threshold)', 'Miss Rate'),
    ]:
        ax.plot(xs, vals, 'k-', linewidth=1, alpha=0.5, zorder=1)
        ax.scatter(xs, vals, c=pt_colors, s=60, zorder=2)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=45, fontsize=7)
        ax.grid(True, alpha=0.3)

        # Phase boundary lines
        prev_phase = phases[0]
        for i, p in enumerate(phases):
            if p != prev_phase:
                ax.axvline(i - 0.5, color='gray', linestyle=':', linewidth=1)
                prev_phase = p

    # Legend
    handles = [mpatches.Patch(color=c, label=f'Phase {p}')
               for p, c in colors_by_phase.items()]
    axes[2].legend(handles=handles, fontsize=8)

    plt.tight_layout()
    save_fig(fig, log_dir, 'training_val_metrics.png')


# =============================================================================
# PREDICTION VISUALIZATIONS
# =============================================================================

def viz_predicted_trajectories(model, val_ds, device, log_dir, n_scenarios=3, n_agents=3):
    """
    Figure: 3×3 grid — K=6 predicted trajectories vs GT future.
    Rows = scenarios, cols = agents from that scenario.
    """
    model.eval()
    scenario_data = []

    with torch.no_grad():
        for tf_batch in val_ds.take(n_scenarios):
            tb  = tf_to_torch_batch(tf_batch, device=str(device))
            out = model(tb, phase=3)

            trajs = out['trajectories'].cpu().numpy()    # [B, 128, K, 80, 2]
            confs = out['confidences'].cpu().numpy()     # [B, 128, K]
            gt_f  = tb['gt_future_states'].cpu().numpy() # [B, 128, 91, 7]
            gt_v  = tb['gt_future_is_valid'].cpu().numpy()
            hist  = tb['input_states'].cpu().numpy()      # [B, 128, 11, 7]
            ttp   = tb['tracks_to_predict'].cpu().numpy()
            types = tb['object_type'].cpu().numpy()
            rg_xyz= tb['roadgraph_xyz'].cpu().numpy()
            rg_valid = tb['roadgraph_valid'].cpu().numpy()

            for b in range(tb['input_states'].shape[0]):
                ttp_agents = np.where(ttp[b])[0]
                if len(ttp_agents) < n_agents:
                    continue
                scenario_data.append({
                    'b': b,
                    'agents': ttp_agents[:n_agents],
                    'trajs': trajs[b], 'confs': confs[b],
                    'gt_f': gt_f[b], 'gt_v': gt_v[b],
                    'hist': hist[b], 'types': types[b],
                    'rg_xyz': rg_xyz[b], 'rg_valid': rg_valid[b, :, 0],
                })
                if len(scenario_data) >= n_scenarios:
                    break
            if len(scenario_data) >= n_scenarios:
                break

    if not scenario_data:
        print("  No suitable scenarios found for trajectory viz.")
        return

    n_sc   = min(len(scenario_data), n_scenarios)
    fig    = plt.figure(figsize=(5 * n_agents, 5 * n_sc))
    fig.suptitle('GB-Phys TrajNet: K=6 Predicted Trajectories vs Ground Truth',
                 fontsize=12, fontweight='bold', y=1.01)
    gs = gridspec.GridSpec(n_sc, n_agents, figure=fig, hspace=0.4, wspace=0.3)

    cmap = plt.colormaps['RdYlGn']

    for sc_i, dat in enumerate(scenario_data[:n_sc]):
        # Small roadgraph snippet for context
        rg_pts = dat['rg_xyz'][dat['rg_valid'] > 0, :2]

        for ag_i, a in enumerate(dat['agents']):
            ax = fig.add_subplot(gs[sc_i, ag_i])

            hist_xy = dat['hist'][a, :, :2]
            gt_pos  = dat['gt_f'][a, NUM_HISTORY_STEPS:, :2]
            gt_val  = dat['gt_v'][a, NUM_HISTORY_STEPS:].astype(bool)
            traj_k  = dat['trajs'][a]   # [K, 80, 2]
            conf_k  = dat['confs'][a]   # [K]
            atype   = int(dat['types'][a])

            # Centre the plot on agent's current position
            cx, cy = hist_xy[-1]
            window = 30

            # Crop roadgraph
            near = np.where((np.abs(rg_pts[:, 0] - cx) < window) &
                            (np.abs(rg_pts[:, 1] - cy) < window))[0]
            if len(near) > 0:
                ax.scatter(rg_pts[near, 0] - cx, rg_pts[near, 1] - cy,
                           s=0.5, c='#BDBDBD', alpha=0.4, linewidths=0)

            # History
            ax.plot(hist_xy[:, 0] - cx, hist_xy[:, 1] - cy,
                    'k-', lw=1.5, alpha=0.7, label='History')
            ax.plot(0, 0, 'ko', markersize=8, zorder=5)

            # GT future
            if gt_val.sum() > 3:
                ax.plot(gt_pos[gt_val, 0] - cx, gt_pos[gt_val, 1] - cy,
                        'g--', lw=2, alpha=0.8, label='GT future', zorder=4)
                ax.plot(gt_pos[gt_val][-1, 0] - cx, gt_pos[gt_val][-1, 1] - cy,
                        'g*', markersize=12, zorder=5)

            # K=6 predictions, colored by confidence
            norm  = Normalize(vmin=conf_k.min(), vmax=conf_k.max())
            order = np.argsort(-conf_k)
            for rank, k in enumerate(order):
                c = cmap(norm(conf_k[k]))
                lw = 2.5 if rank == 0 else 1.0
                ax.plot(traj_k[k, :, 0] - cx, traj_k[k, :, 1] - cy,
                        '-', color=c, lw=lw, alpha=0.85, zorder=3)
                ax.plot(traj_k[k, -1, 0] - cx, traj_k[k, -1, 1] - cy,
                        'o', color=c, markersize=5, zorder=4)

            ax.set_xlim(-window, window)
            ax.set_ylim(-window, window)
            ax.set_aspect('equal')
            ax.set_title(f'Sc{sc_i+1} Agent{a} ({TYPE_NAMES.get(atype, "?")})',
                         fontsize=8)
            ax.grid(True, alpha=0.15)
            if ag_i == 0:
                ax.set_ylabel(f'Scenario {sc_i + 1}', fontsize=8)
            if sc_i == 0 and ag_i == n_agents - 1:
                ax.legend(fontsize=6, loc='upper right')

    # Colorbar for confidence.
    # fig.colorbar() with ax=fig.axes steals space from existing axes, which
    # makes them incompatible with tight_layout and triggers a UserWarning.
    # Use subplots_adjust instead to leave a right-hand margin for the colorbar.
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Relative Confidence', fontsize=9)
    save_fig(fig, log_dir, 'pred_trajectories.png', dpi=150)


def viz_goal_candidates(model, val_ds, device, log_dir):
    """
    Figure: goal candidate sampling for one scenario/agent.
    Shows all N=64 candidates (sized by score), top-6 highlighted, GT endpoint.
    """
    model.eval()
    found = False

    with torch.no_grad():
        for tf_batch in val_ds.take(5):
            tb  = tf_to_torch_batch(tf_batch, device=str(device))
            out = model(tb, phase=3)

            ttp = tb['tracks_to_predict'].cpu().numpy()
            b   = 0
            cands = np.where(ttp[b])[0]
            if len(cands) == 0:
                continue

            hist   = tb['input_states'].cpu().numpy()[b]      # [128, 11, 7]
            gt_f   = tb['gt_future_states'].cpu().numpy()[b]  # [128, 91, 7]
            rg_xyz = tb['roadgraph_xyz'].cpu().numpy()[b]
            rg_v   = tb['roadgraph_valid'].cpu().numpy()[b, :, 0]

            # Access internals for goal visualization
            all_logits   = out.get('goal_logits')    # [B, 128, N]
            all_cands    = out.get('all_candidates')  # [B, 128, N, 4]
            goal_pos     = out.get('goal_positions')  # [B, 128, K, 2]

            if all_logits is None or all_cands is None:
                continue

            a = cands[0]
            logits_a = all_logits[b, a].cpu().numpy()    # [N]
            cands_a  = all_cands[b, a].cpu().numpy()     # [N, 4] (x,y,dx,dy)
            goals_a  = goal_pos[b, a].cpu().numpy()      # [K, 2]
            scores_a = np.exp(logits_a - logits_a.max())
            scores_a /= scores_a.sum()

            cx, cy   = hist[a, -1, 0], hist[a, -1, 1]
            gt_end   = gt_f[a, -1, :2]
            hist_xy  = hist[a, :, :2]
            window   = 40
            found = True

            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f'Goal Candidate Sampling — Agent {a} (Vehicle)',
                         fontsize=13, fontweight='bold')

            for ax in axes:
                near = np.where((np.abs(rg_xyz[:, 0] - cx) < window) &
                                (np.abs(rg_xyz[:, 1] - cy) < window) &
                                (rg_v > 0))[0]
                if len(near) > 0:
                    ax.scatter(rg_xyz[near, 0] - cx, rg_xyz[near, 1] - cy,
                               s=1, c='#BDBDBD', alpha=0.5, linewidths=0, zorder=1)
                ax.plot(hist_xy[:, 0] - cx, hist_xy[:, 1] - cy,
                        'k-', lw=2, alpha=0.8, label='History', zorder=3)
                ax.plot(0, 0, 'ko', markersize=8, zorder=4)
                ax.set_xlim(-window, window)
                ax.set_ylim(-window, window)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.15)

            # Left: all candidates colored by score
            sc_norm = Normalize(vmin=scores_a.min(), vmax=scores_a.max())
            scatter = axes[0].scatter(
                cands_a[:, 0] - cx, cands_a[:, 1] - cy,
                c=scores_a, cmap='YlOrRd', s=50 + 200 * scores_a / scores_a.max(),
                alpha=0.75, zorder=5, edgecolors='white', linewidths=0.5)
            axes[0].plot(gt_end[0] - cx, gt_end[1] - cy,
                         'b*', markersize=16, zorder=6, label='GT endpoint')
            fig.colorbar(scatter, ax=axes[0], label='Goal Score', shrink=0.7)
            axes[0].set_title(f'All {len(cands_a)} Candidates (size ∝ score)', fontsize=10)
            axes[0].legend(fontsize=8)

            # Right: top-K highlighted
            top_k_idx = np.argsort(-scores_a)[:DEFAULT_CONFIG['K']]
            axes[1].scatter(cands_a[:, 0] - cx, cands_a[:, 1] - cy,
                            c='#BDBDBD', s=20, alpha=0.4, zorder=3, label='Other candidates')
            for rank, idx in enumerate(top_k_idx):
                axes[1].plot(cands_a[idx, 0] - cx, cands_a[idx, 1] - cy,
                             'o', color=plt.colormaps['RdYlGn'](1 - rank / len(top_k_idx)),
                             markersize=12, zorder=5)
                axes[1].annotate(f'K{rank+1}\n{scores_a[idx]:.3f}',
                                 (cands_a[idx, 0] - cx, cands_a[idx, 1] - cy),
                                 textcoords='offset points', xytext=(5, 5), fontsize=7)
            axes[1].plot(gt_end[0] - cx, gt_end[1] - cy,
                         'b*', markersize=16, zorder=6, label='GT endpoint')
            axes[1].set_title(f'Top K={DEFAULT_CONFIG["K"]} Goals', fontsize=10)
            axes[1].legend(fontsize=8)

            plt.tight_layout()
            save_fig(fig, log_dir, 'pred_goal_candidates.png', dpi=150)
            break


def viz_confidence_and_winner(model, val_ds, device, log_dir, max_batches=20):
    """
    Figure: confidence score distribution + winner mode histogram.
    """
    model.eval()
    all_confs, winner_modes, winner_confs, nonwinner_confs = [], [], [], []

    with torch.no_grad():
        for idx, tf_batch in enumerate(val_ds.take(max_batches)):
            tb  = tf_to_torch_batch(tf_batch, device=str(device))
            out = model(tb, phase=3)

            trajs       = out['trajectories'].cpu().numpy()  # [B, 128, K, 80, 2]
            confs       = out['confidences'].cpu().numpy()   # [B, 128, K]
            gt_f        = tb['gt_future_states'].cpu().numpy()
            gt_v        = tb['gt_future_is_valid'].cpu().numpy()
            ttp         = tb['tracks_to_predict'].cpu().numpy()

            K = trajs.shape[2]
            B = trajs.shape[0]
            for b in range(B):
                for a in range(NUM_AGENTS):
                    if not ttp[b, a]:
                        continue
                    gt_pos = gt_f[b, a, NUM_HISTORY_STEPS:, :2]
                    all_confs.extend(confs[b, a].tolist())
                    fdes = [np.linalg.norm(trajs[b, a, k, -1] - gt_pos[-1])
                            for k in range(K)]
                    w = int(np.argmin(fdes))
                    winner_modes.append(w)
                    winner_confs.append(confs[b, a, w])
                    nonwinner_confs.extend([confs[b, a, k]
                                            for k in range(K) if k != w])

    K = DEFAULT_CONFIG['K']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Confidence Score Analysis', fontsize=13, fontweight='bold')

    # Confidence distribution
    axes[0].hist(all_confs, bins=40, color='steelblue', edgecolor='none', alpha=0.8)
    axes[0].axvline(1 / K, color='red', linestyle='--', linewidth=2,
                    label=f'Uniform (1/K={1/K:.3f})')
    axes[0].set_xlabel('Confidence score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('All Confidence Scores\n(all K modes, all agents)', fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Winner mode histogram
    mode_counts = np.bincount(winner_modes, minlength=K)
    colors_k    = [plt.colormaps['tab10'](i / K) for i in range(K)]
    bars = axes[1].bar(range(K), mode_counts, color=colors_k,
                       edgecolor='white', alpha=0.85)
    axes[1].axhline(len(winner_modes) / K, color='red', linestyle='--',
                    linewidth=2, label='Uniform (ideal)')
    axes[1].set_xlabel('Mode k')
    axes[1].set_ylabel('Times selected as winner (min FDE)')
    axes[1].set_title('Winner Mode Distribution\n(uniform = model uses all modes equally)', fontsize=10)
    axes[1].set_xticks(range(K))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, cnt in zip(bars, mode_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(cnt), ha='center', va='bottom', fontsize=8)

    # Winner vs non-winner confidence boxplot
    axes[2].boxplot([winner_confs, nonwinner_confs],
                    labels=['Winner mode', 'Non-winner modes'],
                    patch_artist=True,
                    boxprops=dict(facecolor='#A5D6A7'),
                    medianprops=dict(color='black', linewidth=2))
    axes[2].set_ylabel('Confidence score')
    axes[2].set_title('Confidence: Winner vs Non-Winner\n'
                      '(trained model: winner should rank higher)', fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig(fig, log_dir, 'pred_confidence_dist.png')


def viz_cv_vs_gbphys(cv_model, gbphys_model, val_ds, device, log_dir, n_agents=4):
    """
    Figure: direct side-by-side comparison of CV vs GB-Phys TrajNet
    for the same set of agents.
    """
    gbphys_model.eval()
    found_data = None

    with torch.no_grad():
        for tf_batch in val_ds.take(5):
            tb    = tf_to_torch_batch(tf_batch, device=str(device))
            ttp   = tb['tracks_to_predict'].cpu().numpy()
            cands = np.where(ttp[0])[0]
            if len(cands) < n_agents:
                continue
            found_data  = (tb, cands[:n_agents])
            break

    if found_data is None:
        return
    tb, agents = found_data

    cv_out     = cv_model.predict(tb)
    with torch.no_grad():
        gbphys_out = gbphys_model(tb, phase=3)

    # .detach() is required before .numpy() because model output tensors
    # retain the computation graph (requires_grad=True) from their parameters.
    cv_trajs     = cv_out['trajectories'].detach().cpu().numpy()[0]
    gbphys_trajs = gbphys_out['trajectories'].detach().cpu().numpy()[0]
    gbphys_confs = gbphys_out['confidences'].detach().cpu().numpy()[0]
    gt_f         = tb['gt_future_states'].detach().cpu().numpy()[0]
    gt_v         = tb['gt_future_is_valid'].detach().cpu().numpy()[0]
    hist         = tb['input_states'].detach().cpu().numpy()[0]
    types        = tb['object_type'].detach().cpu().numpy()[0]
    rg_xyz       = tb['roadgraph_xyz'].detach().cpu().numpy()[0]
    rg_valid     = tb['roadgraph_valid'].detach().cpu().numpy()[0, :, 0]

    fig, axes = plt.subplots(2, n_agents, figsize=(5 * n_agents, 10))
    fig.suptitle('CV Baseline vs GB-Phys TrajNet — Same Agents',
                 fontsize=13, fontweight='bold')
    row_labels = ['Constant Velocity', 'GB-Phys TrajNet']
    window = 30

    for col_i, a in enumerate(agents):
        cx, cy  = hist[a, -1, 0], hist[a, -1, 1]
        hist_xy = hist[a, :, :2]
        gt_pos  = gt_f[a, NUM_HISTORY_STEPS:, :2]
        gt_val  = gt_v[a, NUM_HISTORY_STEPS:].astype(bool)
        atype   = int(types[a])

        near = np.where((np.abs(rg_xyz[:, 0] - cx) < window) &
                        (np.abs(rg_xyz[:, 1] - cy) < window) &
                        (rg_valid > 0))[0]

        for row_i, (trajs_src, confs_src, row_color) in enumerate([
            (cv_trajs,     np.ones(6) / 6, '#EF9A9A'),
            (gbphys_trajs, gbphys_confs[a], '#A5D6A7')
        ]):
            ax = axes[row_i, col_i]
            if len(near) > 0:
                ax.scatter(rg_xyz[near, 0] - cx, rg_xyz[near, 1] - cy,
                           s=0.5, c='#BDBDBD', alpha=0.4, linewidths=0)
            ax.plot(hist_xy[:, 0] - cx, hist_xy[:, 1] - cy,
                    'k-', lw=1.5, alpha=0.7)
            ax.plot(0, 0, 'ko', markersize=7)
            if gt_val.sum() > 3:
                ax.plot(gt_pos[gt_val, 0] - cx, gt_pos[gt_val, 1] - cy,
                        'g--', lw=2, alpha=0.8)
                ax.plot(gt_pos[gt_val][-1, 0] - cx, gt_pos[gt_val][-1, 1] - cy,
                        'g*', markersize=10)
            traj_a = trajs_src[a] if row_i == 0 else trajs_src
            norm = Normalize(vmin=confs_src.min(), vmax=confs_src.max())
            for k in range(min(6, traj_a.shape[0])):
                c = plt.colormaps['RdYlGn'](norm(confs_src[k]))
                ax.plot(traj_a[k, :, 0] - cx, traj_a[k, :, 1] - cy,
                        '-', color=c, lw=1.5, alpha=0.8)
            ax.set_xlim(-window, window)
            ax.set_ylim(-window, window)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.15)
            if col_i == 0:
                ax.set_ylabel(row_labels[row_i], fontsize=9, fontweight='bold',
                              color='darkred' if row_i == 0 else 'darkgreen')
            ax.set_title(f'{TYPE_NAMES.get(atype, "?")} #{a}', fontsize=8)

    plt.tight_layout()
    save_fig(fig, log_dir, 'pred_cv_vs_gbphys.png', dpi=150)


# =============================================================================
# ERROR ANALYSIS VISUALIZATIONS
# =============================================================================

def viz_error_analysis(model, cv_model, val_ds, device, log_dir, max_batches=50):
    """
    Figures:
      error_ade_histogram.png  — ADE distribution by type (CV vs GB-Phys)
      error_ade_vs_speed.png   — minADE vs agent speed
      error_miss_rate_by_type.png — MR at 3s/5s/8s per agent type
    """
    model.eval()

    records = {
        'type': [], 'speed': [], 'ade': [], 'fde': [], 'mr': [],
        'ade_cv': [], 'fde_cv': [],
        'ade_3s': [], 'fde_3s': [],
        'ade_5s': [], 'fde_5s': [],
        'mr_3s': [], 'mr_5s': [], 'mr_8s': [],
    }

    HORIZON_3S = 30   # index 30 in 80-step future at 10Hz = 3s
    HORIZON_5S = 50
    HORIZON_8S = 79

    with torch.no_grad():
        for idx, tf_batch in enumerate(val_ds.take(max_batches)):
            tb       = tf_to_torch_batch(tf_batch, device=str(device))
            out      = model(tb, phase=3)
            cv_out   = cv_model.predict(tb)

            trajs    = out['trajectories'].cpu().numpy()
            cv_trajs = cv_out['trajectories'].cpu().numpy()
            gt_f     = tb['gt_future_states'].cpu().numpy()
            gt_v     = tb['gt_future_is_valid'].cpu().numpy()
            ttp      = tb['tracks_to_predict'].cpu().numpy()
            types    = tb['object_type'].cpu().numpy()
            states   = tb['input_states'].cpu().numpy()

            B = trajs.shape[0]
            for b in range(B):
                for a in range(NUM_AGENTS):
                    if not ttp[b, a]:
                        continue
                    gt_pos = gt_f[b, a, NUM_HISTORY_STEPS:, :2]
                    gt_val = gt_v[b, a, NUM_HISTORY_STEPS:].astype(float)
                    if gt_val.sum() < 10:
                        continue

                    atype = int(types[b, a])
                    vx    = states[b, a, -1, 5]
                    vy    = states[b, a, -1, 6]
                    speed = float(np.sqrt(vx**2 + vy**2))

                    pred_k    = trajs[b, a]       # [K, 80, 2]
                    cv_pred_k = cv_trajs[b, a]

                    # Full 8s
                    ade8 = compute_minADE(pred_k, gt_pos, gt_val)
                    fde8 = compute_minFDE(pred_k, gt_pos, gt_val)
                    mr8  = compute_miss_rate(pred_k, gt_pos, gt_val, threshold=2.0)

                    ade_cv = compute_minADE(cv_pred_k, gt_pos, gt_val)
                    fde_cv = compute_minFDE(cv_pred_k, gt_pos, gt_val)

                    # Sub-horizons
                    v3 = gt_val.copy(); v3[HORIZON_3S+1:] = 0.0
                    v5 = gt_val.copy(); v5[HORIZON_5S+1:] = 0.0
                    ade3  = compute_minADE(pred_k, gt_pos, v3) if v3.sum() > 0 else np.nan
                    fde3  = compute_minFDE(pred_k, gt_pos, v3) if v3.sum() > 0 else np.nan
                    ade5  = compute_minADE(pred_k, gt_pos, v5) if v5.sum() > 0 else np.nan
                    fde5  = compute_minFDE(pred_k, gt_pos, v5) if v5.sum() > 0 else np.nan
                    mr3   = compute_miss_rate(pred_k, gt_pos, v3, threshold=1.0) if v3.sum() > 0 else np.nan
                    mr5   = compute_miss_rate(pred_k, gt_pos, v5, threshold=1.8) if v5.sum() > 0 else np.nan

                    records['type'].append(atype)
                    records['speed'].append(speed)
                    records['ade'].append(ade8)
                    records['fde'].append(fde8)
                    records['mr'].append(mr8)
                    records['ade_cv'].append(ade_cv)
                    records['fde_cv'].append(fde_cv)
                    records['ade_3s'].append(ade3)
                    records['fde_3s'].append(fde3)
                    records['ade_5s'].append(ade5)
                    records['fde_5s'].append(fde5)
                    records['mr_3s'].append(mr3)
                    records['mr_5s'].append(mr5)
                    records['mr_8s'].append(mr8)

    types_np = np.array(records['type'])
    speeds_np = np.array(records['speed'])
    n_total   = len(records['ade'])
    print(f"  Error analysis: {n_total:,} agents evaluated")

    type_codes = [TYPE_VEHICLE, TYPE_PEDESTRIAN, TYPE_CYCLIST]

    # ── Figure 1: ADE histogram by type (CV vs GB-Phys) ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('minADE Distribution: CV vs GB-Phys TrajNet', fontsize=13, fontweight='bold')

    for ax, atype in zip(axes, type_codes):
        mask = (types_np == atype)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            continue
        ade_gbp = np.array(records['ade'])[mask]
        ade_cv  = np.array(records['ade_cv'])[mask]
        max_x   = np.percentile(np.concatenate([ade_gbp, ade_cv]), 95)
        bins    = np.linspace(0, max_x, 35)
        ax.hist(ade_cv,  bins=bins, alpha=0.6, color='#EF9A9A', label=f'CV  (mean={np.mean(ade_cv):.2f}m)')
        ax.hist(ade_gbp, bins=bins, alpha=0.6, color='#A5D6A7',
                label=f'GB-Phys (mean={np.mean(ade_gbp):.2f}m)')
        ax.set_xlabel('minADE (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{TYPE_NAMES[atype]} (n={mask.sum():,})', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, log_dir, 'error_ade_histogram.png')

    # ── Figure 2: ADE vs speed scatter ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    for atype in type_codes:
        mask  = (types_np == atype)
        if mask.sum() == 0:
            continue
        ax.scatter(speeds_np[mask], np.array(records['ade'])[mask],
                   alpha=0.3, s=8, c=TYPE_COLORS[atype],
                   label=TYPE_NAMES[atype], linewidths=0)

    # Speed bin means
    bins_e = np.arange(0, speeds_np.max() + 2, 2)
    bin_idx = np.digitize(speeds_np, bins_e)
    bin_means = [np.mean(np.array(records['ade'])[bin_idx == i])
                 for i in range(1, len(bins_e))
                 if np.sum(bin_idx == i) > 5]
    bin_centers = [(bins_e[i] + bins_e[i+1]) / 2
                   for i in range(len(bins_e) - 1)
                   if np.sum(bin_idx == i + 1) > 5]
    if bin_means:
        ax.plot(bin_centers, bin_means, 'k-o', lw=2, markersize=5,
                label='Mean ADE per speed bin', zorder=5)

    ax.set_xlabel('Agent Speed (m/s)')
    ax.set_ylabel('minADE (m)')
    ax.set_title('minADE vs Agent Current Speed', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Speed band annotations
    ax.axvspan(0,   1.4, alpha=0.07, color='blue',  label='Stationary')
    ax.axvspan(1.4, 11,  alpha=0.07, color='green', label='Normal')
    ax.axvspan(11,  ax.get_xlim()[1] if ax.get_xlim()[1] > 11 else 30,
               alpha=0.07, color='red', label='Fast')
    plt.tight_layout()
    save_fig(fig, log_dir, 'error_ade_vs_speed.png')

    # ── Figure 3: Miss rate per type at 3s/5s/8s ────────────────────────────
    horizons   = ['3s (lat 1.0m)', '5s (lat 1.8m)', '8s (lat 2.0m)']
    mr_keys    = ['mr_3s', 'mr_5s', 'mr_8s']
    x          = np.arange(3)
    width      = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for ti, atype in enumerate(type_codes):
        mask   = (types_np == atype)
        if mask.sum() == 0:
            continue
        mrs = [np.nanmean(np.array(records[k])[mask]) for k in mr_keys]
        ax.bar(x + ti * width, mrs, width, label=TYPE_NAMES[atype],
               color=TYPE_COLORS[atype], edgecolor='white', alpha=0.85)
        for xi, mr in zip(x + ti * width, mrs):
            ax.text(xi, mr + 0.005, f'{mr:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels(horizons)
    ax.set_ylabel('Miss Rate')
    ax.set_ylim(0, 1.05)
    ax.set_title('Miss Rate per Agent Type at 3s / 5s / 8s Horizons', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, log_dir, 'error_miss_rate_by_type.png')


# =============================================================================
# OFFICIAL METRICS VISUALIZATION
# =============================================================================

def viz_official_metrics_heatmap(result_tensor, breakdown_names, log_dir):
    """
    Figure: official MotionMetrics as a heatmap (breakdown × metric type).
    """
    METRIC_TYPES = ['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'SoftmAP']
    n_metrics    = len(METRIC_TYPES)
    n_breakdowns = result_tensor.shape[1]

    data = np.array([[float(result_tensor[i, j])
                      for j in range(n_breakdowns)]
                     for i in range(n_metrics)])

    # Shorten breakdown labels
    short_names = [b.replace('TYPE_', '').replace('_SPEED_', '\n').lower()
                   for b in breakdown_names]

    fig, ax = plt.subplots(figsize=(max(12, n_breakdowns * 0.9), 5))
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(n_breakdowns))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(METRIC_TYPES, fontsize=9)
    ax.set_title('Official MotionMetrics Breakdown Heatmap\n'
                 '(green = better for MissRate/OverlapRate; higher = better for mAP)',
                 fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(n_metrics):
        for j in range(n_breakdowns):
            val = data[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=7, color='black' if 0.2 < val < 0.8 else 'white')

    plt.tight_layout()
    save_fig(fig, log_dir, 'metrics_official_heatmap.png')


def viz_comparison_bar(cv_metrics, gbphys_metrics, log_dir,
                       official_gbphys=None, official_cv=None):
    """
    Figure: side-by-side bar chart comparing CV and GB-Phys on all metrics.
    """
    metrics = {}
    metrics['minADE (m)']      = [np.mean(cv_metrics['minADE']),
                                   gbphys_metrics['minADE']]
    metrics['minFDE (m)']      = [np.mean(cv_metrics['minFDE']),
                                   gbphys_metrics['minFDE']]
    metrics['Miss Rate']       = [np.mean(cv_metrics['miss_rate']),
                                   gbphys_metrics['miss_rate']]

    labels = ['Constant\nVelocity', 'GB-Phys\nTrajNet']
    colors = ['#EF9A9A', '#A5D6A7']
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    fig.suptitle('Baseline Comparison: Constant Velocity vs GB-Phys TrajNet',
                 fontsize=13, fontweight='bold')

    for ax, (metric_name, vals) in zip(axes, metrics.items()):
        bars = ax.bar(labels, vals, color=colors, edgecolor='white',
                      linewidth=1.5, alpha=0.9, width=0.5)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        # Improvement arrow
        if vals[1] < vals[0]:
            impr = (vals[0] - vals[1]) / vals[0] * 100
            ax.annotate(f'↓ {impr:.1f}% better',
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', fontsize=9, color='darkgreen',
                        fontweight='bold')
        else:
            deter = (vals[1] - vals[0]) / vals[0] * 100
            ax.annotate(f'↑ {deter:.1f}% worse',
                        xy=(0.5, 0.95), xycoords='axes fraction',
                        ha='center', fontsize=9, color='darkred',
                        fontweight='bold')

    plt.tight_layout()
    save_fig(fig, log_dir, 'metrics_comparison_bar.png')


# =============================================================================
# evaluate_quick (collects per-type data too)
# =============================================================================

def evaluate_full(model, val_ds, device, max_batches=50):
    model.eval()
    results = defaultdict(list)

    with torch.no_grad():
        for idx, tf_batch in enumerate(val_ds.take(max_batches)):
            tb  = tf_to_torch_batch(tf_batch, device=str(device))
            out = model(tb, phase=3)

            trajs = out['trajectories'].cpu().numpy()
            ttp   = tb['tracks_to_predict'].cpu().numpy()
            gt_f  = tb['gt_future_states'].cpu().numpy()
            gt_v  = tb['gt_future_is_valid'].cpu().numpy()
            types = tb['object_type'].cpu().numpy()

            gt_pos = gt_f[:, :, NUM_HISTORY_STEPS:, 0:2]
            gt_val = gt_v[:, :, NUM_HISTORY_STEPS:]
            B, A   = ttp.shape

            for b in range(B):
                for a in range(A):
                    if not ttp[b, a] or gt_val[b, a].sum() < 5:
                        continue
                    p     = trajs[b, a]
                    g     = gt_pos[b, a]
                    v     = gt_val[b, a].astype(float)
                    atype = int(types[b, a])
                    ade   = compute_minADE(p, g, v)
                    fde   = compute_minFDE(p, g, v)
                    mr    = compute_miss_rate(p, g, v, threshold=2.0)
                    results['all_ade'].append(ade)
                    results['all_fde'].append(fde)
                    results['all_mr'].append(mr)
                    results[f'{TYPE_NAMES.get(atype,"unk").lower()}_ade'].append(ade)

            if (idx + 1) % 10 == 0:
                print(f"    batch {idx+1}/{max_batches}  "
                      f"minADE={np.mean(results['all_ade']):.3f}")

    return {
        'minADE': float(np.mean(results['all_ade'])) if results['all_ade'] else float('nan'),
        'minFDE': float(np.mean(results['all_fde'])) if results['all_fde'] else float('nan'),
        'miss_rate': float(np.mean(results['all_mr'])) if results['all_mr'] else float('nan'),
        'n_agents': len(results['all_ade']),
    }


# =============================================================================
# Training (with component logging)
# =============================================================================

def train_phase(model, phase, train_files, val_ds, args, device,
                val_history, phase_losses):
    phase_configs = {
        1: {'lr': args.phase1_lr, 'epochs': args.phase1_epochs, 'name': 'Goal Prediction'},
        2: {'lr': args.phase2_lr, 'epochs': args.phase2_epochs, 'name': 'Trajectory Generation'},
        3: {'lr': args.phase3_lr, 'epochs': args.phase3_epochs, 'name': 'End-to-End'},
    }
    cfg     = phase_configs[phase]
    ckpt_dir = os.path.join(args.checkpoint_dir, f'phase{phase}')
    os.makedirs(ckpt_dir, exist_ok=True)

    section_header(f"Phase {phase}: {cfg['name']} "
                   f"(lr={cfg['lr']}, epochs={cfg['epochs']})")

    if phase == 1:
        model.freeze_stage2()
        print("  Stage 2 frozen")
    elif phase == 2:
        model.freeze_stage1()
        print("  Stage 1 frozen")
    else:
        model.unfreeze_all()
        print("  All parameters unfrozen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr'], weight_decay=1e-4,
    )
    total_steps = len(train_files) * cfg['epochs'] * 487 // args.batch_size
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps // args.accumulation_steps, 1), eta_min=1e-6)

    criterion = CombinedLoss(
        lambda_goal=1.0,
        lambda_wta=1.0 if phase >= 2 else 0.0,
        lambda_physics=0.1 if phase >= 2 else 0.0,
        lambda_boundary=0.5 if phase >= 2 else 0.0,
    )

    # AMP (fp16) is disabled for all phases.
    #
    # Phase 1: goal logits overflow fp16 → NaN cross-entropy loss.
    # Phase 2: WTA L2 loss on absolute world coordinates is ~1e7,
    #          which when scaled by GradScaler (×65536) = ~7e11, far
    #          exceeding fp16 max (65504).  GradScaler detects the inf
    #          gradient and skips every optimizer step → completely flat loss.
    # Phase 3: same risk as Phase 2.
    #
    # fp32 is used throughout.  The A100 is fast enough in fp32 for this model.
    use_amp = False
    scaler  = GradScaler('cuda', enabled=False)
    print(f"  AMP: disabled (fp32 throughout — prevents NaN/inf gradient overflow)")

    best_val_loss       = float('inf')
    all_losses          = []
    loss_components_log = []
    global_step         = 0
    nan_batches_skipped = 0
    val_log_for_plot    = []   # initialised here to prevent UnboundLocalError if
                               # the epoch loop exits before the first validation run

    for epoch in range(cfg['epochs']):
        model.train()
        ds = build_dataset(train_files, args.batch_size, shuffle=True)
        epoch_losses = []
        epoch_start  = time.time()
        optimizer.zero_grad()

        for step_in_epoch, tf_batch in enumerate(ds):
            try:
                tb = tf_to_torch_batch(tf_batch, device=str(device))

                with autocast('cuda', enabled=use_amp):
                    out    = model(tb, phase=phase)

                    # One-time diagnostic for first batch of first epoch per phase
                    if step_in_epoch == 0 and epoch == 0:
                        with torch.no_grad():
                            trajs  = out['trajectories']                       # [B, A, K, T, 2]
                            gt_fut = tb['gt_future_states'][:, :, 11:, 0:2]   # [B, A, 80, 2]
                            cp     = tb['input_states'][:, :, -1, 0:2]        # [B, A, 2]
                            p_rel  = trajs - cp.unsqueeze(2).unsqueeze(2)
                            g_rel  = gt_fut - cp.unsqueeze(2)
                            print(f"\n  [DIAG Phase {phase}]"
                                  f"  cur_pos   : {cp.min().item():8.1f} .. {cp.max().item():8.1f}"
                                  f"  pred_trajs: {trajs.min().item():8.1f} .. {trajs.max().item():8.1f}"
                                  f"  gt_future : {gt_fut.min().item():8.1f} .. {gt_fut.max().item():8.1f}"
                                  f"  pred_rel  : {p_rel.min().item():8.1f} .. {p_rel.max().item():8.1f}"
                                  f"  gt_rel    : {g_rel.min().item():8.1f} .. {g_rel.max().item():8.1f}\n")

                    losses = criterion(out, tb, phase=phase)
                    loss   = losses['total'] / args.accumulation_steps

                # Skip batch if loss is NaN (can occur on malformed data shards)
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches_skipped += 1
                    optimizer.zero_grad()
                    if nan_batches_skipped % 10 == 1:
                        print(f"    [WARN] NaN/Inf loss at step {step_in_epoch} "
                              f"(skipped {nan_batches_skipped} batches so far)")
                    continue

            except torch.OutOfMemoryError:
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                nan_batches_skipped += 1
                if nan_batches_skipped % 10 == 1:
                    print(f"    [WARN] CUDA OOM at step {step_in_epoch} — "
                          f"batch skipped ({nan_batches_skipped} skips so far). "
                          f"Clearing cache and continuing.")
                continue

            scaler.scale(loss).backward()

            if (step_in_epoch + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                # scaler.step() returns None if it skips due to inf/NaN grads
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # Only step scheduler if optimizer actually stepped
                if scaler.get_scale() == scale_before:
                    scheduler.step()
                global_step += 1

            epoch_losses.append(losses['total'].item())
            loss_components_log.append({
                'goal':     losses.get('goal',     torch.tensor(0.)).item(),
                'wta':      losses.get('wta',      torch.tensor(0.)).item(),
                'physics':  losses.get('physics',  torch.tensor(0.)).item(),
                'boundary': losses.get('boundary', torch.tensor(0.)).item(),
            })

            if (step_in_epoch + 1) % 50 == 0:
                avg    = np.mean(epoch_losses[-50:]) if epoch_losses else float('nan')
                lr_now = scheduler.get_last_lr()[0]
                # Per-component breakdown (last 50 steps average)
                recent = loss_components_log[-50:]
                avg_goal    = np.mean([c['goal']     for c in recent])
                avg_wta     = np.mean([c['wta']      for c in recent])
                avg_physics = np.mean([c['physics']  for c in recent])
                avg_bound   = np.mean([c['boundary'] for c in recent])
                print(f"    Epoch {epoch+1}/{cfg['epochs']}  "
                      f"step {step_in_epoch+1}  "
                      f"loss={avg:.1f}  lr={lr_now:.2e}  "
                      f"[goal={avg_goal:.1f}  wta={avg_wta:.1f}  "
                      f"phys={avg_physics:.1f}  bound={avg_bound:.1f}]")

        epoch_avg = np.mean(epoch_losses)
        elapsed   = time.time() - epoch_start
        all_losses.extend(epoch_losses)
        print(f"  Epoch {epoch+1}/{cfg['epochs']} complete — "
              f"avg_loss={epoch_avg:.4f}  time={elapsed:.0f}s")

        save_checkpoint(model, optimizer, epoch, global_step, epoch_avg,
                        ckpt_dir, scheduler=scheduler)

        if (epoch + 1) % 5 == 0 or epoch == cfg['epochs'] - 1:
            val_m   = evaluate_full(model, val_ds, device, max_batches=20)
            val_loss = val_m['minADE']
            print(f"  Val: minADE={val_m['minADE']:.4f}  "
                  f"minFDE={val_m['minFDE']:.4f}  "
                  f"MR={val_m['miss_rate']:.4f}  (n={val_m['n_agents']})")
            val_history.append((phase, epoch + 1,
                                 val_m['minADE'], val_m['minFDE'], val_m['miss_rate']))
            # For plotting vs step
            val_log_for_plot = [(idx * len(train_files) * 487 // args.batch_size,
                                  v[2]) for idx, v in enumerate(val_history) if v[0] == phase]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint(model, optimizer, epoch, global_step,
                                     val_loss, ckpt_dir, metric_name='minADE')
                print(f"  New best! minADE={val_loss:.4f}")

    phase_losses[phase] = all_losses
    viz_loss_curve(phase, cfg['name'], all_losses, loss_components_log,
                   val_log_for_plot, args.log_dir)
    return best_val_loss


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='GB-Phys TrajNet full pipeline')
    parser.add_argument('--num_shards', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs')
    parser.add_argument('--submission_dir', type=str, default='results/submissions')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phase1_epochs', type=int, default=15)
    parser.add_argument('--phase2_epochs', type=int, default=20)
    parser.add_argument('--phase3_epochs', type=int, default=15)
    parser.add_argument('--phase1_lr', type=float, default=5e-4)
    parser.add_argument('--phase2_lr', type=float, default=2e-4)
    parser.add_argument('--phase3_lr', type=float, default=1e-4)
    parser.add_argument('--eval_batches', type=int, default=200)
    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--start_phase', type=int, default=1, choices=[1, 2, 3],
                        help='Start training from this phase (1=default, 2=skip Phase 1, '
                             '3=skip Phases 1+2). Phase N-1 best checkpoint is loaded '
                             'automatically before starting.')
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Device: {device}  ({torch.cuda.device_count()} GPU(s))")
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print(f"Device: {device}")

    for d in [args.checkpoint_dir, args.log_dir, args.submission_dir]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # Section 1: Data pipeline + data visualizations
    # =========================================================================
    section_header("Section 1: Data Pipeline & Visualizations")

    train_files = get_train_files(args.num_shards)
    val_files   = sorted(tf.io.gfile.glob(GCS_VAL))
    print(f"Validation shards available: {len(val_files)}")

    # Gather a few batches for data visualizations
    data_ds = build_dataset(train_files[:3], batch_size=4, shuffle=False)
    batches_np = None
    for tf_batch in data_ds.take(1):
        tb = tf_to_torch_batch(tf_batch, device='cpu')
        batches_np = {k: v.numpy() for k, v in tb.items() if hasattr(v, 'numpy')}
        print("\nSample batch shapes:")
        for k, v in batches_np.items():
            print(f"  {k:35s}: {v.shape}")

    if batches_np is not None:
        print("\nGenerating data visualizations...")
        for viz_fn, viz_name in [
            (lambda: viz_scenario_overview(batches_np, args.log_dir, n_scenarios=3),  'scenario_overview'),
            (lambda: viz_agent_type_distribution(batches_np, args.log_dir),            'agent_type_dist'),
            (lambda: viz_velocity_distribution(batches_np, args.log_dir),              'velocity_dist'),
            (lambda: viz_tracks_to_predict(batches_np, args.log_dir),                  'tracks_to_predict'),
            (lambda: viz_roadgraph_types(batches_np, args.log_dir),                    'roadgraph_types'),
        ]:
            try:
                viz_fn()
            except Exception as e:
                print(f"  [WARN] viz_{viz_name} failed (non-fatal): {e}")

    # Validation dataset reused throughout
    val_ds = build_dataset(val_files[:20], batch_size=args.batch_size, shuffle=False)

    # =========================================================================
    # Section 2: Baselines
    # =========================================================================
    section_header("Section 2: Constant Velocity Baseline")

    cv = ConstantVelocityBaseline(T=80, dt=0.1, K=6)
    cv_metrics = {'minADE': [], 'minFDE': [], 'miss_rate': []}

    for tf_batch in val_ds.take(20):
        tb     = tf_to_torch_batch(tf_batch, device='cpu')
        cv_out = cv.predict(tb)
        trajs  = cv_out['trajectories'].numpy()
        gt_f   = tb['gt_future_states'].numpy()
        gt_v   = tb['gt_future_is_valid'].numpy()
        ttp    = tb['tracks_to_predict'].numpy()
        gt_pos = gt_f[:, :, NUM_HISTORY_STEPS:, 0:2]
        gt_val = gt_v[:, :, NUM_HISTORY_STEPS:]
        B, A   = ttp.shape
        for b in range(B):
            for a in range(A):
                if not ttp[b, a] or gt_val[b, a].sum() < 5:
                    continue
                cv_metrics['minADE'].append(
                    compute_minADE(trajs[b,a], gt_pos[b,a], gt_val[b,a].astype(float)))
                cv_metrics['minFDE'].append(
                    compute_minFDE(trajs[b,a], gt_pos[b,a], gt_val[b,a].astype(float)))
                cv_metrics['miss_rate'].append(
                    compute_miss_rate(trajs[b,a], gt_pos[b,a], gt_val[b,a].astype(float)))

    if cv_metrics['minADE']:
        print(f"  CV minADE:  {np.mean(cv_metrics['minADE']):.4f} m")
        print(f"  CV minFDE:  {np.mean(cv_metrics['minFDE']):.4f} m")
        print(f"  CV MR:      {np.mean(cv_metrics['miss_rate']):.4f}")
        print(f"  CV agents:  {len(cv_metrics['minADE'])}")
    else:
        print("  [WARN] No valid CV agents found in validation subset.")

    # =========================================================================
    # Section 3: Model
    # =========================================================================
    section_header("Section 3: GB-Phys TrajNet Architecture")

    model = GBPhysTrajNet(config=DEFAULT_CONFIG).to(device)
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"  K={DEFAULT_CONFIG['K']}  N_candidates={DEFAULT_CONFIG['N_candidates']}"
          f"  T={DEFAULT_CONFIG['T']}  gru_h={DEFAULT_CONFIG['gru_hidden_dim']}")

    if args.resume_from and os.path.exists(args.resume_from):
        opt_tmp = optim.AdamW(model.parameters(), lr=1e-4)
        ep, st, ls = load_checkpoint(model, opt_tmp, args.resume_from, device=str(device))
        print(f"Resumed from {args.resume_from} (epoch={ep}, step={st})")

    # =========================================================================
    # Section 4: Training
    # =========================================================================
    val_history  = []    # (phase, epoch, minADE, minFDE, MR) tuples
    phase_losses = {}    # phase -> list of step losses

    if not args.skip_training:
        section_header("Section 4: Three-Phase Training")
        print(f"  Shards: {args.num_shards}  |  "
              f"batch: {args.batch_size}  |  "
              f"eff. batch: {args.batch_size * args.accumulation_steps}")
        print(f"  Phase 1: {args.phase1_epochs} epochs @ {args.phase1_lr}")
        print(f"  Phase 2: {args.phase2_epochs} epochs @ {args.phase2_lr}")
        print(f"  Phase 3: {args.phase3_epochs} epochs @ {args.phase3_lr}\n")

        for phase in [1, 2, 3]:
            if phase < args.start_phase:
                print(f"  Skipping Phase {phase} (--start_phase={args.start_phase})")
                continue
            if phase > 1:
                prev = os.path.join(args.checkpoint_dir, f'phase{phase-1}', 'best_minADE.pt')
                if os.path.exists(prev):
                    # Load model weights only — optimizer state is NOT reusable
                    # across phases because each phase freezes different parameters,
                    # making the saved optimizer's parameter groups incompatible.
                    ckpt = torch.load(prev, map_location=str(device), weights_only=False)
                    model.load_state_dict(ckpt['model_state'])
                    print(f"  Loaded Phase {phase-1} model weights: {prev}")
                else:
                    print(f"  [WARN] Phase {phase-1} best checkpoint not found at {prev}, "
                          f"continuing with current weights.")
            train_phase(model, phase, train_files, val_ds, args, device,
                        val_history, phase_losses)

        # Training overview plots
        for viz_fn, viz_name in [
            (lambda: viz_all_phases_combined(phase_losses, args.log_dir),       'all_phases'),
            (lambda: viz_val_metrics_over_training(val_history, args.log_dir),  'val_metrics'),
        ]:
            try:
                viz_fn()
            except Exception as e:
                print(f"  [WARN] viz_{viz_name} failed (non-fatal): {e}")

    else:
        section_header("Section 4: Training SKIPPED")
        best = os.path.join(args.checkpoint_dir, 'phase3', 'best_minADE.pt')
        if not os.path.exists(best):
            best = find_latest_checkpoint(os.path.join(args.checkpoint_dir, 'phase3'))
        if best and os.path.exists(best):
            opt_tmp = optim.AdamW(model.parameters(), lr=1e-4)
            load_checkpoint(model, opt_tmp, best, device=str(device))
            print(f"Loaded: {best}")
        else:
            print("WARNING: no checkpoint found.")

    # =========================================================================
    # Section 5: Prediction visualizations
    # =========================================================================
    section_header("Section 5: Prediction Visualizations")

    best_ckpt = os.path.join(args.checkpoint_dir, 'phase3', 'best_minADE.pt')
    if os.path.exists(best_ckpt):
        opt_tmp = optim.AdamW(model.parameters(), lr=1e-4)
        load_checkpoint(model, opt_tmp, best_ckpt, device=str(device))
        print(f"Loaded best Phase 3 checkpoint for visualization.")

    viz_val_ds = build_dataset(val_files[:5], batch_size=min(args.batch_size, 8), shuffle=False)
    for viz_fn, viz_name in [
        (lambda: viz_predicted_trajectories(model, viz_val_ds, device, args.log_dir, 3, 3),
         'predicted_trajectories'),
        (lambda: viz_goal_candidates(model, viz_val_ds, device, args.log_dir),
         'goal_candidates'),
        (lambda: viz_confidence_and_winner(model, viz_val_ds, device, args.log_dir, 20),
         'confidence_and_winner'),
        (lambda: viz_cv_vs_gbphys(cv, model, viz_val_ds, device, args.log_dir, 4),
         'cv_vs_gbphys'),
    ]:
        try:
            viz_fn()
        except Exception as e:
            print(f"  [WARN] viz_{viz_name} failed (non-fatal): {e}")

    # =========================================================================
    # Section 6: Error analysis
    # =========================================================================
    section_header("Section 6: Error Analysis")

    error_ds = build_dataset(val_files[:10], batch_size=args.batch_size, shuffle=False)
    try:
        viz_error_analysis(model, cv, error_ds, device, args.log_dir, max_batches=50)
    except Exception as e:
        print(f"  [WARN] viz_error_analysis failed (non-fatal): {e}")

    # =========================================================================
    # Section 7: Official Evaluation
    # =========================================================================
    section_header("Section 7: Official Evaluation")

    print(f"Running numpy metrics on {args.eval_batches} val batches...")
    full_val_ds  = build_dataset(val_files, batch_size=args.batch_size, shuffle=False)
    final_metrics = evaluate_full(model, full_val_ds, device, max_batches=args.eval_batches)

    print(f"\n{'='*55}")
    print(f"  {'Metric':<20} {'CV':>12} {'GB-Phys TrajNet':>15}")
    print(f"  {'-'*50}")
    for m_key, m_label in [
        ('minADE', 'minADE (m)'), ('minFDE', 'minFDE (m)'), ('miss_rate', 'Miss Rate')
    ]:
        cv_vals = cv_metrics.get(m_key, [])
        cv_v  = float(np.mean(cv_vals)) if cv_vals else float('nan')
        gbp_v = final_metrics.get(m_key, float('nan'))
        if cv_v > 0 and not np.isnan(gbp_v):
            diff = (cv_v - gbp_v) / cv_v * 100
            print(f"  {m_label:<20} {cv_v:>12.4f} {gbp_v:>15.4f}  ({diff:+.1f}%)")
        else:
            print(f"  {m_label:<20} {cv_v:>12.4f} {gbp_v:>15.4f}")
    print(f"{'='*55}\n")

    try:
        viz_comparison_bar(cv_metrics, final_metrics, args.log_dir)
    except Exception as e:
        print(f"  [WARN] viz_comparison_bar failed (non-fatal): {e}")

    official_result = None
    if WAYMO_METRICS_AVAILABLE:
        section_header("Section 7b: Official Waymo MotionMetrics")
        try:
            config        = _default_metrics_config()
            motion_metrics = MotionMetrics(config)
            official_ds   = build_dataset(val_files, batch_size=args.batch_size, shuffle=False)

            print(f"Accumulating over {args.eval_batches} batches...")
            with torch.no_grad():
                for idx, tf_batch in enumerate(official_ds.take(args.eval_batches)):
                    tb  = tf_to_torch_batch(tf_batch, device=str(device))
                    out = model(tb, phase=3)
                    trajs_sub = out['trajectories'].cpu().numpy()[:, :, :, SUBMISSION_INDICES, :]
                    pred_traj_tf  = tf.expand_dims(tf.constant(trajs_sub, dtype=tf.float32), 3)
                    pred_score_tf = tf.constant(out['confidences'].cpu().numpy(), dtype=tf.float32)
                    gt_traj_tf    = tf.cast(tf_batch['gt_future_states'],   tf.float32)
                    gt_valid_tf   = tf.cast(tf_batch['gt_future_is_valid'], tf.bool)
                    obj_type_tf   = tf.cast(tf_batch['object_type'],        tf.int64)
                    motion_metrics.update_state(pred_traj_tf, pred_score_tf,
                                                gt_traj_tf, gt_valid_tf, obj_type_tf)
                    if (idx + 1) % 50 == 0:
                        print(f"  batch {idx+1}/{args.eval_batches}")

            official_result = motion_metrics.result()
            try:
                from waymo_open_dataset.metrics.python import config_util_py as cu
                breakdown_names = cu.get_breakdown_names_from_motion_config(config)
            except Exception:
                breakdown_names = [f'bucket_{j}' for j in range(official_result.shape[1])]

            METRIC_TYPES = ['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'SoftmAP']
            print("\nOfficial MotionMetrics:")
            for i, mtype in enumerate(METRIC_TYPES):
                vals = [float(official_result[i, j]) for j in range(official_result.shape[1])]
                print(f"  {mtype:<15} avg={np.mean(vals):.4f}  "
                      f"per-bucket={[f'{v:.3f}' for v in vals]}")

            try:
                viz_official_metrics_heatmap(official_result, breakdown_names, args.log_dir)
            except Exception as e:
                print(f"  [WARN] viz_official_metrics_heatmap failed (non-fatal): {e}")

        except Exception as e:
            print(f"  [WARN] Official MotionMetrics failed (non-fatal): {e}")
            print("  Continuing to submission generation...")

    # =========================================================================
    # Section 8: Submission
    # =========================================================================
    section_header("Section 8: Submission Generation")

    submission_path = os.path.join(args.submission_dir, 'submission_test.pb')
    if WAYMO_METRICS_AVAILABLE:
        try:
            test_files = sorted(tf.io.gfile.glob(GCS_TEST))[:20]
            test_ds    = build_dataset(test_files, batch_size=args.batch_size, shuffle=False)
            create_submission(model, test_ds, submission_path,
                              device=str(device), max_batches=100)
            if os.path.exists(submission_path):
                print(f"Submission: {submission_path} "
                      f"({os.path.getsize(submission_path)/1e6:.2f} MB)")
        except Exception as e:
            print(f"  [WARN] Submission generation failed (non-fatal): {e}")
    else:
        print("  Skipping submission generation — waymo SDK not available.")

    # =========================================================================
    # Summary
    # =========================================================================
    section_header("PIPELINE COMPLETE")

    # List all generated figures
    all_figs = sorted([f for f in os.listdir(args.log_dir) if f.endswith('.png')])
    print(f"Generated {len(all_figs)} figures in {args.log_dir}/:")
    for fname in all_figs:
        size = os.path.getsize(os.path.join(args.log_dir, fname)) / 1024
        print(f"  {fname:<50s}  ({size:.0f} KB)")

    print(f"\nCheckpoints: {args.checkpoint_dir}/phase{{1,2,3}}/best_minADE.pt")
    print(f"Submission:  {submission_path}")
    print("\nTo copy everything back locally:")
    print(f"  rsync -avP <cluster>:{args.checkpoint_dir}/ ./checkpoints/")
    print(f"  rsync -avP <cluster>:{args.log_dir}/*.png  ./logs/")


if __name__ == '__main__':
    main()
